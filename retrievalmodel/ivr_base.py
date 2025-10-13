import os
import re
import sys
import torch
import json
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from .ivr_utils import compute_recall_metrics, compute_hit_metrics, generate_caption_vqa, generate_augment_caption,\
    get_cosine_similarity, get_cur_retrieval, generate_question_auto, generate_caption_vqa_auto, compute_metrics_multiturn, video_llava_plain_text_qa,\
    generate_augment_caption_auto, get_video_features, get_conjunctions, video_llava_qa_short, generate_object_caption, video_llava_get_video_caption,\
    generate_videomateinfo_reponse, get_img_features, video_llava_qa_short_image
from .qcs import matching_score, compute_semantic_uncertainty
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    
logger = logging.getLogger(__name__)
        
def get_text_features(text, model, device, retrieval_tokenizer):
    tokens = retrieval_tokenizer([text], truncation=True)
    input_ids = torch.tensor(tokens['input_ids'], dtype=torch.long).to(device)
    with torch.inference_mode():
        outputs = model.get_video_tower().video_tower_text_encoder(input_ids=input_ids, output_hidden_states=True)
        text_pooler_output = outputs.pooler_output
        text_features = model.get_model().video_tower.retrieval_text_proj(text_pooler_output)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().detach().numpy()

def ivr_heuristic_multiturn(test_video_dataloader, test_text_dataloader, config,
                            model, retrieval_tokenizer, generation_tokenizer, device, *args, **kwargs):
    all_video_features, all_video_pixels, all_video_path = get_video_features(test_video_dataloader, model, device)
    dialogue_manager = ivrHeuristicDialogueManager(config, model, generation_tokenizer, device)
    total_turn = dialogue_manager.get_total_turn()
    all_text_features = []
    text_bar = tqdm(total=len(test_text_dataloader), file=sys.stdout, position=0)
    for i, text_data in enumerate(test_text_dataloader):
        video_pixel_features = all_video_pixels[i:i+1]
        text_data = text_data[0]
        # cur_case_query_list = [text_data]
        cur_case_text_features = [get_text_features(text_data, model, device, retrieval_tokenizer)]
        video_tensor = torch.tensor(video_pixel_features, dtype=torch.float16).to(device)
        dialogue_manager.reset_state(initial_query=text_data, video_pixel_features=video_tensor)
        for j in range(total_turn):
            question = dialogue_manager.do_question_asking()
            dialogue_manager.do_qa_generation(question)
            new_caption = dialogue_manager.assemble_caption()
            dialogue_manager.add_to_query_list(new_caption)
            query_list = dialogue_manager.get_query_list()
            new_query_text = ' and '.join(query_list)
            cur_case_text_features.append(get_text_features(new_query_text, model, device, retrieval_tokenizer))
            dialogue_manager.update_state()
            logger.info(f"in {i}th video, round {j}, the question is{question} the text query is {new_query_text}")
        all_text_features.append(np.concatenate(cur_case_text_features, axis=0))
        text_bar.update(1)
    compute_metrics_multiturn(all_text_features, all_video_features, total_turn)
        
# UMIVR
def ua_multiturn(test_video_dataloader, test_text_dataloader, config,
                 model, retrieval_tokenizer, generation_tokenizer, device, *args, **kwargs):
    all_video_features, all_video_pixels, all_video_path = get_video_features(test_video_dataloader, model, device)
    dialogue_manager = uaAutoDialogueManager(config, model, generation_tokenizer, retrieval_tokenizer, device, all_video_features, all_video_pixels, all_video_path)
    total_turn = dialogue_manager.total_turn
    all_text_features = []
    text_bar = tqdm(total=len(test_text_dataloader), file=sys.stdout, position=0)
    for i, text_data in enumerate(test_text_dataloader):
        text_bar.update(1)
        video_pixel_features = all_video_pixels[i:i+1]
        text_data = text_data[0]
        cur_turn_text_features = [get_text_features(text_data, model, device, retrieval_tokenizer)]
        video_tensor = torch.tensor(video_pixel_features, dtype=torch.float16).to(device)
        dialogue_manager.reset_state(initial_query=text_data, video_pixel_features=video_tensor)
        for j in range(total_turn):
            logger.info(f"in {i}th video, round {j}")
            text_query = dialogue_manager.do_caption_generation()
            cur_turn_text_features.append(get_text_features(text_query, model, device, retrieval_tokenizer))
        all_text_features.append(np.concatenate(cur_turn_text_features, axis=0))
    
    compute_metrics_multiturn(all_text_features, all_video_features, total_turn)
            
def ivr_auto_multiturn(test_video_dataloader, test_text_dataloader, config,
                       model, retrieval_tokenizer, generation_tokenizer, device, *args, **kwargs):
    all_video_features, all_video_pixels, all_video_path = get_video_features(test_video_dataloader, model, device)
    use_caption_cache = config['dialogue_config'].get('use_caption_cache', False)
    use_caption = config['dialogue_config'].get('use_caption', False)
    video_id_to_caption = None
    if use_caption and use_caption_cache:
        caption_cache_file = config['dialogue_config'].get('caption_cache_file', None)
        assert caption_cache_file is not None, 'Caption cache file not found'
        if not os.path.exists(caption_cache_file):
            video_id_to_caption = {}
            print('Generating captions for video')
            caption_bar = tqdm(total=len(test_text_dataloader), file=sys.stdout, position=0)
            for i, text_data in enumerate(test_text_dataloader):
                cur_video_pixels = all_video_pixels[i:i+1]
                cur_video_caption = video_llava_get_video_caption(cur_video_pixels, model, device, config['model_config'], generation_tokenizer)
                video_id_to_caption[i] = cur_video_caption
                caption_bar.update(1)
            caption_bar.close()
            with open(caption_cache_file, 'w') as f:
                json.dump(video_id_to_caption, f)
        else:
            with open(caption_cache_file, 'r') as f:
                video_id_to_caption = {int(k): v for k, v in json.load(f).items()}
    extend_dialogue = config['dialogue_config'].get('extend_dialogue', False)
    wo_aug_dialogue = config['dialogue_config'].get('wo_aug_dialogue', False)
    if wo_aug_dialogue:
        dialogue_manager = ivrAutoDialogueManagerWOAug(config, model, generation_tokenizer, retrieval_tokenizer, device, all_video_features, all_video_pixels, video_id_to_caption=video_id_to_caption)
    elif not extend_dialogue:
        dialogue_manager = ivrAutoDialogueManagerShort(config, model, generation_tokenizer, retrieval_tokenizer, device, all_video_features, all_video_pixels, video_id_to_caption=video_id_to_caption)
    else:
        logger.info('Using extended dialogue manager')
        dialogue_manager = ivrAutoDialogueManagerExtended(config, model, generation_tokenizer, retrieval_tokenizer, device, all_video_features, all_video_pixels, video_id_to_caption=video_id_to_caption)
    total_turn = dialogue_manager.total_turn
    all_text_features = []
    text_bar = tqdm(total=len(test_text_dataloader), file=sys.stdout, position=0)
    for i, text_data in enumerate(test_text_dataloader):
        video_pixel_features = all_video_pixels[i:i+1]
        text_data = text_data[0]
        cur_case_text_features = [get_text_features(text_data, model, device, retrieval_tokenizer)]
        video_tensor = torch.tensor(video_pixel_features, dtype=torch.float16).to(device)
        dialogue_manager.reset_state(initial_query=text_data, video_pixel_features=video_tensor)
        for j in range(total_turn):
            text_query = dialogue_manager.do_qa_generation_caption()
            cur_case_text_features.append(get_text_features(text_query, model, device, retrieval_tokenizer))
        logger.info(f"in {i}th video, the final captions are {dialogue_manager.query_list}")
        all_text_features.append(np.concatenate(cur_case_text_features, axis=0))
        text_bar.update(1)
    compute_metrics_multiturn(all_text_features, all_video_features, total_turn)

class uaAutoDialogueManager:
    def __init__(self, config, model, generation_tokenizer, retrieval_tokenizer, device, all_video_features, all_video_pixels,all_video_path, *args, **kwargs):
        self.state = 0
        self.max_state = 10
        self.initial_query = ""
        self.dialogue_history = []
        dialogue_config = config.get('dialogue_config', None)
        assert dialogue_config is not None, 'Dialogue config not found'
        total_turn = dialogue_config.get('total_turn', -1)
        assert total_turn <= self.max_state, 'Total turn should be less than or equal to the max state'
        if total_turn == -1:
            total_turn = self.max_state
        self.total_turn = total_turn
        self.dialogue_config = dialogue_config
        self.jsd_threshold = dialogue_config.get('jsd_threshold', 0.2)
        self.text_uncertainty_threshold = dialogue_config.get('text_uncertainty_threshold', 0.5)
        model_config = config.get('model_config', None)
        self.model_config = model_config
        assert model_config is not None, 'Model config not found'
        self.model = model
        self.generation_tokenizer = generation_tokenizer
        self.retrieval_tokenizer = retrieval_tokenizer
        self.retrieval_max_tokens = retrieval_tokenizer.model_max_length
        self.device = device
        self.all_video_features = all_video_features
        self.all_video_pixels = all_video_pixels
        self.all_video_path = all_video_path
        self.video_meta_info = self.generate_video_metadata()
        text_corpus, corpus_embeddings = self.get_embeding_text(self.video_meta_info)
        self.text_corpus = text_corpus
        self.corpus_embeddings = corpus_embeddings
        
    def reset_state(self, initial_query, video_pixel_features):
        self.state = 0
        self.level = 0
        self.initial_query = initial_query
        self.video_pixels = video_pixel_features
        self.query_list = [initial_query]
        self.dialogue_history = []
         
    def generate_video_metadata(self):
        meta_data_file = self.dialogue_config.get('metadata_file', None)
        assert meta_data_file is not None, 'Metadata file config not found'
        if not os.path.exists(meta_data_file):
            total_videos = len(self.all_video_pixels)
            video_bar = tqdm(total=total_videos, file=sys.stdout, position=0)
            video_mata_info = []
            print('Generating video metadata')
            for i in range(total_videos):
                video_bar.update(1)
                cur_video_pixels = self.all_video_pixels[i:i+1]
                cur_meta_info = self.get_cur_video_metadata(cur_video_pixels)
                cur_video_name = os.path.basename(self.all_video_path[i])
                cur_meta_info['video_name'] = cur_video_name
                video_mata_info.append(cur_meta_info)
            video_bar.close()
            with open(meta_data_file, 'w') as f:
                json.dump(video_mata_info, f)
        else:
            with open(meta_data_file, 'r') as f:
                video_mata_info = json.load(f)
        return video_mata_info
    
    def get_embeding_text(self, video_meta_info):
        embedding_file = self.dialogue_config.get('embedding_file', None)
        assert embedding_file is not None, 'Embedding file not defined'
        text_corpus_file = self.dialogue_config.get('text_corpus_file', None)
        assert text_corpus_file is not None, 'Text corpus file not defined'
        if os.path.exists(embedding_file) and os.path.exists(text_corpus_file):
            corpus_embeddings = torch.load(embedding_file, map_location=torch.device('cpu'))
            with open(text_corpus_file, 'r') as f:
                text_corpus = json.load(f)
        else:
            text_corpus = []
            for cur_anno in video_meta_info:
                text_corpus.append(cur_anno['caption'])
            with open(text_corpus_file, 'w') as f:
                json.dump(text_corpus, f)
            corpus_embeddings = self.embedding_text_corpus(text_corpus)
        return text_corpus, corpus_embeddings
                
    def embedding_text_corpus(self, text_corpus, batchsize=16):
        embeddings_list = []
        print("Start embedding texts ...")
        for i in tqdm(range(0, len(text_corpus), batchsize), desc="Embedding texts"):
            batch_text = text_corpus[i:i+batchsize]
            tokens = self.retrieval_tokenizer(batch_text, truncation=True, padding=True)
            input_ids = torch.tensor(tokens['input_ids'],  dtype=torch.long).to(self.device)
            with torch.inference_mode():
                outputs = self.model.get_video_tower().video_tower_text_encoder(input_ids=input_ids, output_hidden_states=True)
                text_pooler_output = outputs.pooler_output
                text_features = self.model.get_model().video_tower.retrieval_text_proj(text_pooler_output)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            embeddings_list.append(text_features.cpu())
        
        all_embeddings = torch.cat(embeddings_list, dim=0)
        print(f"All embeddings shape: {all_embeddings.shape}")
        embedding_file = self.dialogue_config.get('embedding_file', None)
        torch.save(all_embeddings, embedding_file)
        return all_embeddings

    def get_cur_video_metadata(self, cur_video_pixels):
        system_prompt = \
"""A conversation between a curious human and an AI assistant. The assistant is specialized in analyzing video content and provides detailed, precise, and evidence-based descriptions. Follow these guidelines strictly:
- **Precision**: Describe only what is directly observable from the video.
- **Detail**: Include all readily visible details while keeping responses focused.
- **No Speculation**: If any part of the content is uncertain, explicitly state the uncertainty instead of guessing.
"""
        video_rep = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames)
        caption_prompt = (
            f"{video_rep}\n"
            "Please provide a detailed and highly accurate caption that fully describes the overall scene or main activity in this video. Make sure your caption includes all relevant visual details and does not exceed 80 words. Do not add any information that is not clearly supported by the video content."
        )
        main_objects_prompt = (
            f"{video_rep}\n"
            "Based solely on the visible content of the video, list up to five primary objects or characters you can clearly identify. Each item should be provided as a single word or a brief noun phrase (e.g., 'man', 'tree', 'couch'). Only include items that are explicitly visible and avoid any speculation."
        )
        scene_type_prompt = (
            f"{video_rep}\n"
            "Based on the visual content of the video, identify the primary setting, scene type, or dominant visual theme by listing up to five concise keywords (e.g., 'underwater', 'indoor', 'black'). Only include keywords that are directly evident from the video, and do not include any speculative information."
        )
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        tensor = torch.tensor(cur_video_pixels, dtype=torch.float16).to(self.device)
        
        caption_conv = conv_templates[conv_mode].copy()
        caption_conv.system = system_prompt
        caption = generate_videomateinfo_reponse(caption_conv, caption_prompt, tensor, self.generation_tokenizer, self.model, self.device)
        
        main_objects_conv = conv_templates[conv_mode].copy()
        main_objects_conv.system = system_prompt
        main_objects = generate_videomateinfo_reponse(main_objects_conv, main_objects_prompt, tensor, self.generation_tokenizer, self.model, self.device)
        
        scene_type_conv = conv_templates[conv_mode].copy()
        scene_type_conv.system = system_prompt
        scene_type = generate_videomateinfo_reponse(scene_type_conv, scene_type_prompt, tensor, self.generation_tokenizer, self.model, self.device)
        
        return {"caption": caption, "main_objects": main_objects, "scene_type": scene_type}   
    
    def get_query_for_retrieval(self):
        initial_query = self.query_list[0]
        if len(self.query_list) > 1:
            final_query = self.query_list[-1]
            return ' and '.join([initial_query, final_query])
        return initial_query
    
    def get_query_uncertainty(self):
        pre_query_text = self.query_list[-1]
        text_uncertainty = compute_semantic_uncertainty(pre_query_text, self.text_corpus, self.corpus_embeddings, self.model, self.retrieval_tokenizer, self.device, top_k=5, sim_threshold=0.65)
        return text_uncertainty
    
    def set_uncertainty_level(self, normalized_jsd):
        cur_state = self.state
        text_uncertainty = self.get_query_uncertainty()
        jsd_threshold = self.jsd_threshold
        text_uncertainty_threshold = self.text_uncertainty_threshold
        if self.level == 0:
            if cur_state == 0 or (text_uncertainty >= text_uncertainty_threshold):
                self.level = 0
            elif (text_uncertainty < text_uncertainty_threshold) and (normalized_jsd >= jsd_threshold):
                self.level = 1
            elif (text_uncertainty < text_uncertainty_threshold) and (normalized_jsd < jsd_threshold):
                self.level = 2
            else:
                self.level = 0
        elif self.level == 1:
            self.level = 2  
    
    def get_uncertainty_level(self):
        return self.level
    
    def do_caption_generation(self, threshold=0.02):
        if self.state >= self.total_turn:
            raise ValueError('Total turn reached')
        pre_query_text = self.query_list[-1]
        inds, topk_scores = get_cur_retrieval([pre_query_text], self.all_video_features, self.model, self.device, self.retrieval_tokenizer, return_scores=True)
        
        topk_scores = topk_scores.squeeze()
        
        p_dist, normalized_jsd = matching_score(topk_scores)
        self.set_uncertainty_level(normalized_jsd)
        
        cur_meta_info = [{k: v for k, v in self.video_meta_info[i].items() if k != 'video_name'} for i in inds]
        
        filtered_meta_info = [meta for meta, p in zip(cur_meta_info, p_dist) if p >= threshold]
        
        question = self.do_question_generation(filtered_meta_info)
        
        answer = self.generate_answewr(question)
        
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")
        
        self.dialogue_history.append({"round": self.state + 1, "question": question, "answer": answer})
        
        final_query = self.get_final_query()
        
        self.query_list.append(final_query)
        
        self.state += 1
        
        return self.get_query_for_retrieval()
    
    def get_final_query(self):
        pre_query = self.query_list[-1]
        cur_token_count = len(self.retrieval_tokenizer.tokenize(pre_query))
        remaining_tokens = self.retrieval_max_tokens - cur_token_count
        if remaining_tokens < 10:
            return self.concise_caption()
        cur_answer = self.dialogue_history[-1]['answer']
        return ' and '.join([cur_answer, self.query_list[-1]])
    
    def concise_caption(self):
        pre_query = self.query_list[-1]
        system_prompt = (
            "You are an expert at simplifying long text queries without losing important details. "
            "When simplifying, ensure that all key information is preserved—especially visual information such as objects, colors, and events. "
            "The simplified text should be more concise than the original but not overly short."
        )
        user_prompt = \
f"""
Original User Query:
{pre_query}

Simplify the above query by shortening it while preserving all details, especially visual information (e.g., objects, colors, events). 
Ensure that the simplified version is concise yet comprehensive, and does not become overly short.

Only return the simplified query, nothing else.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        concise_conv = conv_templates[conv_mode].copy()
        concise_conv.system = system_prompt
        concise_conv.append_message(concise_conv.roles[0], user_prompt)
        concise_conv.append_message(concise_conv.roles[1], None)
        prompt = concise_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = concise_conv.sep if concise_conv.sep_style != SeparatorStyle.TWO else concise_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        cur_answer = self.dialogue_history[-1]['answer']
        new_query = ' and '.join([outputs, cur_answer])
        # self.query_list.pop()
        return new_query
    
    def do_systhesis_caption(self):
        pre_query = self.query_list[-1]
        cur_answer = self.dialogue_history[-1]['answer']
        system_prompt = (
            "You are an expert in query refinement for interactive text-video retrieval. "
            "Your task is to synthesize and update a previous query with new details from the current answer. "
            "Ensure the new query includes key information (e.g., characters, events, objects, colors, locations) and does not exceed 60 words."
        )
        user_prompt = \
f"""
Previous Query:
{pre_query}

Current Answer (includes new information to enhance video retrieval):
{cur_answer}

Combine the above into one concise, positive declarative sentence that includes key details (characters, events, objects, colors, locations, etc.). 
Ensure the new query leverages the new information from the current answer for better retrieval and is no longer than 60 words.

Only return the refined query, nothing else.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        synthesis_conv = conv_templates[conv_mode].copy()
        synthesis_conv.system = system_prompt
        synthesis_conv.append_message(synthesis_conv.roles[0], user_prompt)
        synthesis_conv.append_message(synthesis_conv.roles[1], None)
        prompt = synthesis_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = synthesis_conv.sep if synthesis_conv.sep_style != SeparatorStyle.TWO else synthesis_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        outputs = outputs.split("Final Query:", 1)[-1].strip().strip("\"'*")
        # print(outputs)
        return outputs
        
    def do_question_generation(self,cur_meta_info):
        ut_lvl = self.get_uncertainty_level()
        if ut_lvl == 0:
            question = self.coarse_grained_question_generation()
        elif ut_lvl == 1:
            question = self.middle_grained_question_generation(cur_meta_info)
        else:
            question = self.fine_grained_question_generation()
        question =  re.sub(r'^[^a-zA-Z0-9]*','', question).replace('*','')
        question = question.strip('\'"')
        return question
    
    def get_answer_system_prompt(self):
        ut_lvl = self.get_uncertainty_level()
        if ut_lvl == 0:
            system_prompt = (
                "You are an expert in video content analysis and question answering. "
                "When provided with a video description and a question, focus on the video's visual and contextual details to craft a concise, one-sentence answer. "
                "Ensure your answer is accurate, centers on key visual information, and avoids overly verbose explanations or simple 'yes'/'no' responses."
            )
        else:
            system_prompt = (
                "You are a video question answering assistant. "
                "When provided with a video and a question, your task is to provide a concise, one-sentence answer. "
                "Your answer should clearly state the key visual details such as people, objects, scenes, and events. "
                "Keep it clear, direct, and focused on essential information."
            )
        return system_prompt
    
    def get_answer_user_prompt(self, video_rep, question):
        ut_lvl = self.get_uncertainty_level()
        if ut_lvl == 0:
            user_prompt = \
f"""
{video_rep}

Question:
{question}

Please provide a concise one-sentence answer that accurately reflects the key visual information in the video with respect to the question.
"""
        else:
            user_prompt = \
f"""
{video_rep}

Question:
{question}

Provide a one-sentence answer that clearly identifies the key visual details in the video, such as people, objects, scenes, and events.
"""
        return user_prompt

    def generate_answewr(self, question):
        video_rep = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames)
        
        user_prompt = self.get_answer_user_prompt(video_rep, question)
        system_prompt = self.get_answer_system_prompt()
        
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        answer_conv = conv_templates[conv_mode].copy()
        if system_prompt is not None:
            answer_conv.system = system_prompt
        answer_conv.append_message(answer_conv.roles[0], user_prompt)
        answer_conv.append_message(answer_conv.roles[1], None)
        tensor = self.video_pixels
        prompt = answer_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = answer_conv.sep if answer_conv.sep_style != SeparatorStyle.TWO else answer_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        return outputs
    
    def middle_grained_question_generation(self,cur_meta_info): 
        cur_text_query = self.query_list[-1]
        middle_system_prompt = (
            "You are a clarifying question generator for text-video retrieval. "
            "Given a user query and multiple video info, your task is to generate one question "
            "that focuses on visual differences. The question must start with What, Where, or Who."
        )
        middle_user_prompt = \
f"""
Query: "{cur_text_query}"
Videos: {cur_meta_info}

Ask one question starting with What, Where, or Who to distinguish these videos based on visual details.
Return ONLY the question.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        middle_grained_conv = conv_templates[conv_mode].copy()
        middle_grained_conv.system = middle_system_prompt
        middle_grained_conv.append_message(middle_grained_conv.roles[0], middle_user_prompt)
        middle_grained_conv.append_message(middle_grained_conv.roles[1], None)
        prompt = middle_grained_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = middle_grained_conv.sep if middle_grained_conv.sep_style != SeparatorStyle.TWO else middle_grained_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        return outputs

    
    def coarse_grained_question_generation(self):
        cur_text_query = self.query_list[-1]
        coarse_grained_system_prompt = (
            "You are an advanced AI specialized in asking clarifying questions for vague queries. "
            "Your task is to extract details—such as appearance, activities, or events—to enable precise retrieval."
        )
        user_prompt = \
f"""
Query: "{cur_text_query}"
Ask one open-ended clarifying question focusing on the subject's appearance, activities, or events.
Return ONLY the question.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        coarse_grained_conv = conv_templates[conv_mode].copy()
        coarse_grained_conv.system = coarse_grained_system_prompt
        coarse_grained_conv.append_message(coarse_grained_conv.roles[0], user_prompt)
        coarse_grained_conv.append_message(coarse_grained_conv.roles[1], None)
        prompt = coarse_grained_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = coarse_grained_conv.sep if coarse_grained_conv.sep_style != SeparatorStyle.TWO else coarse_grained_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        s = outputs.strip()
        return s

    def fine_grained_question_generation(self):
        cur_text_query = self.query_list[-1]
        user_prompt = \
f"""
You need to ask a question based on a user query.
1. First you need to evaluate whether the user's query includes sufficient visual details (such as characters, colors, objects, or locations).
User Query: "{cur_text_query}"

2. Ask a question
    - If details are missing, generate one question to gather them.
    - If the query is already detailed, generate a clarifying question to further enrich the description (e.g., 'What other objects are present?', 'What is the main color?', or 'Where is the event taking place?').


Return ONLY the question, nothing else.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        fine_grained_conv = conv_templates[conv_mode].copy()
        # fine_grained_conv.system = fine_grained_system_prompt
        fine_grained_conv.append_message(fine_grained_conv.roles[0], user_prompt)
        fine_grained_conv.append_message(fine_grained_conv.roles[1], None)
        prompt = fine_grained_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = fine_grained_conv.sep if fine_grained_conv.sep_style != SeparatorStyle.TWO else fine_grained_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        return outputs
    
class uaAutoImageManager:
    def __init__(self, config, model, generation_tokenizer, retrieval_tokenizer, device, all_video_features, all_video_pixels,all_video_path, *args, **kwargs):
        self.state = 0
        self.max_state = 10
        self.initial_query = ""
        self.dialogue_history = []
        dialogue_config = config.get('dialogue_config', None)
        assert dialogue_config is not None, 'Dialogue config not found'
        total_turn = dialogue_config.get('total_turn', -1)
        assert total_turn <= self.max_state, 'Total turn should be less than or equal to the max state'
        if total_turn == -1:
            total_turn = self.max_state
        self.total_turn = total_turn
        self.dialogue_config = dialogue_config
        self.jsd_threshold = dialogue_config.get('jsd_threshold', 0.2)
        self.text_uncertainty_threshold = dialogue_config.get('text_uncertainty_threshold', 0.5)
        model_config = config.get('model_config', None)
        self.model_config = model_config
        assert model_config is not None, 'Model config not found'
        self.model = model
        self.generation_tokenizer = generation_tokenizer
        self.retrieval_tokenizer = retrieval_tokenizer
        self.retrieval_max_tokens = retrieval_tokenizer.model_max_length
        self.device = device
        self.all_video_features = all_video_features
        self.all_video_pixels = all_video_pixels
        self.all_video_path = all_video_path
        self.video_meta_info = self.generate_image_metadata()
        text_corpus, corpus_embeddings = self.get_embeding_text(self.video_meta_info)
        self.text_corpus = text_corpus
        self.corpus_embeddings = corpus_embeddings
        print(len(self.video_meta_info))
        
    def reset_state(self, initial_query, video_pixel_features):
        self.state = 0
        self.level = 0
        self.initial_query = initial_query
        self.video_pixels = video_pixel_features
        self.query_list = [initial_query]
        self.dialogue_history = []
         
    def generate_image_metadata(self):
        meta_data_file = self.dialogue_config.get('metadata_file', None)
        assert meta_data_file is not None, 'Metadata file config not found'
        if not os.path.exists(meta_data_file):
            total_videos = len(self.all_video_pixels)
            video_bar = tqdm(total=total_videos, file=sys.stdout, position=0)
            video_mata_info = []
            print('Generating video metadata')
            for i in range(total_videos):
                video_bar.update(1)
                cur_video_pixels = self.all_video_pixels[i:i+1]
                cur_meta_info = self.get_cur_video_metadata(cur_video_pixels)
                cur_video_name = os.path.basename(self.all_video_path[i])
                cur_meta_info['video_name'] = cur_video_name
                video_mata_info.append(cur_meta_info)
            video_bar.close()
            with open(meta_data_file, 'w') as f:
                json.dump(video_mata_info, f)
        else:
            with open(meta_data_file, 'r') as f:
                video_mata_info = json.load(f)
        return video_mata_info
    
    def get_embeding_text(self, video_meta_info):
        embedding_file = self.dialogue_config.get('embedding_file', None)
        assert embedding_file is not None, 'Embedding file not defined'
        text_corpus_file = self.dialogue_config.get('text_corpus_file', None)
        assert text_corpus_file is not None, 'Text corpus file not defined'
        if os.path.exists(embedding_file) and os.path.exists(text_corpus_file):
            corpus_embeddings = torch.load(embedding_file, map_location=torch.device('cpu'))
            with open(text_corpus_file, 'r') as f:
                text_corpus = json.load(f)
        else:
            text_corpus = []
            for cur_anno in video_meta_info:
                text_corpus.append(cur_anno['caption'])
            with open(text_corpus_file, 'w') as f:
                json.dump(text_corpus, f)
            corpus_embeddings = self.embedding_text_corpus(text_corpus)
        return text_corpus, corpus_embeddings
                
    def embedding_text_corpus(self, text_corpus, batchsize=16):
        embeddings_list = []
        print("Start embedding texts ...")
        for i in tqdm(range(0, len(text_corpus), batchsize), desc="Embedding texts"):
            batch_text = text_corpus[i:i+batchsize]
            tokens = self.retrieval_tokenizer(batch_text, truncation=True, padding=True)
            input_ids = torch.tensor(tokens['input_ids'],  dtype=torch.long).to(self.device)
            with torch.inference_mode():
                outputs = self.model.get_video_tower().video_tower_text_encoder(input_ids=input_ids, output_hidden_states=True)
                text_pooler_output = outputs.pooler_output
                text_features = self.model.get_model().video_tower.retrieval_text_proj(text_pooler_output)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            embeddings_list.append(text_features.cpu())
        
        all_embeddings = torch.cat(embeddings_list, dim=0)
        print(f"All embeddings shape: {all_embeddings.shape}")
        embedding_file = self.dialogue_config.get('embedding_file', None)
        torch.save(all_embeddings, embedding_file)
        return all_embeddings

    def get_cur_video_metadata(self, cur_video_pixels):
        system_prompt = \
"""A conversation between a curious human and an AI assistant. The assistant is specialized in analyzing image content and provides detailed, precise, and evidence-based descriptions. Follow these guidelines strictly:
- **Precision**: Describe only what is directly observable from the iamge.
- **Detail**: Include all readily visible details while keeping responses focused.
- **No Speculation**: If any part of the content is uncertain, explicitly state the uncertainty instead of guessing.
"""
        video_rep = ' '.join([DEFAULT_IMAGE_TOKEN])
        caption_prompt = (
            f"{video_rep}\n"
            "Please provide a detailed and highly accurate caption that fully describes the overall scene or main activity in this image. Make sure your caption includes all relevant visual details and does not exceed 80 words. Do not add any information that is not clearly supported by the image content."
        )
        main_objects_prompt = (
            f"{video_rep}\n"
            "Based solely on the visible content of the image, list up to five primary objects or characters you can clearly identify. Each item should be provided as a single word or a brief noun phrase (e.g., 'man', 'tree', 'couch'). Only include items that are explicitly visible and avoid any speculation."
        )
        scene_type_prompt = (
            f"{video_rep}\n"
            "Based on the visual content of the image, identify the primary setting, scene type, or dominant visual theme by listing up to five concise keywords (e.g., 'underwater', 'indoor', 'black'). Only include keywords that are directly evident from the image, and do not include any speculative information."
        )
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        tensor = torch.tensor(cur_video_pixels, dtype=torch.float16).to(self.device)
        
        caption_conv = conv_templates[conv_mode].copy()
        caption_conv.system = system_prompt
        caption = generate_videomateinfo_reponse(caption_conv, caption_prompt, tensor, self.generation_tokenizer, self.model, self.device)
        
        main_objects_conv = conv_templates[conv_mode].copy()
        main_objects_conv.system = system_prompt
        main_objects = generate_videomateinfo_reponse(main_objects_conv, main_objects_prompt, tensor, self.generation_tokenizer, self.model, self.device)
        
        scene_type_conv = conv_templates[conv_mode].copy()
        scene_type_conv.system = system_prompt
        scene_type = generate_videomateinfo_reponse(scene_type_conv, scene_type_prompt, tensor, self.generation_tokenizer, self.model, self.device)
        
        return {"caption": caption, "main_objects": main_objects, "scene_type": scene_type}   
    
    def get_query_for_retrieval(self):
        initial_query = self.query_list[0]
        if len(self.query_list) > 1:
            final_query = self.query_list[-1]
            return ' and '.join([initial_query, final_query])
        return initial_query
    
    def get_query_uncertainty(self):
        pre_query_text = self.query_list[-1]
        text_uncertainty = compute_semantic_uncertainty(pre_query_text, self.text_corpus, self.corpus_embeddings, self.model, self.retrieval_tokenizer, self.device, top_k=5, sim_threshold=0.65)
        return text_uncertainty
    
    def set_uncertainty_level(self, normalized_jsd):
        cur_state = self.state
        text_uncertainty = self.get_query_uncertainty()
        logger.info(f"Uncertainty Score: {text_uncertainty}")
        jsd_threshold = self.jsd_threshold
        text_uncertainty_threshold = self.text_uncertainty_threshold
        if self.level == 0:
            # if cur_state == 0 or (cur_state < 2 and text_uncertainty >= text_uncertainty_threshold):
            if cur_state == 0 or (text_uncertainty >= text_uncertainty_threshold):
                self.level = 0
            elif (text_uncertainty < text_uncertainty_threshold) and (normalized_jsd >= jsd_threshold):
                self.level = 1
            elif (text_uncertainty < text_uncertainty_threshold) and (normalized_jsd < jsd_threshold):
                self.level = 2
            else:
                self.level = 0
        elif self.level == 1:
            self.level = 2  
    
    def get_uncertainty_level(self):
        return self.level
    
    def do_caption_generation(self, threshold=0.02):
        if self.state >= self.total_turn:
            raise ValueError('Total turn reached')
        pre_query_text = self.query_list[-1]
        logger.info(f"Current query: {pre_query_text}")
        # text_uncertainty = compute_overall_uncertainty(pre_query_text, lambda_param=0.7, delta=0.3, L_min=8)
        # logger.info(f"Uncertainty Score: {text_uncertainty}")
        inds, topk_scores = get_cur_retrieval([pre_query_text], self.all_video_features, self.model, self.device, self.retrieval_tokenizer, return_scores=True)
        
        topk_scores = topk_scores.squeeze()
        
        p_dist, normalized_jsd = matching_score(topk_scores)
        self.set_uncertainty_level(normalized_jsd)
        logger.info(f"lvl:{self.get_uncertainty_level()} inds: {inds}, softmax prob: {p_dist}, normalized jsd: {normalized_jsd}")
        
        cur_meta_info = [{k: v for k, v in self.video_meta_info[i].items() if k != 'video_name'} for i in inds]
        
        filtered_meta_info = [meta for meta, p in zip(cur_meta_info, p_dist) if p >= threshold]
        
        question = self.do_question_generation(filtered_meta_info)
        
        answer = self.generate_answewr(question)
        
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")
        
        self.dialogue_history.append({"round": self.state + 1, "question": question, "answer": answer})
        
        final_query = self.get_final_query()
        
        self.query_list.append(final_query)
        
        self.state += 1
        
        return self.get_query_for_retrieval()
    
    def get_final_query(self):
        lvl = self.get_uncertainty_level()
        # if lvl == 0:
        #     return self.do_systhesis_caption()
        # if self.state >= 2:
        #     pre_query = self.query_list[-1]
        #     cur_token_count = len(self.retrieval_tokenizer.tokenize(pre_query))
        #     remaining_tokens = self.retrieval_max_tokens - cur_token_count
        #     if remaining_tokens < 15:
        #         return self.concise_caption()
        pre_query = self.query_list[-1]
        cur_token_count = len(self.retrieval_tokenizer.tokenize(pre_query))
        remaining_tokens = self.retrieval_max_tokens - cur_token_count
        if remaining_tokens < 10:
            return self.concise_caption()
        cur_answer = self.dialogue_history[-1]['answer']
        return ' and '.join([cur_answer, self.query_list[-1]])
    
    def concise_caption(self):
        pre_query = self.query_list[-1]
        system_prompt = (
            "You are an expert at simplifying long text queries without losing important details. "
            "When simplifying, ensure that all key information is preserved—especially visual information such as objects, colors, and events. "
            "The simplified text should be more concise than the original but not overly short."
        )
        user_prompt = \
f"""
Original User Query:
{pre_query}

Simplify the above query by shortening it while preserving all details, especially visual information (e.g., objects, colors, events). 
Ensure that the simplified version is concise yet comprehensive, and does not become overly short.

Only return the simplified query, nothing else.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        concise_conv = conv_templates[conv_mode].copy()
        concise_conv.system = system_prompt
        concise_conv.append_message(concise_conv.roles[0], user_prompt)
        concise_conv.append_message(concise_conv.roles[1], None)
        prompt = concise_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = concise_conv.sep if concise_conv.sep_style != SeparatorStyle.TWO else concise_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        cur_answer = self.dialogue_history[-1]['answer']
        new_query = ' and '.join([outputs, cur_answer])
        # self.query_list.pop()
        return new_query
    
    def do_systhesis_caption(self):
        pre_query = self.query_list[-1]
        cur_answer = self.dialogue_history[-1]['answer']
        system_prompt = (
            "You are an expert in query refinement for interactive text-image retrieval. "
            "Your task is to synthesize and update a previous query with new details from the current answer. "
            "Ensure the new query includes key information (e.g., characters, events, objects, colors, locations) and does not exceed 60 words."
        )
        user_prompt = \
f"""
Previous Query:
{pre_query}

Current Answer (includes new information to enhance video retrieval):
{cur_answer}

Combine the above into one concise, positive declarative sentence that includes key details (characters, events, objects, colors, locations, etc.). 
Ensure the new query leverages the new information from the current answer for better retrieval and is no longer than 60 words.

Only return the refined query, nothing else.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        synthesis_conv = conv_templates[conv_mode].copy()
        synthesis_conv.system = system_prompt
        synthesis_conv.append_message(synthesis_conv.roles[0], user_prompt)
        synthesis_conv.append_message(synthesis_conv.roles[1], None)
        prompt = synthesis_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = synthesis_conv.sep if synthesis_conv.sep_style != SeparatorStyle.TWO else synthesis_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        outputs = outputs.split("Final Query:", 1)[-1].strip().strip("\"'*")
        # print(outputs)
        return outputs
        
    def do_question_generation(self,cur_meta_info):
        ut_lvl = self.get_uncertainty_level()
        if ut_lvl == 0:
            question = self.coarse_grained_question_generation()
        elif ut_lvl == 1:
            question = self.middle_grained_question_generation(cur_meta_info)
        else:
            question = self.fine_grained_question_generation()
        question =  re.sub(r'^[^a-zA-Z0-9]*','', question).replace('*','')
        question = question.strip('\'"')
        return question
    
    def get_answer_system_prompt(self):
        ut_lvl = self.get_uncertainty_level()
        if ut_lvl == 0:
            system_prompt = (
                "You are an expert in image content analysis and question answering. "
                "When provided with an image description and a question, focus on the image's visual and contextual details to craft a concise, one-sentence answer. "
                "Ensure your answer is accurate, centers on key visual information, and avoids overly verbose explanations or simple 'yes'/'no' responses."
            )
        else:
            system_prompt = (
                "You are an image question answering assistant. "
                "When provided with a image and a question, your task is to provide a concise, one-sentence answer. "
                "Your answer should clearly state the key visual details such as people, objects, scenes, and events. "
                "Keep it clear, direct, and focused on essential information."
            )
        return system_prompt
    
    def get_answer_user_prompt(self, video_rep, question):
        ut_lvl = self.get_uncertainty_level()
        if ut_lvl == 0:
            user_prompt = \
f"""
{video_rep}

Question:
{question}

Please provide a concise one-sentence answer that accurately reflects the key visual information in the image with respect to the question.
"""
        else:
            user_prompt = \
f"""
{video_rep}

Question:
{question}

Provide a one-sentence answer that clearly identifies the key visual details in the image, such as people, objects, scenes, and events.
"""
        return user_prompt

    def generate_answewr(self, question):
        video_rep = ' '.join([DEFAULT_IMAGE_TOKEN])
        
        user_prompt = self.get_answer_user_prompt(video_rep, question)
        system_prompt = self.get_answer_system_prompt()
        
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        answer_conv = conv_templates[conv_mode].copy()
        if system_prompt is not None:
            answer_conv.system = system_prompt
        answer_conv.append_message(answer_conv.roles[0], user_prompt)
        answer_conv.append_message(answer_conv.roles[1], None)
        tensor = self.video_pixels
        prompt = answer_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = answer_conv.sep if answer_conv.sep_style != SeparatorStyle.TWO else answer_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        return outputs
    
    def middle_grained_question_generation(self,cur_meta_info): 
        cur_text_query = self.query_list[-1]
        middle_system_prompt = (
            "You are a clarifying question generator for text-image retrieval. "
            "Given a user query and multiple image info, your task is to generate one question "
            "that focuses on visual differences. The question must start with What, Where, or Who."
        )
        middle_user_prompt = \
f"""
Query: "{cur_text_query}"
Images: {cur_meta_info}

Ask one question starting with What, Where, or Who to distinguish these image based on visual details.
Return ONLY the question.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        middle_grained_conv = conv_templates[conv_mode].copy()
        middle_grained_conv.system = middle_system_prompt
        middle_grained_conv.append_message(middle_grained_conv.roles[0], middle_user_prompt)
        middle_grained_conv.append_message(middle_grained_conv.roles[1], None)
        prompt = middle_grained_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = middle_grained_conv.sep if middle_grained_conv.sep_style != SeparatorStyle.TWO else middle_grained_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        return outputs

    
    def coarse_grained_question_generation(self):
        cur_text_query = self.query_list[-1]
        coarse_grained_system_prompt = (
            "You are an advanced AI specialized in asking clarifying questions for vague queries. "
            "Your task is to extract details—such as appearance, activities, or events—to enable precise retrieval."
        )
        user_prompt = \
f"""
Query: "{cur_text_query}"
Ask one open-ended clarifying question focusing on the subject's appearance, activities, or events.
Return ONLY the question.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        coarse_grained_conv = conv_templates[conv_mode].copy()
        coarse_grained_conv.system = coarse_grained_system_prompt
        coarse_grained_conv.append_message(coarse_grained_conv.roles[0], user_prompt)
        coarse_grained_conv.append_message(coarse_grained_conv.roles[1], None)
        prompt = coarse_grained_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = coarse_grained_conv.sep if coarse_grained_conv.sep_style != SeparatorStyle.TWO else coarse_grained_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        s = outputs.strip()
        return s

    def fine_grained_question_generation(self):
        cur_text_query = self.query_list[-1]
        # fine_grained_system_prompt = (
        #     "You are a top-tier AI for question asking based on a user query. "
        #     "First you need to evaluate whether the user's query includes sufficient visual details (such as characters, colors, objects, or locations). "
        #     "If details are missing, generate one question to gather them. "
        #     "If the query is already detailed, ask a clarifying question to further enrich the description (e.g., 'What other objects are present?', 'What is the main color?', or 'Where is the event taking place?'). "
        #     "Ensure your question starts with What, Where, or Who, is logically coherent, does not duplicate the previous question, and is concise. "
        # )

        user_prompt = \
f"""
You need to ask a question based on a user query.
1. First you need to evaluate whether the user's query includes sufficient visual details (such as characters, colors, objects, or locations).
User Query: "{cur_text_query}"

2. Ask a question
    - If details are missing, generate one question to gather them.
    - If the query is already detailed, generate a clarifying question to further enrich the description (e.g., 'What other objects are present?', 'What is the main color?', or 'Where is the event taking place?').


Return ONLY the question, nothing else.
"""
        conv_mode = self.model_config.get('conv_mode', None)
        assert conv_mode is not None, 'Conv mode not found'
        fine_grained_conv = conv_templates[conv_mode].copy()
        # fine_grained_conv.system = fine_grained_system_prompt
        fine_grained_conv.append_message(fine_grained_conv.roles[0], user_prompt)
        fine_grained_conv.append_message(fine_grained_conv.roles[1], None)
        prompt = fine_grained_conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = fine_grained_conv.sep if fine_grained_conv.sep_style != SeparatorStyle.TWO else fine_grained_conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.generation_tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '')
        return outputs
 
class ivrAutoDialogueManagerExtended:
    def __init__(self, config, model, generation_tokenizer, retrieval_tokenizer, device, all_video_features, all_video_pixels, video_id_to_caption=None, *args, **kwargs):
        dialogue_config = config.get('dialogue_config', None)
        if dialogue_config is None:
            raise ValueError('Dialogue config not found') 
        self.state = 0
        self.max_state = 17
        self.state_to_video = {0:0, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:2, 11:2, 12:2, 13:2, 14:2, 15:2, 16:2}
        self.state_to_prototype = {0:1, 1:2, 2:2, 3:3, 4:3, 5:3, 6:3, 7:4, 8:4, 9:4, 10:3, 11:3, 12:3, 13:3, 14:4, 15:4, 16:4}
        self.change_query_state = [6, 9, 13, 16]
        self.prototype3_to_object = {3:0, 4:0, 5:0, 6:0, 10:1, 11:1, 12:1, 13:1}
        self.prototype4_to_object = {7:0, 8:0, 9:0, 14:1, 15:1, 16:1}
        self.num_segments = dialogue_config.get('num_segments', 2)
        self.all_video_features = all_video_features
        self.all_video_pixels = all_video_pixels
        self.model = model
        self.generation_tokenizer = generation_tokenizer
        self.retrieval_tokenizer = retrieval_tokenizer
        self.device = device
        self.config = config
        self.state_function_map = {
            1: self.get_caption_from_prototype_1,
            2: self.get_caption_from_prototype_2,
            3: self.get_caption_from_prototype_3,
            4: self.get_caption_from_prototype_4
        }
        self.total_turn = dialogue_config.get('total_turn', -1)
        assert self.total_turn <= self.max_state, 'Total turn should be less than or equal to the max state'
        use_caption_cache = dialogue_config.get('use_caption_cache', False)
        use_caption = dialogue_config.get('use_caption', False)
        if use_caption and use_caption_cache:
            assert video_id_to_caption is not None, "video id to caption not found"
            self.video_id_to_caption = video_id_to_caption
        self.use_caption_cache = use_caption_cache
        self.use_caption = use_caption

    def reset_state(self, initial_query, video_pixel_features):
        self.state = 0
        self.initial_query = initial_query
        self.video_pixels = video_pixel_features
        self.total_num_frames = video_pixel_features.shape[2]
        self.frame_range = self.total_num_frames // self.num_segments
        self.query_list = [initial_query]
        self.question = None
        self.heuristic_vqa_caption_list = []
        for i in range(2):
            self.heuristic_vqa_caption_list.append(HeuristicCaptionManager(initial_query, self.model, self.video_pixels[:,:,i*self.frame_range:(i+1)*self.frame_range], self.config['model_config'], self.generation_tokenizer, self.device))
        self.heuristic_object_caption_list = []
        for i in range(2):
            self.heuristic_object_caption_list.append(HeuristicObjectManager(initial_query, self.model, self.video_pixels[:,:,i*self.frame_range:(i+1)*self.frame_range], self.config['model_config'], self.generation_tokenizer, self.device))
            
    def do_qa_generation_caption(self):
        if self.state >= self.total_turn:
            raise ValueError('Invalid state')
        cur_state = self.state % self.max_state
        prototype = self.state_to_prototype[cur_state]
        caption = self.state_function_map[prototype]()
        self.state += 1
        return self.assemble_query(caption, prototype)
    
    def get_query_list(self):
        if len(self.query_list) == 1:
            return self.query_list[0]
        else:
            return ' and '.join(self.query_list)
    
    def assemble_query(self, caption, prototype):
        assert len(self.query_list) >= 1, 'Query list is empty'
        cur_query_list = self.query_list.copy()
        cur_query_list.append(caption)
        cur_query_list = list(dict.fromkeys(filter(None, cur_query_list)))
        if (prototype not in [3,4]) or (self.state in self.change_query_state):
            self.query_list.append(caption)
        return ' and '.join(cur_query_list)

    def update_question(self, question):
        self.question = question
        
    def get_question(self):
        return self.question
    
    def get_cur_video_pixels(self):
        cur_state = self.state % self.max_state
        bin = self.state_to_video[cur_state]
        if bin == 0:
            return self.video_pixels
        elif bin == 1:
            return self.video_pixels[:,:,0:0+self.frame_range]
        elif bin == 2:
            return self.video_pixels[:,:,self.frame_range:self.frame_range*2]
        else:
            raise ValueError('Invalid bin')
        
    def get_caption_from_prototype_1(self):
        query_text = self.get_query_list()
        inds = get_cur_retrieval([query_text], self.all_video_features, self.model, self.device, self.retrieval_tokenizer)
        caption_list = None
        selected_video_pixels = None
        if self.use_caption and self.use_caption_cache:
            caption_list = []
            for cur_ind in inds:
                caption_list.append(self.video_id_to_caption[cur_ind])
        else:
            selected_video_pixels = self.all_video_pixels[inds]
        question = generate_question_auto(selected_video_pixels, self.model, self.device, self.config, self.generation_tokenizer, query_text, caption_list=caption_list)
        question = question.replace('"', '')
        self.update_question(question)
        caption = generate_caption_vqa_auto(self.get_cur_video_pixels(), self.model, self.device, self.config['model_config'], self.generation_tokenizer, question)
        return caption
    
    def get_caption_from_prototype_2(self):
        cur_video_pixels = self.get_cur_video_pixels()
        num_frames = cur_video_pixels.shape[2]
        question = self.get_question()
        caption = generate_caption_vqa_auto(cur_video_pixels, self.model, self.device, self.config['model_config'], self.generation_tokenizer, question, num_frames=num_frames)
        return caption
    
    def get_caption_from_prototype_3(self):
        cur_state = self.state % self.max_state
        object_bin = self.prototype3_to_object[cur_state]
        caption = self.heuristic_vqa_caption_list[object_bin].get_caption()
        return caption
    
    def get_caption_from_prototype_4(self):
        cur_state = self.state % self.max_state
        object_bin = self.prototype4_to_object[cur_state]
        caption = self.heuristic_object_caption_list[object_bin].get_caption()
        return caption
         
class ivrAutoDialogueManagerShort:
    def __init__(self, config, model, generation_tokenizer, retrieval_tokenizer, device,all_video_features,all_video_pixels, video_id_to_caption=None, *args, **kwargs):
        dialogue_config = config.get('dialogue_config', None)
        if dialogue_config is None:
            raise ValueError('Dialogue config not found')
        self.state = 0
        self.max_state = 7
        self.state_to_video = {0:0, 1:1, 2:2, 3:1, 4:1, 5:2, 6:2}
        self.state_to_prototype = {0:1, 1:2, 2:2, 3:3, 4:4, 5:3, 6:4}
        self.num_segments = dialogue_config.get('num_segments', 2)
        self.all_video_features = all_video_features
        self.all_video_pixels = all_video_pixels
        self.model = model
        self.generation_tokenizer = generation_tokenizer
        self.retrieval_tokenizer = retrieval_tokenizer
        self.device = device
        self.config = config
        self.state_function_map = {
            1: self.get_caption_from_prototype_1,
            2: self.get_caption_from_prototype_2,
            3: self.get_caption_from_prototype_3,
            4: self.get_caption_from_prototype_4
        }
        self.total_turn = dialogue_config.get('total_turn', None)
        assert self.total_turn is not None, 'Total turn not found'
        assert self.total_turn <= self.max_state, 'Total turn should be less than or equal to the max state'
        use_caption_cache = dialogue_config.get('use_caption_cache', False)
        use_caption = dialogue_config.get('use_caption', False)
        if use_caption and use_caption_cache:
            assert video_id_to_caption is not None, 'Video id to caption not found'
            self.video_id_to_caption = video_id_to_caption
        self.use_caption_cache = use_caption_cache
        self.use_caption = use_caption
    
    def reset_state(self, initial_query, video_pixel_features):
        self.state = 0
        self.initial_query = initial_query
        self.video_pixels = video_pixel_features
        self.total_num_frames = video_pixel_features.shape[2]
        self.frame_range = self.total_num_frames // self.num_segments
        self.query_list = [initial_query]
        self.question = None
    
    def do_qa_generation_caption(self):
        if self.state >= self.total_turn:
            raise ValueError('Invalid state')
        cur_state = self.state % self.max_state
        prototype = self.state_to_prototype[cur_state]
        caption = self.state_function_map[prototype]()
        self.query_list.append(caption)
        self.state += 1
        return self.assembel_query()
        
    def get_query_list(self):
        return self.query_list
    
    def assembel_query(self):
        self.query_list = list(dict.fromkeys(filter(None, self.query_list)))
        if len(self.query_list) == 1:
            return self.query_list[0]
        else:
            return ' and '.join(self.query_list)
        
    def update_question(self, question):
        self.question = question
    
    def get_question(self):
        return self.question
    
    def get_cur_video_pixels(self):
        cur_state = self.state % self.max_state
        bin = self.state_to_video[cur_state]
        if bin == 0:
            return self.video_pixels
        elif bin == 1:
            return self.video_pixels[:,:,0:0+self.frame_range]
        elif bin == 2:
            return self.video_pixels[:,:,self.frame_range:self.frame_range*2]
        else:
            raise ValueError('Invalid bin')
    
    def get_caption_from_prototype_1(self):
        query_text = self.assembel_query()
        inds = get_cur_retrieval([query_text], self.all_video_features, self.model, self.device, self.retrieval_tokenizer)
        caption_list = None
        selected_video_pixels = None
        if self.use_caption and self.use_caption_cache:
            caption_list = []
            for cur_ind in inds:
                caption_list.append(self.video_id_to_caption[cur_ind])
        else:
            selected_video_pixels = self.all_video_pixels[inds]
        question = generate_question_auto(selected_video_pixels, self.model, self.device, self.config, self.generation_tokenizer, query_text, caption_list=caption_list)
        question = question.replace('"', '')
        self.update_question(question)
        caption = generate_caption_vqa_auto(self.get_cur_video_pixels(), self.model, self.device, self.config['model_config'], self.generation_tokenizer, question)
        return caption
    
    def get_caption_from_prototype_2(self):
        cur_video_pixels = self.get_cur_video_pixels()
        num_frames = cur_video_pixels.shape[2]
        question = self.get_question()
        caption = generate_caption_vqa_auto(cur_video_pixels, self.model, self.device, self.config['model_config'], self.generation_tokenizer, question, num_frames=num_frames)
        return caption
    
    def get_caption_from_prototype_3(self):
        cur_video_pixels = self.get_cur_video_pixels()
        num_frames = cur_video_pixels.shape[2]
        original_query = self.query_list[0]
        caption = generate_caption_vqa(cur_video_pixels, original_query, self.model, self.device, self.config['model_config'], self.generation_tokenizer, num_frames=num_frames)
        caption = caption.lower()
        return caption
    
    def get_caption_from_prototype_4(self):
        cur_video_pixels = self.get_cur_video_pixels()
        num_frames = cur_video_pixels.shape[2]
        caption = generate_object_caption(cur_video_pixels, self.model, self.device, self.config['model_config'], self.generation_tokenizer, num_frames=num_frames)
        caption = caption.lower()
        return caption
                  
class ivrHeuristicDialogueManager:
    def __init__(self, config, model, generation_tokenizer, device, *args, **kwargs):
        dialogue_config = config.get('dialogue_config', None)
        if dialogue_config is None:
            raise ValueError('Dialogue config not found')
        self.question_list = [["What is the human doing?","Given the following video snippet, categorize it into one of three groups: (1) Cartoon, (2) Animal, or (3) Other. \
            Identify the most fitting category based on visual characteristics."],
                          ["Where is the human doing?", "What is the character?","What is the animal?", "What is the object?"],
                          ["What is the main object?", "What is the character doing?", "What is the animal doing?", "What is the object doing?"],
                          ["What color is the main object?", "Where is the character doing", "Where is the animal?", "Where is the object?"]
                        ]
        self.augment_object_question = ["What is the object?", "What color is the object?", "Where is the object?"]
        self.special_state = [[0,0],[1,1],[3,1],[1,0],[3,0]]
        self.prefix_answer_list = ["The human is", ["This is a cartoon video","This is a video of animal","This is neither a catoon video nor animal video"],
                            "The human is in", "The character is", "The animal is", "The object is","The main object is", "The character is", "The animal is",
                            "The object is", "The event happened in", "The character is in", "The animal is in", "The object is in",
                            "The object is", "The color of the object is", "The object is in"]
        self.qa_appendix = [{"property":"one gerund or gerund phrase ", "example_output":"Example Output: walking the dog"},
                            {"property":"one word among catoon, animal, other", "example_output": "Example Output: other"},
                            {"property":"one word or phrase about the location"},
                            {"property":"one word or phrase about the character"},
                            {"property":"one word or phrase about the animal"},
                            {"property":"one word or phrase about the object"},
                            {"property":"one word or phrase about the color"}]
        self.state_to_qaappendix = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:5, 7:0, 8:0, 9:0, 10:6, 11:2, 12:2, 13:2, 14:5, 15:6, 16:2}
        self.special_answer_state = [1, 2, 7, 8, 9, 10, 11, 12, 13, 15, 16]
        self.total_turn = dialogue_config.get('total_turn', -1)
        self.num_segments = dialogue_config.get('num_segments', 2)
        self.basic_turn = len(self.question_list)
        self.object_turn = len(self.augment_object_question)
        self.max_turn = self.basic_turn + self.num_segments * (self.object_turn + self.basic_turn)
        if self.total_turn == -1:
            self.total_turn = self.max_turn
        assert self.total_turn <= self.max_turn, 'Total turn should be less than or equal to the max turn'
        self.state_tracker = [0,0,0]
        self.num_stages = self.num_segments + 1
        self.model = model
        self.generation_tokenizer = generation_tokenizer
        self.device = device
        self.model_config = config.get('model_config', None)
        self.initial_query = ""
        self.query_list = None
    
    def reset_state(self, initial_query, video_pixel_features):
        self.state_tracker = [0,0,0]
        self.initial_query = initial_query
        self.video_pixels = video_pixel_features
        self.total_num_frames = video_pixel_features.shape[2]
        self.frame_range = self.total_num_frames // self.num_segments
        self.is_human = 0
        self.verb = ""
        self.answer_list = []
        self.cur_stage_answer_list = []
        self.valid_turn = True
        self.answer_state = -1
        self.query_list = [initial_query]
        self.cur_stage_query_list = []
        
    def get_query_list(self):
        return self.query_list
    
    def add_to_query_list(self, new_query):
        self.query_list.append(new_query)
        self.cur_stage_query_list.append(new_query)
        
    def prepare_for_next_stage(self):
        self.cur_stage_answer_list = []
        self.cur_stage_query_list = []
        self.state_tracker[0] += 1
        self.state_tracker[1] = 0
        self.state_tracker[2] = 0
        if self.state_tracker[0] > self.num_segments:
            self.valid_turn = False
    
    def update_answer_state(self,cur_turn):
        cur_turn = self.state_tracker[1], self.state_tracker[2]
        if cur_turn[0] == 0:
            self.answer_state = cur_turn[1]
        elif cur_turn[0] < self.basic_turn:
            self.answer_state = (cur_turn[0] - 1) * 4 + 2 + cur_turn[1]
        elif cur_turn[0] >= self.basic_turn and cur_turn[0] < (self.basic_turn + self.object_turn):
            self.answer_state = 14 + cur_turn[0] - self.basic_turn
        else:
            raise ValueError('Invalid turn')
        
    def do_basic_turn_asking(self):
        cur_turn = [self.state_tracker[1], self.state_tracker[2]]
        if cur_turn not in self.special_state:
            question = self.question_list[cur_turn[0]][cur_turn[1]]
            self.update_answer_state(cur_turn)
        else:
            if cur_turn == self.special_state[0]:
                question = self.process_state_0()
            elif cur_turn == self.special_state[1]:
                question = self.process_state_1()
            elif cur_turn == self.special_state[2]:
                question = self.process_state_2()
            elif cur_turn == self.special_state[3]:
                question = self.process_state_3()
            elif cur_turn == self.special_state[4]:
                question = self.process_state_4()
            else:
                raise ValueError('Invalid state')
        return question
    
    def do_object_turn_asking(self):
        cur_turn = [self.state_tracker[1], self.state_tracker[2]]
        cur_turn[0] -= self.basic_turn 
        question = self.augment_object_question[cur_turn[0]]
        self.update_answer_state([self.state_tracker[1], self.state_tracker[2]])
        return question
    
    def do_question_asking(self):
        if not self.valid_turn:
            raise ValueError('Invalid turn')
        cur_stage = self.state_tracker[0]
        if cur_stage == 0:
            question = self.do_basic_turn_asking()
        else:
            if self.state_tracker[1] < self.basic_turn:
                question = self.do_basic_turn_asking()
            else:
                question = self.do_object_turn_asking()
        return question
        
    def process_state_0(self):
        split_captions = self.initial_query.split(' ')
        human_captions = ['person', 'people', 'man', 'men', 'woman', 'women', 'girl', 'girls', 'boy',
                      'boys', 'child', 'children', 'male', 'female', 'lady', 'family', 'models']
        singular_human_captions = ['person', 'man', 'woman', 'girl', 'boy', 'child', 'male', 'female', 'lady', 'family']
        for human_caption in human_captions:
            if human_caption in split_captions:
                is_human = 1
                break
            else:
                is_human = 0
        self.is_human = is_human
        self.verb = get_conjunctions(singular_human_captions, human_caption)
        self.human_caption = human_caption
        if self.is_human:
            self.state_tracker[2] = 0
            question = f"What {self.verb} the {self.human_caption} doing?"
        else:
            self.state_tracker[2] = 1
            question = self.question_list[self.state_tracker[1]][self.state_tracker[2]]
        self.update_answer_state([self.state_tracker[1], self.state_tracker[2]])
        return question 

    def process_state_1(self):
        assert len(self.cur_stage_answer_list) > 0, 'Answer list is empty'
        last_answer = self.answer_list[-1]
        if 'cartoon' in last_answer.lower():
            self.state_tracker[2] = 1
        elif 'animal' in last_answer.lower():    
            self.state_tracker[2] = 2
        elif 'other' in last_answer.lower():
            self.state_tracker[2] = 3
        else:
            raise ValueError('Invalid answer')
        question = self.question_list[self.state_tracker[1]][self.state_tracker[2]]
        self.update_answer_state([self.state_tracker[1], self.state_tracker[2]])
        return question

    def process_state_2(self):
        assert len(self.cur_stage_answer_list) > 1, 'Answer list is empty'
        question = self.question_list[self.state_tracker[1]][self.state_tracker[2]] + ' ' + self.answer_list[-1]
        self.update_answer_state([self.state_tracker[1], self.state_tracker[2]])
        return question
    
    def process_state_3(self):
        assert self.is_human, "This state should be in the human-related video"
        assert len(self.cur_stage_answer_list) == 1, 'There should be only one dialogue turn happened'
        question = f"Where {self.verb} the {self.human_caption} {self.answer_list[-1].lower()}?"
        self.update_answer_state([self.state_tracker[1], self.state_tracker[2]])
        return question
    
    def process_state_4(self):
        assert self.is_human, "This state should be in the human-related video"
        assert len(self.cur_stage_answer_list) == 3, 'There should be 3 dialogue turns happened'
        object = self.cur_stage_answer_list[-1].lower()
        question = f"What color is the {object}?"
        self.update_answer_state([self.state_tracker[1], self.state_tracker[2]])
        return question     
        
    def do_qa_generation(self, question):
        cur_stage = self.state_tracker[0]
        if cur_stage == 0:
            video_frames = self.video_pixels
        else:
            k = cur_stage - 1
            video_frames = self.video_pixels[:,:,k:k+self.frame_range] #TODO: possible bug here
        cur_answer_state = self.answer_state
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[cur_answer_state]]
        if cur_stage:
            cur_qa_appendix['num_frames'] = video_frames.shape[2]
        answer = video_llava_qa_short(self.model, video_frames, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix)
        self.answer_list.append(answer)
        self.cur_stage_answer_list.append(answer)
          
    def assemble_caption(self):
        if self.answer_state not in self.special_answer_state:
            new_caption = self.prefix_answer_list[self.answer_state] + ' ' + self.answer_list[-1].lower() + '.'
        else:
            state_function_map = {
                state: getattr(self, f"do_answer_state_{i}")
                for i, state in enumerate(self.special_answer_state)
            }
            if self.answer_state in state_function_map:
                new_caption = state_function_map[self.answer_state]()
            else:
                raise ValueError('Invalid answer state')
        return new_caption

    def update_state(self):
        self.state_tracker[1] += 1
        if self.state_tracker[0] == 0 and self.state_tracker[1] >= self.basic_turn:
            self.prepare_for_next_stage()
        elif self.state_tracker[0] > 0 and self.state_tracker[1] >= (self.basic_turn + self.object_turn):
            self.prepare_for_next_stage()
    
    def do_answer_state_0(self):
        last_answer = self.answer_list[-1].lower()
        if 'cartoon' in last_answer:
            new_caption = self.prefix_answer_list[self.answer_state][0]
        elif 'animal' in last_answer:
            new_caption = self.prefix_answer_list[self.answer_state][1]
        else:
            new_caption = self.prefix_answer_list[self.answer_state][2]
        return new_caption
    
    def do_answer_state_1(self):
        assert len(self.cur_stage_answer_list) == 2, 'There should be only one dialogue turn happened'
        assert len(self.cur_stage_query_list) == 1, f'There should be only one query in the list, current query list: {self.cur_stage_query_list}'
        event = self.cur_stage_answer_list[0].lower()
        place = self.cur_stage_answer_list[1].lower()
        new_caption = f"The {self.human_caption} {self.verb} {event} in {place}."
        self.query_list.pop()
        self.cur_stage_query_list.pop()
        return new_caption
    
    def __do_answer_state_234(self):
        assert len(self.cur_stage_answer_list) == 3, 'It should be in the third dialogue turn'
        assert len(self.cur_stage_query_list) == 2, 'There should be two queries in the list'
        subject = self.cur_stage_answer_list[1].lower()
        object = self.cur_stage_answer_list[2].lower()
        new_caption = f"The {subject} is {object}."
        self.query_list.pop()
        self.query_list.pop()
        self.cur_stage_query_list.pop()
        self.cur_stage_query_list.pop()
        return new_caption
    
    def do_answer_state_2(self):
        return self.__do_answer_state_234()
    
    def do_answer_state_3(self):
        return self.__do_answer_state_234()
    
    def do_answer_state_4(self):
        return self.__do_answer_state_234()
    
    def do_answer_state_5(self):
        assert len(self.cur_stage_answer_list) == 4, "It should be in the fourth dialogue turn"
        assert len(self.cur_stage_query_list) == 2, "There should be two queries in the list"
        subject = self.cur_stage_answer_list[2].lower()
        object = self.cur_stage_answer_list[3].lower()
        new_caption = f"The {subject} is {object}."  
        self.query_list.pop()
        self.cur_stage_query_list.pop()
        return new_caption
    
    def __do_answer_state_678(self):
        assert len(self.cur_stage_answer_list) == 4, 'It should be in the fourth dialogue turn'
        assert len(self.cur_stage_query_list) == 1, 'There should be only one query in the list'
        subject = self.cur_stage_answer_list[1].lower()
        object = self.cur_stage_answer_list[2].lower()
        place = self.cur_stage_answer_list[3].lower()
        new_caption = f"The {subject} is {object} in {place}."
        self.query_list.pop()
        self.cur_stage_query_list.pop()
        return new_caption

    def do_answer_state_6(self):
        return self.__do_answer_state_678()
    
    def do_answer_state_7(self):
        return self.__do_answer_state_678()
    
    def do_answer_state_8(self):
        return self.__do_answer_state_678()
    
    def do_answer_state_9(self):
        assert len(self.cur_stage_answer_list) == 6, 'There should be 6 dialogue turns happened'
        if self.is_human:
            assert len(self.cur_stage_query_list) == 3, 'There should be 3 queries in the list'
        else:
            assert len(self.cur_stage_query_list) == 2, 'There should be 2 queries in the list'
        subject = self.cur_stage_answer_list[-2].lower()
        object = self.cur_stage_answer_list[-1].lower()
        new_caption = f"The {subject} is {object}."
        self.query_list.pop()
        self.cur_stage_query_list.pop()
        return new_caption
    
    def do_answer_state_10(self):
        assert len(self.cur_stage_answer_list) == 7, 'There should be 7 dialogue turns happened'
        if self.is_human:
            assert len(self.cur_stage_query_list) == 3, 'There should be 3 queries in the list'
        else:
            assert len(self.cur_stage_query_list) == 2, 'There should be 2 queries in the list'
        subject = self.cur_stage_answer_list[-3].lower()
        color = self.cur_stage_answer_list[-2].lower()
        location = self.cur_stage_answer_list[-1].lower()
        new_caption = f"The {subject} is {color} in {location}."
        self.query_list.pop()
        self.cur_stage_query_list.pop()
        return new_caption

    def get_total_turn(self):
        return self.total_turn
        
class HeuristicCaptionManager:
    def __init__(self, query_text, model, video_pixels, model_config, generation_tokenizer, device):
        self.state = -1
        # self.next_state = -1
        self.initial_query = query_text
        self.qa_appendix = [{"property":"one gerund or gerund phrase ", "example_output":"Example Output: walking the dog"},
                            {"property":"one word among catoon, animal, other", "example_output": "Example Output: other"},
                            {"property":"one word or phrase about the location"},
                            {"property":"one word or phrase about the character"},
                            {"property":"one word or phrase about the animal"},
                            {"property":"one word or phrase about the object"},
                            {"property":"one word or phrase about the color"}]
        self.state_to_qaappendix = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:5, 7:0, 8:0, 9:0, 10:6, 11:2, 12:2, 13:2}
        self.model = model
        self.video_pixels = video_pixels
        self.model_config = model_config
        self.generation_tokenizer = generation_tokenizer
        self.device = device
        self.answer_list = []
        self.dialogue_turn = 0
    
    def get_caption(self):
        if self.state == -2:
            raise ValueError('state is -2')
        elif self.state == -1:
            return self.process_state_minus1()
        try:
            process_method = getattr(self, f"process_state_{self.state}")
            return process_method()
        except AttributeError:
            raise ValueError('Invalid state: {}'.format(self.state))
    
    def process_state_minus1(self):
        split_captions = self.initial_query.split(' ')
        human_captions = ['person', 'people', 'man', 'men', 'woman', 'women', 'girl', 'girls', 'boy',
                      'boys', 'child', 'children', 'male', 'female', 'lady', 'family', 'models']
        singular_human_captions = ['person', 'man', 'woman', 'girl', 'boy', 'child', 'male', 'female', 'lady', 'family']
        for human_caption in human_captions:
            if human_caption in split_captions:
                is_human = 1
                break
            else:
                is_human = 0
        self.is_human = is_human
        self.verb = get_conjunctions(singular_human_captions, human_caption)
        self.human_caption = human_caption
        if self.is_human:
            return self.process_state_0()
        else:
            return self.process_state_1()
    
    def process_state_0(self):
        question = f'What {self.verb} the {self.human_caption} doing?'
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[0]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        query = f"The {self.human_caption} {self.verb} {answer}"
        self.state = 2
        return query
           
    def process_state_1(self):
        question = "Given the following video snippet, categorize it into one of three groups: (1) Cartoon, (2) Animal, or (3) Other. Identify the most fitting category based on visual characteristics."
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[1]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        if 'cartoon' in answer:
            query = "This is a cartoon video"
            self.state = 3
        elif 'animal' in answer:
            query = "This is a video of animal"
            self.state = 4
        else:
            query = "This is neither a catoon video nor animal video"
            self.state = 5
        return query
    
    def process_state_2(self):
        assert self.is_human and len(self.answer_list) == 1, 'Invalid state goto state 2'
        question = f"Where {self.verb} the {self.human_caption} {self.answer_list[0]}?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[2]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        query = f"The {self.human_caption} {self.verb} {self.answer_list[0]} in {answer}"
        self.state = 6
        return query
    
    def process_state_3(self):
        assert not self.is_human and len(self.answer_list) == 1, 'Invalid state goto state 3'
        question = "What is the character?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[3]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        query = "This is a cartoon video and " + f"the character is {answer}"
        self.state = 7
        return query
    
    def process_state_4(self):
        assert not self.is_human and len(self.answer_list) == 1, 'Invalid state goto state 4'
        question = "What is the animal?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[4]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        query = "This is a video of animal and " + f"the animal is {answer}"
        self.state = 8
        return query
    
    def process_state_5(self):
        assert not self.is_human and len(self.answer_list) == 1, 'Invalid state goto state 5'
        question = "What is the object?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[5]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        query = f"The object is {answer}"
        self.state = 9
        return query
    
    def process_state_6(self):
        assert self.is_human and len(self.answer_list) == 2, 'Invalid state goto state 6'
        question = f"What is the main object?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[6]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        query = f"the {self.human_caption} {self.verb} {self.answer_list[0]} in {self.answer_list[1]} and "
        query += f"the main object is {answer}"
        self.state = 10
        return query
    
    def process_state_7(self):
        assert not self.is_human and len(self.answer_list) == 2, 'Invalid state goto state 7'
        question = f"What is the character doing?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[7]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        character = self.answer_list[1]
        query = f"{character} is {answer}"
        self.state = 11
        return query
    
    def process_state_8(self):
        assert not self.is_human and len(self.answer_list) == 2, 'Invalid state goto state 8'
        question = f"What is the animal doing?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[8]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        animal = self.answer_list[1]
        query = f"{animal} is {answer}"
        self.state = 12
        return query
    
    def process_state_9(self):
        assert not self.is_human and len(self.answer_list) == 2, 'Invalid state goto state 9'
        question = f"What is the object doing?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[9]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        object = self.answer_list[1]
        query = f"{object} is {answer}"
        self.state = 13
        return query
    
    def process_state_10(self):
        assert self.is_human and len(self.answer_list) == 3, 'Invalid state goto state 10'
        question = f"What color is the main object?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[10]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        doing = self.answer_list[0]
        loc = self.answer_list[1]
        object = self.answer_list[2]
        color = self.answer_list[3]
        query = f"the {self.human_caption} {self.verb} {doing} in {loc} and the {object} is {color}"
        self.state = -2
        return query
    
    def process_state_11(self):
        assert not self.is_human and len(self.answer_list) == 3, 'Invalid state goto state 11'
        event = self.answer_list[2]
        question = f"Where is the character {event}?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[11]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        character = self.answer_list[1]
        query = f"the {character} is {event} in {answer}"
        self.state = -2
        return query
    
    def process_state_12(self):
        assert not self.is_human and len(self.answer_list) == 3, 'Invalid state goto state 12'
        question = f"Where is the animal?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[12]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        animal = self.answer_list[1]
        doing = self.answer_list[2]
        query = f"The {animal} is {doing} in {answer}"
        self.state = -2
        return query
    
    def process_state_13(self):
        assert not self.is_human and len(self.answer_list) == 3, 'Invalid state goto state 13'
        question = f"Where is the object?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[13]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        object = self.answer_list[1]
        doing = self.answer_list[2]
        query = f"The {object} is {doing} in {answer}"
        self.state = -2
        return query

class HeuristicObjectManager:
    def __init__(self, query_text, model, video_pixels, model_config, generation_tokenizer, device):
        self.state = 0
        self.initial_query = query_text
        self.qa_appendix = [{"property":"one gerund or gerund phrase ", "example_output":"Example Output: walking the dog"},
                            {"property":"one word among catoon, animal, other", "example_output": "Example Output: other"},
                            {"property":"one word or phrase about the location"},
                            {"property":"one word or phrase about the character"},
                            {"property":"one word or phrase about the animal"},
                            {"property":"one word or phrase about the object"},
                            {"property":"one word or phrase about the color"}]
        self.state_to_qaappendix = {0:5, 1:6, 2:2}
        self.model = model
        self.video_pixels = video_pixels
        self.model_config = model_config
        self.generation_tokenizer = generation_tokenizer
        self.device = device
        self.answer_list = []
        self.dialogue_turn = 0
    
    def get_caption(self):
        if self.state == -1:
            raise ValueError('state is -1')
        try:
            process_method = getattr(self, f"process_state_{self.state}")
            return process_method()
        except AttributeError:
            raise ValueError('Invalid state: {}'.format(self.state))
        
    def process_state_0(self):
        assert len(self.answer_list) == 0, 'Invalid state goto state 0'
        question = "What is the object?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[0]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        query = f"The object is {answer}"
        self.state = 1
        return query
    
    def process_state_1(self):
        assert len(self.answer_list) == 1, 'Invalid state goto state 1'
        question = f"What color is the object?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[1]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        object = self.answer_list[0]
        query = f"The {object} is {answer}"
        self.state = 2
        return query
        
    def process_state_2(self):
        assert len(self.answer_list) == 2, 'Invalid state goto state 2'
        question = "Where is the object?"
        cur_qa_appendix = self.qa_appendix[self.state_to_qaappendix[2]]
        cur_qa_appendix['num_frames'] = self.video_pixels.shape[2]
        answer = video_llava_qa_short(self.model, self.video_pixels, self.model_config, question, self.generation_tokenizer, self.device, **cur_qa_appendix).lower()
        self.answer_list.append(answer)
        object = self.answer_list[0]
        color = self.answer_list[1]
        query = f"The {object} is {color} and {answer}"
        self.state = -1
        return query  
    
class ivrAutoDialogueManagerWOAug:
    def __init__(self, config, model, generation_tokenizer, retrieval_tokenizer, device, all_video_features, all_video_pixels, video_id_to_caption=None, *args, **kwargs):
        dialogue_config = config.get('dialogue_config', None)
        if dialogue_config is None:
            raise ValueError('Dialogue config not found') 
        self.state = 0
        self.max_state = 17
        self.num_segments = dialogue_config.get('num_segments', 2)
        self.all_video_features = all_video_features
        self.all_video_pixels = all_video_pixels
        self.model = model
        self.generation_tokenizer = generation_tokenizer
        self.retrieval_tokenizer = retrieval_tokenizer
        self.device = device
        self.config = config
        self.total_turn = dialogue_config.get('total_turn', -1)
        assert self.total_turn <= self.max_state, 'Total turn should be less than or equal to the max state'
        if self.total_turn == -1:
            self.total_turn = self.max_state
        use_caption_cache = dialogue_config.get('use_caption_cache', False)
        use_caption = dialogue_config.get('use_caption', False)
        if use_caption and use_caption_cache:
            assert video_id_to_caption is not None, "video id to caption not found"
            self.video_id_to_caption = video_id_to_caption
        self.use_caption_cache = use_caption_cache
        self.use_caption = use_caption

    def reset_state(self, initial_query, video_pixel_features):
        self.state = 0
        self.initial_query = initial_query
        self.video_pixels = video_pixel_features
        self.total_num_frames = video_pixel_features.shape[2]
        self.frame_range = self.total_num_frames // self.num_segments
        self.query_list = [initial_query]
        self.question = None
            
    def do_qa_generation_caption(self):
        if self.state >= self.total_turn:
            raise ValueError('Invalid state')
        cur_state = self.state % self.max_state
        caption = self.get_caption_from_prototype_1()
        self.state += 1
        return self.assemble_query(caption)
    
    def get_query_list(self):
        return self.query_list
        
    def get_querys(self):
        if len(self.query_list) == 1:
            return self.query_list[0]
        else:
            return ' and '.join(self.query_list)
    
    def assemble_query(self, caption):
        assert len(self.query_list) >= 1, 'Query list is empty'
        cur_query_list = self.query_list.copy()
        cur_query_list.append(caption)
        cur_query_list = list(dict.fromkeys(filter(None, cur_query_list)))
        self.query_list.append(caption)
        return ' and '.join(cur_query_list)

    def update_question(self, question):
        self.question = question
        
    def get_question(self):
        return self.question
    
    def get_cur_video_pixels(self):
        return self.video_pixels
        
    def get_caption_from_prototype_1(self):
        # query_text = self.get_query_list()
        inds = get_cur_retrieval([self.get_querys()], self.all_video_features, self.model, self.device, self.retrieval_tokenizer)
        caption_list = None
        selected_video_pixels = None
        if self.use_caption and self.use_caption_cache:
            caption_list = []
            for cur_ind in inds:
                caption_list.append(self.video_id_to_caption[cur_ind])
        else:
            selected_video_pixels = self.all_video_pixels[inds]
        query_list = self.get_query_list()
        question = self.generate_question_wo_aug(selected_video_pixels, query_list, caption_list=caption_list)
        question = question.replace('"', '')
        self.update_question(question)
        caption = generate_caption_vqa_auto(self.get_cur_video_pixels(), self.model, self.device, self.config['model_config'], self.generation_tokenizer, question, heuristic=False)
        return caption
    
    def generate_question_wo_aug(self, selected_video_pixels, query_text, caption_list=None):
        dialogue_config = self.config.get('dialogue_config', None)
        assert dialogue_config is not None, 'dialogue_config not found'
        use_caption = dialogue_config.get('use_caption', False)
        use_caption_cache = dialogue_config.get('use_caption_cache', False)
        if use_caption and not use_caption_cache:
            model_config = self.config.get('model_config', None)
            assert model_config is not None, 'model_config not found'
            video_num = len(selected_video_pixels)
            caption_list = []
            for i in range(video_num):
                cur_video_pixels = selected_video_pixels[i:i+1]
                cur_video_caption = video_llava_get_video_caption(cur_video_pixels, self.model, self.device, model_config, self.generation_tokenizer)
                caption_list.append(cur_video_caption)
            seperator = ', '
            all_caps = seperator.join(caption_list)
            q1 = f"Suppose you are given the following video descriptions: \"{all_caps}\". " \
                f"What question would you ask to help you uniquely identify the video described as follows: \"{query_text}\" ? You must directly output the question."
        elif use_caption and use_caption_cache:
            q1 = f"Suppose you are given the following video descriptions: \"{caption_list}\". " \
                f"What question would you ask to help you uniquely identify the video described as follows: \"{query_text}\" ? You must directly output the question."
        else:
            q1 = f"Suppose you are given the following video descriptions: \"{query_text}\". " \
                f"What question would you ask to help you uniquely identify the video? You must directly output the question."
        question1 = video_llava_plain_text_qa(self.model, self.config, q1, self.generation_tokenizer, self.device)
        return question1
       
