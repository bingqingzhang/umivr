import numpy as np
import json
import torch
import math
import logging
import sys
from tqdm import tqdm
from collections import defaultdict

from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

logger = logging.getLogger(__name__)


def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms

def get_cosine_similarity(all_text_features, all_video_features):
    normalized_text_features = normalize_features(all_text_features)
    normalized_video_features = normalize_features(all_video_features)
    cosine_similarities = normalized_text_features @ normalized_video_features.T
    return cosine_similarities

def get_one_query_cosine_similarity(a, b):
    return np.dot(b, a.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1, keepdims=True))

def get_cur_retrieval(text_data, all_video_features, model, device, retrieval_tokenizer, topk=5, return_scores=False):
    tokens = retrieval_tokenizer(text_data, truncation=True)
    input_ids = torch.tensor(tokens['input_ids'], dtype=torch.long).to(device)
    with torch.inference_mode():
        outputs = model.get_video_tower().video_tower_text_encoder(input_ids=input_ids, output_hidden_states=True)
        text_pooler_output = outputs.pooler_output
        text_features = model.get_model().video_tower.retrieval_text_proj(text_pooler_output)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    cur_text_feature = text_features.cpu().detach().numpy()
    similarity_score = get_one_query_cosine_similarity(cur_text_feature, all_video_features)
    topk_inds = np.argsort(similarity_score, axis=0)[-topk:][::-1].flatten()
    topk_scores = similarity_score[topk_inds]
    if return_scores:
        return topk_inds, topk_scores
    return topk_inds

def compute_metrics_multiturn(all_text_features, all_video_features, total_turn):
    recall_by_turn = defaultdict()
    for i in range(1, total_turn + 1):
        all_text_features_turn = np.array([text_features[i] for text_features in all_text_features])
        sim_matrix = torch.tensor(all_text_features_turn @ all_video_features.T)
        recall_by_turn[i] = compute_recall_metrics(sim_matrix)
    
    all_sim_matrices = []
    for i in range(total_turn + 1):
        all_text_features_turn = np.array([text_features[i] for text_features in all_text_features])
        sim_matrix = torch.tensor(all_text_features_turn @ all_video_features.T)
        all_sim_matrices.append(sim_matrix)
    hit_by_turn = compute_hit_metrics(all_sim_matrices)
    
    bri_by_turn = compute_bri_metrics(all_sim_matrices)
    
    # new_metrics = compute_new_metrics_v2(all_sim_matrices)
    
    # for i in range(total_turn+1):
    #     logger.info(f't2v {i}: {recall_by_turn[i]}')
    for key, value in recall_by_turn.items():
        logger.info(f't2v in {key}: {value}')
        # print(f'v2t {i}', recall_by_turn[i])
    # for i in range(total_turn+1):
    #     logger.info(f'hit {i}, {hit_by_turn[i]}')
    for k, v in hit_by_turn.items():
        logger.info(f'Hit in {k}: {v}')
        
    for k, v in bri_by_turn.items():
        logger.info(f'BRI in {k}: {v}')
        
    # for k, v in new_metrics.items():
    #     logger.info(f'New metrics in {k}: {v}')

def compute_recall_metrics(x):
    num_samples = x.shape[0]
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    diff = sx - d
    cur_ranks = np.array([
        np.min(np.where(np.abs(diff[i]) < 1e-4)[0])
        for i in range(num_samples)
    ])
    metrics = {}
    metrics['R@1'] = float(np.sum(cur_ranks == 0)) * 100 / num_samples
    metrics['R@5'] = float(np.sum(cur_ranks < 5)) * 100 / num_samples
    metrics['R@10'] = float(np.sum(cur_ranks < 10)) * 100 / num_samples
    # metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = np.median(cur_ranks) + 1
    metrics["MeanR"] = np.mean(cur_ranks) + 1
    return metrics

def compute_hit_metrics(x_list):
    num_queries = x_list[0].shape[0]
    cumulative_best = np.full(num_queries, np.inf)
    hit_results = {}
    for round_idx, x in enumerate(x_list, start=0):
        sx = np.sort(-x, axis=1)
        d = np.diag(-x)[:, np.newaxis]
        diff = sx - d
        current_round_ranks = np.array([
            np.min(np.where(np.abs(diff[i]) < 1e-5)[0])
            for i in range(num_queries)
        ])
        cumulative_best = np.minimum(cumulative_best, current_round_ranks)
        hit1 = float(np.sum(cumulative_best == 0)) * 100 / num_queries
        hit5 = float(np.sum(cumulative_best < 5)) * 100 / num_queries
        hit10 = float(np.sum(cumulative_best < 10)) * 100 / num_queries
        hit_results[f'Round_{round_idx}'] = {
            'hit@1': hit1,
            'hit@5': hit5,
            'hit@10': hit10
        }
    return hit_results

def rank_by_turn_analysis(all_sim_matrices):
    total_turn = len(all_sim_matrices)
    total_samples = all_sim_matrices[0].shape[0]
    rank_in_sample = defaultdict(list)
    for i in range(0, total_turn+1):
        cur_sim_matrix = all_sim_matrices[i]
        for j in range(total_samples):
            sorted_indicies = torch.argsort(cur_sim_matrix[j], descending=True)
            cur_rank = torch.where(sorted_indicies == j)[0].item() + 1
            rank_in_sample[j].append(cur_rank)
    # write rank_in_sample to file
    with open('rank_in_sample.json', 'w') as f:
        json.dump(rank_in_sample, f)
    print('Rank in sample has been written to file')

def compute_bri_metrics(all_sim_matrices):
    total_turn = len(all_sim_matrices) - 1
    total_samples = all_sim_matrices[0].shape[0]
    best_rank = defaultdict(list)
    bri_by_turn = defaultdict()
    # initialize best rank
    init_sim_matrix = all_sim_matrices[0]
    for i in range(total_samples):
        sorted_indices = torch.argsort(init_sim_matrix[i], descending=True)
        best_rank[i] = [torch.where(sorted_indices == i)[0].item() + 1]
    # update best rank
    for i in range(1, total_turn+1):
        cur_sim_matrix = all_sim_matrices[i]
        for j in range(total_samples):
            sorted_indices = torch.argsort(cur_sim_matrix[j], descending=True)
            cur_rank = torch.where(sorted_indices == j)[0].item() + 1
            previous_rank = best_rank[j][-1]
            best_rank[j].append(cur_rank if cur_rank < previous_rank else previous_rank)
        # compute BRI metrics at the round i
        cur_bri = 0
        for j in range(total_samples):
            term1 = 0
            term2 = 0
            
            term1 = (1 / (2 * i)) * np.log(best_rank[j][0] * best_rank[j][i])
            
            for k in range(1, i):
                term2 += np.log(best_rank[j][k])
            term2 = (1 / i) * term2
            
            cur_bri += term1 + term2
        bri_by_turn[i] = cur_bri / total_samples
    
    return bri_by_turn
                
                

def get_conjunctions(singular_captions, human_caption):
    singular = human_caption in singular_captions
    verb = "is" if singular else "are"
    return verb

def generate_caption_vqa(video_frame, origin_caption, model, device, model_config, tokenizer, video_path=None, preprocess=None, num_frames=None):
    split_captions = origin_caption.split(' ')
    video_tensor = torch.tensor(video_frame, dtype=torch.float16).to(device)
    num_frames = video_frame.shape[2]
    human_captions = ['person', 'people', 'man', 'men', 'woman', 'women', 'girl', 'girls', 'boy',
                      'boys', 'child', 'children', 'male', 'female', 'lady', 'family', 'models']
    singular_human_captions = ['person', 'man', 'woman', 'girl', 'boy', 'child', 'male', 'female', 'lady', 'family']
    for human_caption in human_captions:
        if human_caption in split_captions:
            is_human = 1
            break
        else:
            is_human = 0
            
    verb = get_conjunctions(singular_human_captions, human_caption)
    if is_human:
        question1 = f'What {verb} the {human_caption} doing ?'
        answer1 =video_llava_qa_short(model, video_tensor, model_config, question1, tokenizer, device, property="one gerund or gerund phrase ", example_output="Example Output: walking the dog ", num_frames=num_frames)
        # answer_verify = video_llava_qa_debug(model, video_path, model_config, question1, tokenizer, device, preprocessor=preprocess)
        question2 = f'Where {verb} the {human_caption} {answer1} ?'
        answer2 = video_llava_qa_short(model, video_tensor, model_config, question2, tokenizer, device, property=" one word or phrase about location ", num_frames=num_frames)
        if 'no' in answer2:
            new_caption = f'{human_caption} {verb} {answer1}'
        else:
            new_caption = f'{human_caption} {verb} {answer1} {answer2}'
    else:
        question = 'Is this cartoon ?'
        answer = video_llava_qa_short(model,video_tensor, model_config, question, tokenizer, device, property="yes or no ", num_frames=num_frames)
        if 'yes' in answer:
            question1 = 'What is the character ?'
            answer1 = video_llava_qa_short(model, video_tensor, model_config, question1, tokenizer, device, property="one word or phrase about the character ", num_frames=num_frames)
            question2 = 'What is the character doing ?'
            answer2 = video_llava_qa_short(model, video_tensor, model_config, question2, tokenizer, device, property="one gerund or gerund phrase ", example_output="Example Output: running ", num_frames=num_frames)
            new_caption = f'{answer1} is {answer2}'
        else:
            question = 'Is there any animal ?'
            answer = video_llava_qa_short(model, video_tensor, model_config, question, tokenizer, device, property="yes or no ", num_frames=num_frames)
            if 'yes' in answer:
                question1 = 'What is the animal ?'
                answer1 = video_llava_qa_short(model, video_tensor, model_config, question1, tokenizer, device, property="one word or phrase about the animal ", num_frames=num_frames)
                question2 = 'What is the animal doing ?'
                answer2 = video_llava_qa_short(model, video_tensor, model_config, question2, tokenizer, device, property="one gerund or gerund phrase ", example_output="Example Output: running ", num_frames=num_frames)
                question3 = 'Where is the animal ?'
                answer3 = video_llava_qa_short(model, video_tensor, model_config, question3, tokenizer, device, property="one word or phrase about location ", num_frames=num_frames)
                new_caption = f'{answer1} is {answer2} {answer3}'
            else:
                question1 = 'What is the object ?'
                answer1 = video_llava_qa_short(model, video_tensor, model_config, question, tokenizer, device, property="one word or phrase ", num_frames=num_frames)
                question2 = 'What is the object doing ?'
                answer2 = video_llava_qa_short(model, video_tensor, model_config, question1, tokenizer, device, property="one gerund or gerund phrase ", example_output="Example Output: running ", num_frames=num_frames)
                question3 = 'Where is the object ?'
                answer3 = video_llava_qa_short(model, video_tensor, model_config, question3, tokenizer, device, property="one word or phrase about location ", num_frames=num_frames)
                new_caption = f'{answer1} is {answer2} {answer3}'
    return new_caption        

def generate_caption_vqa_auto(video_frame, model, device, model_config, tokenizer, question, num_frames=None, heuristic=True):
    num_frames = video_frame.shape[2]
    video_tensor = torch.tensor(video_frame, dtype=torch.float16).to(device)
    if not heuristic:
        answer = video_llava_qa_normal(model, video_tensor, model_config, question, tokenizer, device, num_frames=num_frames, question_suffix="Respond questions concisely and directly.")
        return answer
    answer = video_llava_qa_short(model, video_tensor, model_config, question, tokenizer, device, property="one word or one phrase ", example_output="Example Output1: walking the dog ;Example Output2: ball ", num_frames=num_frames)
    answer = answer.lower()
    question = question.replace('"', '').lower()
    question = question.strip('?')
    split_qs = question.split(' ')
    verb_list = ["is", "are", "'s"]
    caption = ""
    try:
        if split_qs[0] == "what" and (split_qs[1] == "does" or split_qs[1] == "did" or split_qs[1] == "do"):
            new_list = split_qs[split_qs.index(split_qs[1]) + 1:]
            new_list[new_list.index("do")] = answer
            caption = ' '.join(new_list)
        elif split_qs[0] == "what":
            found = False
            for verb in verb_list:
                if verb in split_qs:
                    found = True
                    break
            if found:
                if "doing" in split_qs:
                    rest = ' '.join(split_qs[split_qs.index(verb) + 1:])
                    caption = rest.replace("doing", answer)
                elif "'s" in split_qs:
                    rest = ' '.join(split_qs[split_qs.index("'s") + 1:])
                    caption = f'{answer} {rest}'
                elif verb in split_qs:
                    rest = ' '.join(split_qs[split_qs.index(verb) + 1:])
                    caption = f'{rest} {verb} {answer}'
        elif split_qs[0] == "how" and split_qs[1] == "many":
            if "there" in split_qs:
                split_qs.remove("there")
            rest = ' '.join(split_qs[2:])
            if int(answer) == 0:
                caption = f'no {rest}'
            elif int(answer) == 1:
                caption = f'a {rest}'
            elif int(answer[0]) > 10:
                caption = f'many {rest}'
            else:
                caption = f'a few {rest}'
        elif split_qs[0] == "who":
            rest = ' '.join(split_qs[1:])
            caption = f'{answer} {rest}'
        elif split_qs[0] == "where":
            singular = "is" in split_qs
            verb = "is" if singular else "are"
            rest = ' '.join(split_qs[2:])
            caption = f'{rest} {verb} {answer}'
    except:
        pass
    return caption

def video_llava_qa_normal(model, video_tensor, model_config, question, tokenizer, device, num_frames=None, question_suffix=""):
    frames_num = model.get_video_tower().config.num_frames if num_frames is None else num_frames
    question = ' '.join([DEFAULT_IMAGE_TOKEN] * frames_num ) + '\n' + question
    if question_suffix:
        question += ' ' + question_suffix
    conv_mode = model_config.get('conv_mode', None)
    assert conv_mode is not None
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def video_llava_qa_short(model, video_tensor, model_config, question, tokenizer, device, property="one word or phrase ", num_frames=None, example_output=""):
    question_suffix = "Respond only with " + property + " that directly answers the question. " + example_output
    frames_num = model.get_video_tower().config.num_frames if num_frames is None else num_frames
    question = ' '.join([DEFAULT_IMAGE_TOKEN] * frames_num ) + '\n' + question + ' ' + question_suffix
    conv_mode = model_config.get('conv_mode', None)
    assert conv_mode is not None
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def video_llava_qa_short_image(model, image_tensor, model_config, question, tokenizer, device, property="one word or phrase ", num_frames=None, example_output=""):
    question_suffix = "Respond only with " + property + " that directly answers the question. " + example_output
    # frames_num = model.get_video_tower().config.num_frames if num_frames is None else num_frames
    question = ' '.join([DEFAULT_IMAGE_TOKEN]) + '\n' + question + ' ' + question_suffix
    conv_mode = model_config.get('conv_mode', None)
    assert conv_mode is not None
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def video_llava_plain_text_qa(model, model_config, question, tokenizer, device):
    conv_model = model_config.get('conv_mode', None)
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=None,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def video_llava_qa_debug(model, video_path, model_config, question, tokenizer, device, preprocessor):
    question = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + question
    conv_mode = model_config.get('conv_mode', None)
    assert conv_mode is not None
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    video_tensor = preprocessor(video_path, return_tensors='pt')['pixel_values']
    tensor = video_tensor.to(device, dtype=torch.float16)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
    
def generate_object_caption(video_frame, model, device, model_config, tokenizer, num_frames=None):
    video_tensor = torch.tensor(video_frame, dtype=torch.float16).to(device)
    question1 = 'What is the object ?'
    answer1 = video_llava_qa_short(model, video_tensor, model_config, question1, tokenizer, device, num_frames=num_frames)
    question2 = 'What color is the object ?'
    answer2 = video_llava_qa_short(model, video_tensor, model_config, question2, tokenizer, device, num_frames=num_frames)
    question3 = 'Where is the object ?'
    answer3 = video_llava_qa_short(model, video_tensor, model_config, question3, tokenizer, device, num_frames=num_frames)
    new_caption = f'{answer1} is {answer2} and {answer3}'
    return new_caption

def generate_augment_caption(video_frame, origin_caption, model, device, model_config, tokenizer, num_segment=2, ask_object=True, ask_regular=True):
    num_frames = video_frame.shape[2]
    step = num_frames // num_segment
    generate_cap = []
    for k in range(0, num_frames, step):
        if ask_regular:
            caption = generate_caption_vqa(video_frame[:,:,k:k+step], origin_caption, model, device, model_config, tokenizer, num_frames=step)
            generate_cap.append(caption)
        if ask_object:
            caption = generate_object_caption(video_frame[:,:,k:k+step], model, device, model_config, tokenizer, num_frames=step)
            generate_cap.append(caption)
    return generate_cap

def generate_augment_caption_auto(video_frame, model, device, model_config, tokenizer, question, num_segment=2):
    num_frames = video_frame.shape[2]
    step = num_frames // num_segment
    generate_cap = []
    for k in range(0, num_frames, step):
        caption = generate_caption_vqa_auto(video_frame[:,:,k:k+step], model, device, model_config, tokenizer, question, num_frames=step)
        generate_cap.append(caption)
    return generate_cap

def video_llava_get_video_caption(video_pixels, model, device, model_config, generation_tokenizer):
    question = "Use one sentence to describe the video. Make sure to write in concise and clear language."
    question = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + question
    conv_mode = model_config.get('conv_mode', None)
    assert conv_mode is not None
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, generation_tokenizer, input_ids)
    tensor = torch.tensor(video_pixels, dtype=torch.float16).to(device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = generation_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def generate_question_auto(selected_video_pixels, model, device, config, generation_tokenizer, query_text, caption_list=None):
    dialogue_config = config.get('dialogue_config', None)
    assert dialogue_config is not None, 'dialogue_config not found'
    use_caption = dialogue_config.get('use_caption', False)
    use_caption_cache = dialogue_config.get('use_caption_cache', False)
    if use_caption and not use_caption_cache:
        model_config = config.get('model_config', None)
        assert model_config is not None, 'model_config not found'
        video_num = len(selected_video_pixels)
        caption_list = []
        for i in range(video_num):
            cur_video_pixels = selected_video_pixels[i:i+1]
            cur_video_caption = video_llava_get_video_caption(cur_video_pixels, model, device, model_config, generation_tokenizer)
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
    question1 = video_llava_plain_text_qa(model, config, q1, generation_tokenizer, device)
    return question1

def get_video_features(test_video_dataloader, model, device):
    video_pbar = tqdm(total=len(test_video_dataloader), file=sys.stdout, position=0)
    all_video_features = []
    all_video_pixels = []
    all_video_path = []
    for data in test_video_dataloader:
        video_data, video_path, text = data
        all_video_pixels.append(video_data.numpy())
        if type(video_data) is list:
            video_tensor = [video.to(device, dtype=torch.float16) for video in video_data]
        else:
            video_tensor = video_data.to(device, dtype=torch.float16)
        with torch.inference_mode():
            video_forward_outs, video_features, model_logit = model.get_video_tower()(video_tensor, return_val="all_with_logit")
            video_pooler_output = video_forward_outs.pooler_output
            video_features = model.get_model().video_tower.retrieval_video_proj(video_pooler_output)
            video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
            video_features = video_features * model_logit.exp()
            all_video_features.append(video_features.cpu().detach().numpy())
            # all_video_pixels.append(video_data.cpu().detach().numpy())
        all_video_path.extend(video_path)
        video_pbar.update(1)
    all_video_features = np.concatenate(all_video_features, axis=0)
    all_video_pixels = np.concatenate(all_video_pixels, axis=0)
    video_pbar.close() 
    return all_video_features, all_video_pixels, all_video_path

def get_img_features(test_img_dataloader, model, device):
    img_pbar = tqdm(total=len(test_img_dataloader), file=sys.stdout, position=0)
    all_img_features = []
    all_img_pixels = []
    all_img_path = []
    for data in test_img_dataloader:
        img_data, img_path, text = data
        all_img_pixels.append(img_data.numpy())
        img_tensor = img_data.to(device, dtype=torch.float16)
        with torch.inference_mode():
            img_forward_outs, img_features, model_logit = model.get_image_tower()(img_tensor, return_val="all_with_logit")
            img_pooler_output = img_forward_outs.pooler_output
            img_features = model.get_model().image_tower.retrieval_image_proj(img_pooler_output)
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
            img_features = img_features * model_logit.exp()
            all_img_features.append(img_features.cpu().detach().numpy())
        all_img_path.extend(img_path)
        img_pbar.update(1)
    all_img_features = np.concatenate(all_img_features, axis=0)
    all_img_pixels = np.concatenate(all_img_pixels, axis=0)
    img_pbar.close()
    return all_img_features, all_img_pixels, all_img_path 

def generate_videomateinfo_reponse(conv_obj, user_input, tensor, generation_tokenizer, model,  device):
    conv_obj.append_message(conv_obj.roles[0], user_input)
    conv_obj.append_message(conv_obj.roles[1], None)
    prompt = conv_obj.get_prompt()
    input_ids = tokenizer_image_token(prompt, generation_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device)
    stopping_str = conv_obj.sep if conv_obj.sep_style != SeparatorStyle.TWO else conv_obj.sep2
    keywords = [stopping_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, generation_tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    output_text = generation_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().rstrip("</s>")
    return output_text
             
        
 
        