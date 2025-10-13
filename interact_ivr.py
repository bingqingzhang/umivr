import random
import json
import os
import sys
import logging
import time
import argparse
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollava.model.multimodal_encoder.languagebind import LanguageBindImageTokenizer

from retrievalmodel.dataset import VideoDataset, TextDataset, video_collate_fn, frame_selection
from retrievalmodel.registry import assemble_parameters_ivr, do_dialogue_retrieval

def setup_logger(log_dir='logs', log_level=logging.INFO, log_type=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, log_type + f'{time.strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger()
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG) 
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    
    return logger

def build_test_dataset_dataloader(dataset_config): 
    video_dataset = VideoDataset(dataset_config)
    video_dataloader = DataLoader(
        video_dataset,
        batch_size=dataset_config['video_batch_size'],
        shuffle=False,
        num_workers=dataset_config['video_num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda x: video_collate_fn(x, using_llava=dataset_config.get('using_llava_preprocess', False))
    )
    text_dataset = TextDataset(dataset_config)
    text_dataloader = DataLoader(
        text_dataset,
        batch_size=dataset_config['text_batch_size'],
        shuffle=False,
        num_workers=dataset_config['text_num_workers'],
        pin_memory=True,
        drop_last=False
    )
    return video_dataset, video_dataloader, text_dataset, text_dataloader

def reserve_all_available_gpu_memory(gpu_device="cuda:0"):
    if ":" in gpu_device:
        gpu_id = int(gpu_device.split(":")[-1])
    else:
        gpu_id = 0
    device = gpu_device
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    gc.collect()
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    already_allocated = torch.cuda.memory_allocated(gpu_id)
    cached_memory = torch.cuda.memory_reserved(gpu_id)
    available_memory = total_memory - already_allocated - cached_memory
    bytes_to_reserve = int(available_memory * 0.995)
    step_size = 0.05
    current_ratio = 1.0
    while current_ratio > 0:
        try:
            num_elements = bytes_to_reserve // 4
            reserved_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
            reserved_tensor.zero_()       
            actual_reserved = torch.cuda.memory_allocated(gpu_id) - already_allocated
            return reserved_tensor       
        except RuntimeError as e:
            current_ratio -= step_size
            bytes_to_reserve = int(available_memory * current_ratio)
    return None

def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="retrieval_config/ivr_auto_multiturn.json")
    parser.add_argument("--output-dir", default="./configs/retrieval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--absolute-file-path", default="/path/to/dataset_root", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--reverse", default=False, action='store_true')
    args = parser.parse_args()
    
    if args.reverse:
        reserved_memory = reserve_all_available_gpu_memory(gpu_device=args.device)
        if reserved_memory is None:
            print(f"not enough memory for {args.device}")
        
    config_file_path = args.config_file
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f'Config file not found: {config_file_path}')
    
    if config.get('dialogue_config', None) is not None:
        dialogue_type = config['dialogue_config'].get('dialogue_type', 'unknown')
        dialogue_type = dialogue_type.lower()
    else:
        dialogue_type = 'unknown'
        
    if config.get("test_dataset_config", None) is None and config.get("train_dataset_config", None) is None:
        raise ValueError("dataset_config not found in config file")
    dataset_name = config["test_dataset_config"].get("dataset_name", None)
    logger = setup_logger(log_dir='retrieval_logs',log_type=dialogue_type+'-'+dataset_name)
    logger.info(f"Start running dialogue retrieval with dialogue type: {dialogue_type}")
        
    
    if config['test_dataset_config'].get('video_path', None) is None:
        raise ValueError("video_path not found in dataset_config")
    config['test_dataset_config']['video_path'] = os.path.join(args.absolute_file_path, config['test_dataset_config']['video_path'])
    if config['test_dataset_config'].get('dataset_anno_path', None) is None:
        raise ValueError("dataset_anno_path not found in dataset_config")
    config['test_dataset_config']['dataset_anno_path'] = config['test_dataset_config']['dataset_anno_path']
    if config.get("test_dataset_config", None) is not None:
        test_video_dataset, test_video_dataloader, test_text_dataset, test_text_dataloader = build_test_dataset_dataloader(config["test_dataset_config"])
    assert config.get("model_config", None) is not None, "model_config not found in config file"
    model_config = config["model_config"]
    model_name = get_model_name_from_path(model_config.get("model_name", None))
    assert model_name is not None, "model_name not found in model_config"
    generation_tokenizer, model, processor, _ = load_pretrained_model(model_config["model_name"], None, model_name, model_config.get("load_8bit", False), model_config.get("load_4bit", False), device=args.device, cache_dir=model_config.get("cache_dir", "./cache_dir"))
    retrieval_tokenizer = LanguageBindImageTokenizer.from_pretrained('lb203/LanguageBind_Image', cache_dir=model_config.get("cache_dir", "./cache_dir/"))
    img_dataset = config['test_dataset_config'].get('image_dataset', False)
    if config.get("test_dataset_config", None) is not None and config['test_dataset_config'].get("using_llava_preprocess", False):
        if img_dataset:
            test_video_dataset.regist_video_prepocessor(processor['image'])
        else:
            test_video_dataset.regist_video_prepocessor(processor['video'])
    if config.get("dialogue_config", None) is None:
        raise ValueError("dialogue_config not found in config file")
    dialogue_config = config["dialogue_config"]
    dialogue_type = dialogue_config.get("dialogue_type", None)
    if dialogue_type is None:
        raise ValueError("dialogue type not found in dialogue_config")
    
    assembled_params = assemble_parameters_ivr(type=dialogue_type, test_video_dataloader=test_video_dataloader, test_text_dataloader=test_text_dataloader, \
                                                model=model, config=config, retrieval_tokenizer=retrieval_tokenizer, args=args, processor=processor, \
                                                generation_tokenizer=generation_tokenizer)
    iqa_config = config['test_dataset_config'].get('iqa_config', None)
    if iqa_config is not None:
        logger.info("Start frame quality assessment")
        stat = do_iqa_geration(iqa_config, test_video_dataset.anno_data, test_video_dataset.video_base_path, test_video_dataset.video_fmt, model, processor['image'], args.device)
        if stat == 1:
            logger.info("Frame quality assessment finished")
        else:
            logger.info("Frame quality assessment using cache file")
        test_video_dataset.load_iqa_file()
    return do_dialogue_retrieval(dialogue_type, **assembled_params)

def do_iqa_geration(iqa_config, anno_data, video_base_path, video_fmt, model, preprocess, device):
    file_path = iqa_config['file_path']
    using_cache = iqa_config.get('using_cache', False)
    num_frame = iqa_config.get('num_frame', 8)
    if (not using_cache) or (not os.path.exists(file_path)):
        total_videos = len(anno_data)
        pbar = tqdm(total=total_videos, file=sys.stdout, position=0)
        all_data = {}
        for i in range(len(anno_data)):
            clip_id = anno_data[i]['clip_id']
            cur_video_path = os.path.join(video_base_path, f'{clip_id}.{video_fmt}')
            frame_index_list = frame_selection(cur_video_path, num_frame, device, model, preprocess)
            # all_data.append({'clip_id': clip_id, 'frame_index_list': frame_index_list})
            all_data[cur_video_path] = frame_index_list
            pbar.update(1)
        pbar.close()
        with open(file_path, 'w') as f:
            json.dump(all_data, f)
        return 1
    else:
        return 0
    
    
if __name__ == "__main__":
    main()
    




