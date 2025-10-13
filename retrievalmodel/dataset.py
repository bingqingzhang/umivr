import torch
import os
import numpy as np
from torch.utils.data import Dataset
import json

import decord
from decord import VideoReader

import cv2
from sklearn.cluster import KMeans
from PIL import Image
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path
from videollava.model.builder import load_pretrained_model
from torchvision import transforms


def extract_frames(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open file: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = target_fps
    frame_interval = int(round(video_fps / target_fps))
    
    frames = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            timestamp = frame_index / video_fps
            frames.append((frame_index, timestamp, frame))
        frame_index += 1
    cap.release()
    return frames

def compute_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def select_uniform_candidates(frames, num_candidates=24):
    if not frames:
        return []
    start_time = frames[0][1]
    end_time = frames[-1][1]
    bins = np.linspace(start_time, end_time, num_candidates + 1)
    candidate_list = []
    for i in range(num_candidates):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        bin_frames = []
        for (frame_index, t, frame) in frames:
            if bin_start <= t < bin_end:
                quality = compute_quality(frame)
                bin_frames.append((frame_index, t, frame, quality))
        if len(bin_frames) == 0:
            continue
        best_frame = max(bin_frames, key=lambda x: x[3])
        candidate_list.append(best_frame)
    return candidate_list

def extract_clip_features(candidate_list, model, preprocess, device):
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
    ])
    features = []
    for (frame_index, timestamp, frame, quality) in candidate_list:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        image_tensor = transform(pil_img)
        image_input = image_tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            image_forward_outs, image_features, model_logit = model.get_image_tower()(image_input, return_val="all_with_logit")
            image_pooler_output = image_forward_outs.pooler_output
            image_features = model.get_model().image_tower.retrieval_image_proj(image_pooler_output)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            image_features = image_features * model_logit.exp()
            features.append(image_features.cpu().numpy().flatten())
    features = np.stack(features)
    return features

def cluster_candidates(features, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

def select_final_frames(candidate_list, cluster_labels):
    cluster_dict = {}
    for idx, (frame_index, timestamp, frame, quality) in enumerate(candidate_list):
        label = cluster_labels[idx]
        if label not in cluster_dict or quality > cluster_dict[label][3]:
            cluster_dict[label] = (frame_index, timestamp, frame, quality)
    final_frames = list(cluster_dict.values())
    final_frames.sort(key=lambda x: x[1])
    return final_frames

def frame_selection(video_path, num_frame, device, model, preprocess):
    try:
        frames = extract_frames(video_path, target_fps=5)
        if len(frames) == 0:
            raise ValueError("No frames extracted!")
        
        candidates = select_uniform_candidates(frames, num_candidates=max((len(frames) // 3), 3 * num_frame))
        if len(candidates) == 0:
            raise ValueError("No candidate frames found!")
        
        features = extract_clip_features(candidates, model, preprocess, device)
        
        actual_num_frame = min(num_frame, len(features))
        
        if actual_num_frame < 2:
            raise ValueError("Too few samples for clustering")
            
        cluster_labels = cluster_candidates(features, n_clusters=actual_num_frame)
        final_frames = select_final_frames(candidates, cluster_labels)
        
        frame_index_list = []
        for (frame_index, timestamp, frame, quality) in final_frames:
            frame_index_list.append(frame_index)
        
        return frame_index_list
        
    except Exception as e:
        print(f"Clustering failed: {e}. Falling back to uniform sampling.")
        frames = extract_frames(video_path, target_fps=5)
        if len(frames) == 0:
            return []

        total_frames = len(frames)
        if total_frames <= num_frame:
            return [frame[0] for frame in frames]
        else:
            indices = np.linspace(0, total_frames - 1, num_frame, dtype=int)
            return [frames[i][0] for i in indices]

def frame_selection_no_semantic(video_path, num_frame=8):
    frames = extract_frames(video_path, target_fps=5)
    if len(frames) == 0:
        raise ValueError("No frames extracted!")

    frame_with_quality = []
    for (frame_index, timestamp, frame) in frames:
        q = compute_quality(frame)
        frame_with_quality.append((frame_index, timestamp, frame, q))
    frame_with_quality.sort(key=lambda x: x[3], reverse=True)
    top_k_frames = frame_with_quality[:num_frame]
    top_k_frames.sort(key=lambda x: x[1])
    frame_index_list = [item[0] for item in top_k_frames]
    return frame_index_list

def video_collate_fn(batch, using_llava=False):
    if not using_llava:
        return batch
    else:
        video_tensor_list = []
        video_path_list = []
        text_list = []
        for item in batch:
            video_tensor_list.append(item['video_tensor'])
            video_path_list.append(item['video_path'])
            text_list.append(item['text'])
        return torch.cat(video_tensor_list, dim=0), video_path_list, text_list
    

class ImageNorm(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)

class VideoDataset(Dataset):
    def __init__(self, dataset_config):
        self.dataset_name = dataset_config.get('dataset_name', None)
        assert self.dataset_name is not None, 'Dataset name not found'
        self_dataset_name = self.dataset_name.lower() + "_video"
        self.anno_data = []
        load_format = dataset_config.get('anno_file_format', None)
        if load_format == 'jsonl':
            dataset_anno_path = dataset_config.get('dataset_anno_path', None)
            assert dataset_anno_path is not None and os.path.exists(dataset_anno_path), f'File not found: {dataset_anno_path}'
            with open(dataset_anno_path, 'r') as f:
                for line in f:
                    self.anno_data.append(json.loads(line))
        else:
            raise ValueError(f'Unsupported load format: {load_format}')
        
        self.video_base_path = dataset_config.get('video_path', None)
        self.video_fmt = dataset_config.get('video_fmt', None)
        assert self.video_fmt is not None, 'Video format not found'
        assert self.video_base_path is not None and \
            os.path.exists(self.video_base_path), f'File not found: {self.video_base_path}'
            
        self.iqa_config = dataset_config.get('iqa_config', None)
        
        self.using_llava_preprocess = dataset_config.get('using_llava_preprocess', False)
        self.image_dataset = dataset_config.get('image_dataset', False)
        self.preprocess_init = False
        if not self.using_llava_preprocess:
            assert dataset_config.get('video_preprocess', None) is not None, 'Video preprocessor not found'
            video_prepross_config = dataset_config['video_preprocess']
            self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            self.sample_num_frame = video_prepross_config.get('sample_num_frame', 8)
            self.sample_strategy = video_prepross_config.get('sample_strategy', 'uniform')
            self.preprocess_init = True
    
    def regist_video_prepocessor(self, video_preprocessor):
        assert self.using_llava_preprocess is True and self.preprocess_init is False, 'Preprocessor has been initialized'
        self.video_preprocessor = video_preprocessor
        self.preprocess_init = True
        
    def load_iqa_file(self):
        iqa_file_path = self.iqa_config.get('file_path', None)
        assert iqa_file_path is not None and os.path.exists(iqa_file_path), f'File not found: {iqa_file_path}'
        with open(iqa_file_path, 'r') as f:
            iqa_data = json.load(f)
        self.iqa_data = iqa_data
    
    def __len__(self):
        return len(self.anno_data)

    def _load_video(self, video_path):
        assert os.path.exists(video_path), f'File not found: {video_path}'
        vr = VideoReader(video_path)
        video_len = len(vr)
        
        if self.sample_strategy == 'uniform':
            indices = np.arange(0, video_len, video_len / self.sample_num_frame).astype(int)
        elif self.sample_strategy == 'random':
            indices = np.random.choice(video_len, self.sample_num_frame, replace=False)
        elif self.sample_strategy == 'iqa':
            indices = self.iqa_data[video_path]
            indices = np.array(indices)
        else:
            raise ValueError(f'Unsupported sample strategy: {self.sample_strategy}')
        raw_sample_frms = vr.get_batch(indices)  
        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)
        
        return raw_sample_frms
    
    def load_img(self, img_path, text):
        image_tensor = self.video_preprocessor(img_path, return_tensors='pt')['pixel_values']
        return {"video_tensor": image_tensor, "video_path": img_path, "text": text}
        
    
    def __getitem__(self, idx):
        data = self.anno_data[idx]
        clip_id = data['clip_id']
        text = data['text']
        video_path = os.path.join(self.video_base_path, f'{clip_id}.{self.video_fmt}')
        if self.image_dataset:
            return self.load_img(video_path, text)
        assert self.preprocess_init is True, 'Preprocessor not initialized'
        if self.using_llava_preprocess:
            video_tensor = self.video_preprocessor(video_path, return_tensors='pt')['pixel_values']
        elif self.using_llava_preprocess is False:
            video_tensor = self._load_video(video_path)
            video_tensor = self.img_norm(video_tensor.float())
            
        return {"video_tensor": video_tensor, "video_path": video_path, "text": text}
    

class TextDataset(Dataset):
    def __init__(self, dataset_config):
        self.dataset_name = dataset_config.get('dataset_name', None)
        assert self.dataset_name is not None, 'Dataset name not found'
        self_dataset_name = self.dataset_name.lower() + "_text"
        assert dataset_config.get('text_batch_size', None) is not None \
            and dataset_config['text_batch_size'] == 1, 'Text batch size not found'
        self.anno_data = []
        load_format = dataset_config.get('anno_file_format', None)
        if load_format == 'jsonl':
            dataset_anno_path = dataset_config.get('dataset_anno_path', None)
            assert dataset_anno_path is not None and os.path.exists(dataset_anno_path), f'File not found: {dataset_anno_path}'
            with open(dataset_anno_path, 'r') as f:
                for line in f:
                    self.anno_data.append(json.loads(line))
        else:
            raise ValueError(f'Unsupported load format: {load_format}')
    
    def __len__(self):
        return len(self.anno_data)
    
    def __getitem__(self, idx):
        data = self.anno_data[idx]
        text = data['text']
        return text

