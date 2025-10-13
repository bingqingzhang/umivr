import numpy as np
import pandas as pd
import torch
import json
import os
import random
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans

import nltk
import math
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class TextDataset(Dataset):
    def __init__(self, text_list):
        self.text_list = text_list
        
    def __len__(self):
        return len(self.text_list)
    
    def __getitem__(self, idx):
        return self.text_list[idx]
    
def collate_fn(batch):
    return batch

def filter_text(dataset_file, batchsize):
    print('Loading datatset annotation...')
    with open(dataset_file, 'r') as f:
        vidal_anno = json.load(f)
    print('extracting text data...')
    df = pd.DataFrame.from_dict(vidal_anno, orient='index')
    df = df.reset_index().rename(columns={'index': 'id'})
    text_corpus = [random.choice(sublist) for sublist in df['ofa'].tolist()]
    print('text data extracted')
    return text_corpus

def get_caption(dataset_file):
    print('Loading datatset annotation...')
    with open(dataset_file, 'r') as f:
        caption_anno = json.load(f)
    print('extracting text data...')
    caption_list = []
    for cur_anno in caption_anno:
        caption_list.append(cur_anno['caption'])
    print('text data extracted')
    return caption_list

def embedding_text(text_corpus_list, model, retrieval_tokenizer, batchsize, device, output_dir):
    embeddings_list = []
    print("Start embedding texts ...")
    # pbar = tqdm(total=len(dataset), desc="Processing", unit="text")
    for i in tqdm(range(0, len(text_corpus_list), batchsize), desc="Embedding texts"):
        batch_text = text_corpus_list[i:i+batchsize]
        tokens = retrieval_tokenizer(batch_text, truncation=True, padding=True)
        input_ids = torch.tensor(tokens['input_ids'],  dtype=torch.long).to(device)
        with torch.inference_mode():
            outputs = model.get_video_tower().video_tower_text_encoder(input_ids=input_ids, output_hidden_states=True)
            text_pooler_output = outputs.pooler_output
            text_features = model.get_model().video_tower.retrieval_text_proj(text_pooler_output)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        embeddings_list.append(text_features.cpu())
    #     pbar.update(len(batch_text))
    # pbar.close()
    all_embeddings = torch.cat(embeddings_list, dim=0)
    print(f"All embeddings shape: {all_embeddings.shape}")
    torch.save(all_embeddings, os.path.join(output_dir, "text_embeddings.pt"))
    
def quick_cluster_sampling(text_corpus, corpus_embeddings):
    L = len(text_corpus)
    target_num = 100000
    if L > target_num:
        embeddings_np = corpus_embeddings.cpu().detach().numpy()
        kmeans = MiniBatchKMeans(n_clusters=target_num, random_state=42, batch_size=1000, verbose=1)
        kmeans.fit(embeddings_np)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        representative_indices = []

        for cluster in range(target_num):
            cluster_indices = np.where(labels == cluster)[0]
            if len(cluster_indices) > 0:
                cluster_points = embeddings_np[cluster_indices]
                distances = np.linalg.norm(cluster_points - centers[cluster], axis=1)
                best_idx = cluster_indices[np.argmin(distances)]
                representative_indices.append(best_idx)
        if len(representative_indices) < target_num:
            all_indices = set(range(L))
            selected_set = set(representative_indices)
            remaining_indices = list(all_indices - selected_set)
            num_to_add = target_num - len(representative_indices)
            extra_indices = np.random.choice(remaining_indices, size=num_to_add, replace=False).tolist()
            representative_indices.extend(extra_indices)
        representative_indices = sorted(representative_indices)
        sampled_text_corpus = [text_corpus[i] for i in representative_indices]
        sampled_embeddings = corpus_embeddings[representative_indices, :]
        
        return sampled_text_corpus, sampled_embeddings
    else:
        return text_corpus, corpus_embeddings
    
def sigmoid_normalize(entropy, k=3, threshold=4):
    midpoint = threshold / 2
    return 1 / (1 + math.exp(-k * (entropy - midpoint)))

def compute_semantic_uncertainty(text_query, text_corpus, corpus_embeddings, model, retrieval_tokenizer, device, top_k=5, sim_threshold=0.65, downsample=False, L_min=8, delta=0.2, enhanced=True):
    if downsample:
        text_corpus, corpus_embeddings = downsample_corpus(text_corpus, corpus_embeddings)
    query_tokens = retrieval_tokenizer([text_query], truncation=True, padding=True)
    input_ids = torch.tensor(query_tokens['input_ids'],  dtype=torch.long).to(device)
    with torch.inference_mode():
        outputs = model.get_video_tower().video_tower_text_encoder(input_ids=input_ids, output_hidden_states=True)
        text_pooler_output = outputs.pooler_output
        text_features = model.get_model().video_tower.retrieval_text_proj(text_pooler_output)
        query_embeddings = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    query_embeddings = query_embeddings.float().cpu()
    corpus_embeddings = corpus_embeddings.float().cpu()
    cos_scores = torch.nn.functional.cosine_similarity(query_embeddings, corpus_embeddings)
    assert len(text_corpus) == corpus_embeddings.shape[0]
    k = min(top_k, corpus_embeddings.shape[0])
    top_results = torch.topk(cos_scores,k=k)
    candidate_indices = top_results.indices.cpu().tolist()
    candidate_texts = [text_corpus[i] for i in candidate_indices]
    candidate_scores = top_results.values
    
    candidate_probs = torch.nn.functional.softmax(candidate_scores, dim=0)
    candidate_probs = candidate_probs.cpu().tolist()
    
    candidate_embeddings = corpus_embeddings[candidate_indices]
    
    clusters = []
    for i, text in enumerate(candidate_texts):
        assigned = False
        emb = candidate_embeddings[i]
        prob = candidate_probs[i]
        for cluster in clusters:
            rep_text, rep_prob, rep_emb = cluster[0]
            sim = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), rep_emb.unsqueeze(0)).item()
            if sim >= sim_threshold:
                cluster.append((text, prob, emb))
                assigned = True
                break
        if not assigned:
            clusters.append([(text, prob, emb)])
            
    cluster_probs = [sum(item[1] for item in cluster) for cluster in clusters]
    total_p = sum(cluster_probs)
    if total_p == 0:
        return math.log(len(clusters) + 1e-12), clusters, cluster_probs
    
    cluster_probs = [p / total_p for p in cluster_probs]
    semantic_entropy = - sum(p * math.log(p + 1e-12) for p in cluster_probs)
    semantic_entropy_bits = semantic_entropy / math.log(2)
    semantic_entropy_bits = sigmoid_normalize(semantic_entropy_bits)
    if enhanced:
        uspec = compute_query_specificity_adjusted(text_query, L_min=L_min)
        semantic_entropy_bits = delta * semantic_entropy_bits + (1-delta) * uspec
    return semantic_entropy_bits

def downsample_corpus(text_corpus, corpus_embeddings, target_num=100000, method='uniform'):
    L = len(text_corpus)
    if L > target_num:
        if method == 'uniform':
            indices = np.linspace(0, L - 1, target_num, dtype=int)
        elif method == 'random':
            indices = np.random.choice(L, size=target_num, replace=False)
            indices.sort()
        else:
            raise ValueError("Unsupported sampling method. Use 'uniform' or 'random'.")
        
        sampled_text_corpus = [text_corpus[i] for i in indices]
        sampled_embeddings = corpus_embeddings[indices, :]
        return sampled_text_corpus, sampled_embeddings
    else:
        return text_corpus, corpus_embeddings

def get_named_entity_indices(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    ne_tree = ne_chunk(tagged)
    named_indices = set()
    idx = 0
    for node in ne_tree:
        if isinstance(node, nltk.Tree):
            num_leaves = len(node.leaves())
            for i in range(idx, idx + num_leaves):
                named_indices.add(i)
            idx += num_leaves
        else:
            idx += 1
    return named_indices

def is_important_verb(token, pos):
    if not pos.startswith('VB'):
        return False
    aux_verbs = {'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
                 'do', 'does', 'did', 'have', 'has', 'had'}
    return token.lower() not in aux_verbs


def compute_query_specificity_adjusted(sentence, 
                                       L_min=8,
                                       specificity_constant=21.31,
                                       adj_weight=3.0,
                                       adv_weight=2.0,
                                       noun_weight=0.5,
                                       verb_weight=0.5,
                                       entity_weight=1.0):
    tokens = word_tokenize(sentence)
    total_tokens = len(tokens)
    tagged_tokens = pos_tag(tokens)
    ne_indices = get_named_entity_indices(sentence)
    
    descriptive_total = 0.0
    for i, (token, pos) in enumerate(tagged_tokens):
        if pos.startswith('JJ'):
            descriptive_total += adj_weight
        elif pos.startswith('RB'):
            descriptive_total += adv_weight
        elif pos.startswith('NN'):
            descriptive_total += noun_weight
        elif pos.startswith('VB'):
            if is_important_verb(token, pos):
                descriptive_total += verb_weight
        
        if i in ne_indices:
            descriptive_total += entity_weight
    if total_tokens < L_min:
        length_factor = total_tokens / L_min
    else:
        length_factor = 1 + 0.5 * math.log(total_tokens - L_min + 1)
    effective_score = descriptive_total * length_factor
    specificity = effective_score / (effective_score + specificity_constant)
    U_spec = 1 - specificity
    return U_spec

# mus
def custom_normalize(scores, cutoff=None, p=2.0):
    scores = np.array(scores)
    if cutoff is None:
        cutoff = np.mean(scores)
    s_max = scores[0]
    if abs(s_max - cutoff) < 1e-8:
        return np.ones_like(scores) / len(scores)
    new_scores = np.array([
        ((s - cutoff) / (s_max - cutoff))**p if s >= cutoff else 0.0 
        for s in scores
    ])
    total = new_scores.sum()
    if total == 0:
        normalized = np.ones_like(new_scores) / len(new_scores)
    else:
        normalized = new_scores / total
    return normalized

def kl_divergence(p, q):
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def max_jsd(k):
    q = np.zeros(k)
    q[0] = 1.0
    p = np.ones(k) / k
    m = 0.5 * (q + p)
    jsd_max = 0.5 * (kl_divergence(q, m) + kl_divergence(p, m))
    return jsd_max

def matching_score(scores, cutoff=None, p=2.0):
    scores = np.array(scores)
    p_dist = custom_normalize(scores, cutoff, p)
    q = np.zeros_like(p_dist)
    q[np.argmax(scores)] = 1.0
    jsd = js_divergence(p_dist, q)
    k = len(scores)
    jsd_max = max_jsd(k)
    normalized_jsd = jsd / jsd_max if jsd_max > 0 else 0
    normalized_jsd = 1.0 if normalized_jsd > 1.0 else normalized_jsd
    return p_dist, normalized_jsd