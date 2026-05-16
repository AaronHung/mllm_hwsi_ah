import time
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Dict, List, Optional
from conch.open_clip_custom import get_tokenizer, tokenize

def read_slide_report(reports_path: Path, slide_id: str) -> List[Dict]:
    """Read JSONL report rows where question_id contains slide_id."""
    matches: List[Dict] = []
    with reports_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_id = str(row.get("question_id", ""))
            if slide_id in question_id:
                row["_line_num"] = line_num
                matches.append(row)
    if not matches:
        print(f"No report rows found for slide_id: {slide_id}")
        return [],[]
    else:
        return matches[0].get("question", []), matches[0].get("T-answer", [])

def compute_patch_similarity(x, window_size):
        """Compute similarity between patches within sliding windows"""
        N, D = x.shape
        x_norm = nn.functional.normalize(x, p=2, dim=-1)
        
        similarities = []
        selected_indices = []
        
        for i in range(0, N, window_size):
            window = x_norm[i:i+window_size]
            
            # Skip if window is too small
            if len(window) < 2:
                selected_indices.append(torch.arange(i, min(i+window_size, N), device=x.device))
                continue
                
            # Local similarity computation
            window_sim = torch.mm(window, window.t())
            
            # Adaptive thresholding
            # Only compute std if we have enough samples
            if window_sim.numel() > 1:
                threshold = window_sim.mean() + window_sim.std(unbiased=False)  # use biased std
            else:
                threshold = window_sim.mean()  # fallback to just mean if not enough samples
            
            # Select non-redundant patches
            redundant = window_sim.mean(1) > threshold
            keep_indices = torch.where(~redundant)[0] + i
            
            # If all patches are marked as redundant, keep at least one
            if len(keep_indices) == 0:
                keep_indices = torch.tensor([i], device=x.device)
                
            selected_indices.append(keep_indices)
            similarities.append(window_sim)
        
        # Handle case where no indices were selected
        if not selected_indices:
            return [], torch.arange(N, device=x.device)
            
        return similarities, torch.cat(selected_indices)

def adaptive_token_selection(features, text_features, window_size):
        """Select tokens based on text relevance and local structure"""
        N, D = features.shape
        L_max = 64
        
        # Compute similarities and get initial selection
        similarities, indices = compute_patch_similarity(features, window_size)
        
        # Project features to match text_features dimension if needed
        if features.shape[-1] != text_features.shape[-1]:
            projection = nn.Linear(features.shape[-1], text_features.shape[-1], device=features.device)
            features_projected = projection(features)
        else:
            features_projected = features
        
        # Text-guided importance scoring
        text_relevance = torch.matmul(features_projected, text_features.T).mean(-1)
        
        # Create importance mask
        importance_mask = torch.zeros(N, device=features.device)
        importance_mask[indices] = text_relevance[indices]
        
        # Select top tokens
        num_tokens = min(L_max, N)
        _, selected_indices = torch.topk(importance_mask, num_tokens)
        selected_indices, _ = torch.sort(selected_indices)  # maintain sequence order
        
        selected_features = features[selected_indices]
        
        return selected_features, selected_indices

def semantic_patch_filtering(features, reports_path, slide_id, conch_model, 
                             device, filtering_window_size, top_k=5):

    features = features.to(device)
    question, answer = read_slide_report(Path(reports_path), slide_id)

    tokenizer = get_tokenizer()
    answer_tokens = tokenize(texts=[answer], tokenizer=tokenizer).to(device)

    with torch.inference_mode():
        text_features = conch_model.encode_text(answer_tokens)
    
    selected_patch_features, selected_patch_indices = [], []

    for i in range(features.shape[0]):
        patch_feat = features[i].squeeze() 
        selected_features, selected_indices = \
            adaptive_token_selection(patch_feat, text_features, filtering_window_size)
        selected_patch_features.append(selected_features[:top_k])
        selected_patch_indices.append(selected_indices[:top_k])

    return torch.stack(selected_patch_features, dim=0), torch.stack(selected_patch_indices, dim=0)


def select_dissimilar_features(x, n, seed_idx=None, return_full_similarity=False):
    """
    Select n dissimilar patch features using global cosine similarity.

    Args:
        x (torch.Tensor): [N, D] feature matrix
        n (int): number of features to select
        seed_idx (int or None): optional starting index
        return_full_similarity (bool): whether to return full NxN similarity matrix

    Returns:
        dict with:
            - selected_indices: [n_selected]
            - selected_scores: [n_selected]
            - selected_pairwise_similarity: [n_selected, n_selected]
            - full_similarity: [N, N] or None
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [N, D], got shape {tuple(x.shape)}")

    N, D = x.shape
    if N == 0:
        return {
            "selected_indices": torch.empty(0, dtype=torch.long, device=x.device),
            "selected_scores": torch.empty(0, dtype=x.dtype, device=x.device),
            "selected_pairwise_similarity": torch.empty(0, 0, dtype=x.dtype, device=x.device),
            "full_similarity": torch.empty(0, 0, dtype=x.dtype, device=x.device) if return_full_similarity else None,
        }

    n = min(n, N)

    x_norm = F.normalize(x, p=2, dim=-1)
    sim = x_norm @ x_norm.T  # [N, N]

    # Choose seed: least similar to all others on average
    if seed_idx is None:
        mean_sim = (sim.sum(dim=1) - 1.0) / max(N - 1, 1)  # remove self-similarity
        seed_idx = torch.argmin(mean_sim).item()

    selected = [seed_idx]
    selected_scores = [sim[seed_idx].mean()]  # seed score

    max_sim_to_selected = sim[seed_idx].clone()
    max_sim_to_selected[seed_idx] = float("inf")

    for _ in range(1, n):
        next_idx = torch.argmin(max_sim_to_selected).item()
        selected.append(next_idx)
        selected_scores.append(max_sim_to_selected[next_idx])

        max_sim_to_selected = torch.maximum(max_sim_to_selected, sim[next_idx])
        max_sim_to_selected[selected] = float("inf")

    selected_indices = torch.tensor(selected, device=x.device, dtype=torch.long)
    selected_scores = torch.stack([
        s if torch.is_tensor(s) else torch.tensor(s, device=x.device, dtype=x.dtype)
        for s in selected_scores
    ])

    selected_pairwise_similarity = sim[selected_indices][:, selected_indices]

    return {
        "selected_indices": selected_indices,
        "selected_scores": selected_scores,
        "selected_pairwise_similarity": selected_pairwise_similarity,
        "full_similarity": sim if return_full_similarity else None,
    }

def select_topk_by_text_similarity(selected_features, selected_indices, text_feature, k):
    """
    Select top-k patch features most similar to a text feature using cosine similarity.

    Args:
        selected_features (torch.Tensor): [M, D]
        selected_indices (torch.Tensor): [M] original indices of selected_features
        text_feature (torch.Tensor): [D] or [1, D]
        k (int): number of top features to select

    Returns:
        dict with:
            - topk_features: [k, D]
            - topk_similarities: [k]
            - topk_local_indices: [k] indices within selected_features
            - topk_original_indices: [k] indices from original selected_indices
    """
    if selected_features.ndim != 2:
        raise ValueError(f"selected_features must have shape [M, D], got {tuple(selected_features.shape)}")

    if selected_indices.ndim != 1:
        raise ValueError(f"selected_indices must have shape [M], got {tuple(selected_indices.shape)}")

    if selected_features.size(0) != selected_indices.size(0):
        raise ValueError(
            f"selected_features and selected_indices must match on first dimension, "
            f"got {selected_features.size(0)} and {selected_indices.size(0)}"
        )

    if text_feature.ndim == 1:
        text_feature = text_feature.unsqueeze(0)   # [1, D]
    elif text_feature.ndim != 2 or text_feature.size(0) != 1:
        raise ValueError(f"text_feature must have shape [D] or [1, D], got {tuple(text_feature.shape)}")

    if selected_features.size(1) != text_feature.size(1):
        raise ValueError(
            f"Feature dimensions must match, got {selected_features.size(1)} and {text_feature.size(1)}"
        )

    M = selected_features.size(0)
    if M == 0:
        return {
            "topk_features": torch.empty(0, selected_features.size(1), device=selected_features.device, dtype=selected_features.dtype),
            "topk_similarities": torch.empty(0, device=selected_features.device, dtype=selected_features.dtype),
            "topk_local_indices": torch.empty(0, device=selected_features.device, dtype=torch.long),
            "topk_original_indices": torch.empty(0, device=selected_features.device, dtype=selected_indices.dtype),
        }

    k = min(k, M)

    # Normalize features
    selected_features_norm = F.normalize(selected_features, p=2, dim=-1)   # [M, D]
    text_feature_norm = F.normalize(text_feature, p=2, dim=-1)             # [1, D]

    # Cosine similarity: [M]
    similarities = torch.matmul(selected_features_norm, text_feature_norm.t()).squeeze(1)

    # Top-k most similar
    topk_similarities, topk_local_indices = torch.topk(similarities, k=k, largest=True)

    # Map back to original indices
    topk_original_indices = selected_indices[topk_local_indices]

    # Selected features
    topk_features = selected_features[topk_local_indices]

    return {
        "topk_features": topk_features,
        "topk_similarities": topk_similarities,
        "topk_local_indices": topk_local_indices,
        "topk_original_indices": topk_original_indices,
    }


def semantic_patch_filtering2(features, reports_path, slide_id, conch_model, 
                             device, n_diss_features=32, top_k=16):

    features = features.to(device)
    question, answer = read_slide_report(Path(reports_path), slide_id)

    tokenizer = get_tokenizer()
    answer_tokens = tokenize(texts=[answer], tokenizer=tokenizer).to(device)

    with torch.inference_mode():
        text_features = conch_model.encode_text(answer_tokens)
    
    selected_patch_features, selected_patch_indices = [], []

    for i in range(features.shape[0]):
        patch_feat = features[i].squeeze() 

        # Compute dissimilar features and their indices
        dissimilarity_results = select_dissimilar_features(patch_feat, n_diss_features, return_full_similarity=True)
        selected_dissimilar_indices = dissimilarity_results["selected_indices"]
        selected_dissimilar_features = patch_feat[selected_dissimilar_indices]
        
        # Compute report similarity
        report_similarity_results = select_topk_by_text_similarity(selected_dissimilar_features, 
                                                                   selected_dissimilar_indices, text_features, top_k)
        
        selected_patch_features.append(report_similarity_results["topk_features"])
        selected_patch_indices.append(report_similarity_results["topk_original_indices"])

    return torch.stack(selected_patch_features, dim=0), torch.stack(selected_patch_indices, dim=0)
