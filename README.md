<img src="./assets/Logo.PNG" align="left" width="60" height="60" alt="Project Logo">

# MLLM-HWSI: A Multimodal Large Language Model for Hierarchical Whole Slide Image Understanding

**Repository under construction...**

- Official repository for our paper MLLM-HWSI: A Multimodal Large Language Model for Hierarchical Whole Slide Image Understanding

* <a href="https://https://arxiv.org/abs/2603.23067"><img src="https://img.shields.io/badge/arXiv-Paper_Link-blue"></a> 

* <img src="https://img.shields.io/badge/Accepted-CVPR_2026-success?style=for-the-badge&logo=academia&logoColor=white" alt="Accepted at CVPR 2026">

## Model Overview

![Model Overview](./assets/overview.png)

## Abstract

Whole Slide Images (WSIs) exhibit hierarchical structure,
where diagnostic information emerges from cellular morphology, regional tissue organization, and global context. Existing Computational Pathology (CPath) Multimodal Large Language Models (MLLMs) typically compress an entire WSI into a single embedding, which hinders fine-grained grounding and ignores how pathologists
synthesize evidence across different scales. We introduce
MLLM-HWSI, a Hierarchical WSI-level MLLM that aligns
visual features with pathology language at four distinct
scales, cell as word, patch as phrase, region as sentence,
and WSI as paragraph to support interpretable evidencegrounded reasoning. MLLM-HWSI decomposes each WSI
into multi-scale embeddings with scale-specific projectors
and jointly enforces (i) a hierarchical contrastive objective
and (ii) a cross-scale consistency loss, preserving semantic
coherence from cells to the WSI. We compute diagnostically
relevant patches and aggregate segmented cell embeddings
into a compact cellular token per-patch using a lightweight
Cell–Cell Attention Fusion (CCAF) transformer. The projected multi-scale tokens are fused with text tokens and
fed to an instruction-tuned LLM for open-ended reasoning,
VQA, report, and caption generation tasks. Trained in three
stages, MLLM-HWSI achieves new SOTA results on 13 WSIlevel benchmarks across six CPath tasks. By aligning language with multi-scale visual evidence, MLLM-HWSI provides accurate, interpretable outputs that mirror diagnostic
workflows and advance holistic WSI understanding.

## Environment Setup

## Inference

## Training

## Acknowledgements

## Citation
