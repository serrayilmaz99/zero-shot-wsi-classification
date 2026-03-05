# Zero-Shot WSI Classification

This repository contains the zero-shot classification component of a larger project on multimodal whole-slide image (WSI) analysis.

The script performs slide-level classification without additional training by comparing image embeddings with textual embeddings derived from pathology reports and class descriptions. Cosine similarity between these embeddings is computed and combined through weighted scoring to predict the most likely class.

The implementation supports evaluation using pretrained models and extracted WSI features.

### Files
- `zero_shot_classifier.py` — Zero-shot classification and evaluation script.
