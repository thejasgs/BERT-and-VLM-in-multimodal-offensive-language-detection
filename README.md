# BERT and Local Vision Language Models in Hate Speech Detection on Small Multimodal Datasets

This repository contains the code for the paper **"BERT and Local Vision Language Models in Hate Speech Detection on Small Multimodal Datasets"** (IEEE Access, 2026). DOI: 10.1109/ACCESS.2026.3691270

_N. P. Solagratia and G. S. Thejas, "BERT and Local Vision Language Models in Hate Speech Detection on Small Multimodal Datasets," in IEEE Access, doi: 10.1109/ACCESS.2026.3691270_
## Link to the paper and DOI: [10.1109/ACCESS.2026.3691270](https://ieeexplore.ieee.org/document/11511693)

We investigate multimodal hate speech detection by combining BERT-variant text encoders with EfficientNetV2L image features, and separately evaluating local Vision Language Models (VLMs) via zero-shot inference using LM Studio. Experiments are conducted across three benchmark datasets.

---

## Datasets

| Dataset | Type | Task |
|---|---|---|
| [HarMemeC](https://www.kaggle.com/datasets/mohdaamir21/harmeme-dataset) | Meme images + text | Multiclass (Can be converted into Binary) hate speech classification |
| [Multi3Hate](https://github.com/MinhDucBui/Multi3Hate) | Multilingual meme images + text | Binary hate speech classification |
| [MultiOFF](https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text) | Meme images + text | Binary offensive content classification |

Datasets are not included in this repository. Download them from the links above and place them under a `*_Dataset/` directory at the project root.

### HarMemeC Preprocessing

HarMemeC provides three separate harmfulness indicator columns (`not harmful`, `somewhat harmful`, `very harmful`) rather than a single label. We convert these to a binary label as follows:

- **Label 0 (not harmful):** `not harmful == 1`
- **Label 1 (harmful):** `somewhat harmful == 1` or `very harmful == 1`

The preprocessing script is located at `BERT_Evaluation_Examples/preprocess.ipynb`. It outputs `train_val_bin.csv` and `test_bin.csv`, which are the files consumed by the HarMemeC training notebooks.

---

## Repository Structure

```
.
├── BERT_Evaluation_Examples/          # BERT-variant multimodal classifier notebooks
│   ├── harmemeC_bert.ipynb            # HarMemeC - standard fine-tuning
│   ├── harmemeC_lora_bert.ipynb       # HarMemeC - LoRA fine-tuning
│   ├── multi3hate_bert.ipynb          # Multi3Hate - standard fine-tuning
│   ├── multi3hate_lora_bert.ipynb     # Multi3Hate - LoRA fine-tuning
│   ├── multioff_bert.ipynb            # MultiOFF - standard fine-tuning
│   ├── multioff_lora_bert.ipynb       # MultiOFF - LoRA fine-tuning
│   └── multimodal_architecture_updated.py  # Model architecture definitions
│
├── EfficientNetV2L_Image_Feature_E/   # Image feature extraction
│   ├── extract_image.ipynb
│   └── extract_image_features.py
│
├── LMStudio_Zero_Shot_Example/        # Zero-shot VLM inference via LM Studio
│   ├── HarMemeC/
│   │   ├── Gemma4-E4b.ipynb
│   │   └── Magistral-Small_30runs.ipynb
│   ├── Multi3Hate/
│   │   ├── LFM2-1_6b.ipynb
│   │   └── Qwen3-8b_30runs.ipynb
│   └── MultiOFF/
│       ├── Gemma3_27B_30runs.ipynb
│       └── Gemma4-E4B.ipynb
│
└── Setup/                             # Shared utilities
    ├── all_imports.py                 # Common imports across all notebooks
    ├── compute_metrics.py             # Evaluation metrics (F1, AUC-PR, etc.)
    ├── gpu_warm_up.py                 # GPU warm-up routine
    └── text_cleaner.py                # Text preprocessing for meme captions
```

---

## Architecture

### BERT Multimodal Classifier

Text and image modalities are encoded separately and fused via an MLP head:

```
Text (caption) --> BERT variant --> Mean Pooling --> Linear Projection (512d) --+
                                                                                |--> MLP Fusion --> Classification
Image (meme)   --> EfficientNetV2L (offline) --> Linear Projection (512d) ------+
```

Two variants are implemented in `multimodal_architecture_updated.py`:

- `MultimodalClassifier` — full fine-tuning of the text encoder
- `LoRAMultiModalClassifier` — parameter-efficient fine-tuning using LoRA (via PEFT)

Supported text encoders:

| Model | HuggingFace ID |
|---|---|
| BERT-base | `google-bert/bert-base-uncased` |
| HateBERT | `GroNLP/hateBERT` |
| RoBERTa | `FacebookAI/roberta-base` |
| ModernBERT | `answerdotai/ModernBERT-base` |
| DistilBERT | `distilbert-base-uncased` |
| IBM Granite | `ibm-granite/granite-embedding-30m-english` |
| All-MiniLM-L6 | `sentence-transformers/all-MiniLM-L6-v2` |

### VLM Zero-Shot Baseline

Local VLMs are served via [LM Studio](https://lmstudio.ai/) and queried over the local API. No fine-tuning is performed — models are evaluated directly on meme images in a zero-shot setting.

---

## Training Procedure

Each BERT notebook follows this pipeline:

1. Text cleaning via `TextCleaner`
2. Adaptive `max_token_length` selection based on the 99th percentile of training token lengths
3. Offline image feature extraction using EfficientNetV2L (380x380)
4. Hyperparameter search using [Optuna](https://optuna.org/) with 5-fold stratified cross-validation (50 trials)
5. Final evaluation: 30 repeated runs using best parameters, reporting mean and std across runs

Effective hyperparameter search space:

| Parameter | Range |
|---|---|
| Learning rate | ~1e-5 |
| Batch size | 2, 4 |
| Training epochs | 6-10 |
| Weight decay | ~1e-3 |
| Warmup ratio | 0.1-0.2 |
| LR scheduler | linear, cosine |

---

## Setup

### Requirements

```bash
# Core
pip install torch transformers peft optuna datasets scikit-learn scipy
pip install pandas numpy matplotlib Pillow colorama tqdm

# Image feature extraction only
pip install tensorflow

# VLM zero-shot inference only
pip install lmstudio
```

### LM Studio (for VLM experiments)

Download [LM Studio](https://lmstudio.ai/) and load the desired model locally. The notebooks expect the LM Studio local server running at `http://localhost:1234`.

### List of Vision Language Models Used

| Model | Size | Storage (GB) |
|---|---|---|
| [Gemma3 4B](https://lmstudio.ai/models/google/gemma-3-4b) | 4B + 417M | 3.34 |
| [Gemma3 12B](https://lmstudio.ai/models/google/gemma-3-12b) | 12B + 417M | 8.15 |
| [Gemma3 27B](https://lmstudio.ai/models/google/gemma-3-27b) | 27B + 417M | 16.43 |
| [Gemma4 E2B](https://lmstudio.ai/models/google/gemma-4-e2b) | 5B (E 2B) | 4.41 |
| [Gemma4 E4B](https://lmstudio.ai/models/google/gemma-4-e4b) | 7.9B (E 4B) | 6.33 |
| [Gemma4 26B-A4B](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | 25.2B (A 3.8B) | 15.71 |
| [LFM2 450M](https://huggingface.co/bartowski/LiquidAI_LFM2-VL-450M-GGUF) | 350M + 86M | 0.42 |
| [LFM2 1.6B](https://huggingface.co/lmstudio-community/LFM2-VL-1.6B-GGUF) | 1.2B + 400M | 1.56 |
| [LFM2 3B](https://huggingface.co/LiquidAI/LFM2-VL-3B) | 2.6B + 400M | 2.34 |
| [Ministral3 3B](https://lmstudio.ai/models/mistralai/ministral-3-3b) | 3.4B + 400M | 2.99 |
| [Magistral Small 2509](https://lmstudio.ai/models/mistralai/magistral-small-2509#magistral-small-12) | 24B | 15.21 |
| [Qwen2.5 7B](https://lmstudio.ai/models/qwen/qwen2.5-vl-7b) | ~8B* | 6.04 |
| [Qwen3 4B](https://lmstudio.ai/models/qwen/qwen3-vl-4b) | 4.44B | 3.33 |
| [Qwen3 8B](https://lmstudio.ai/models/qwen/qwen3-vl-8b) | 8.77B | 6.19 |
| [Qwen3 30B](https://lmstudio.ai/models/qwen/qwen3-vl-30b) | 31.1B | 19.64 |

*Qwen2.5 7B parameter size is closer to 8B when the vision encoder is included.

### Image Feature Extraction

Run `EfficientNetV2L_Image_Feature_E/extract_image_features.py` before any BERT notebook to pre-extract and cache image features as `.npz` files.

---

## Results

Results are saved per experiment as CSV files under `Results CSV/` (not included in the repository). Each run logs: Loss, Accuracy, Precision, Recall, F1-Score, and AUC-PR.

---

## Citation

If this repository is useful for your research, please cite our paper:

```bibtex
@ARTICLE{11511693,
  author={Solagratia, Nathanael P. and Thejas, G.S.},
  journal={IEEE Access}, 
  title={BERT and Local Vision Language Models in Hate Speech Detection on Small Multimodal Datasets}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Bit error rate;Hate speech;Training;Accuracy;Machine learning;Labeling;LoRa;Social networking (online);Conferences;BERT;hate speech detection;low-resource learning;multimodal learning;offensive content detection;vision-language models},
  doi={10.1109/ACCESS.2026.3691270}}
```
---
URL: https://ieeexplore.ieee.org/document/11511693

## License

This paper is under Creative Common License 4.0.
This project is for academic research purposes. Please refer to the respective dataset licenses before use.
