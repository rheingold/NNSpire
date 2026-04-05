# Hugging Face Hub API

**Document Version:** 1.0  
**Generated:** December 4, 2025  
**Standard:** Hugging Face Model Hub Protocol  
**Status:** Community Standard

---

## Overview

**Hugging Face Hub** is a central repository for machine learning models, datasets, and spaces. The Hub API provides programmatic access for uploading, downloading, and managing models with integrated model cards, versioning, and community features.

### Key Features

- **Model Repository**: Central hub for sharing models
- **Model Cards**: Integrated documentation
- **Multiple Backends**: Support for Transformers, Diffusers, etc.
- **Versioning**: Git-like version control
- **Access Control**: Public/private models
- **Community**: Social features and discussions
- **API Access**: REST and Python SDK

### Primary Use Cases

1. **Model Sharing**: Distribute trained models
2. **Model Discovery**: Find community models
3. **Model Cards**: Standardized documentation
4. **Fine-tuning**: Start from pre-trained models
5. **Inference**: Direct model access via API

---

## Authoritative References

- **Hugging Face Hub Docs**: https://huggingface.co/docs/hub
- **Hub API Reference**: https://huggingface.co/docs/hub/api
- **Model Cards**: https://huggingface.co/docs/hub/models-cards
- **GitHub**: https://github.com/huggingface/hub

---

## Format Structure

### Model Repository Structure

```
model-repo/
├── README.md                        # Model card
├── config.json                      # Model configuration
├── model.safetensors               # Model weights (safetensors format)
├── tokenizer.json                  # Tokenizer config
├── tokenizer.model                 # Tokenizer vocab
├── special_tokens_map.json         # Special tokens
├── preprocessor_config.json        # Processor config
└── .gitattributes                  # Git LFS pointers
```

### Model Configuration (JSON)

**Abstract pseudo-structure** — field roles and types:
```
config.json
{
  "_name_or_path":            <string>   // Original model name or local path
  "architectures":            [<string>] // List of model class names
  "model_type":               <string>   // Architecture family (e.g. "bert", "gpt2")
  "hidden_size":              <int>      // Dimensionality of hidden layers
  "num_hidden_layers":        <int>      // Number of transformer blocks
  "num_attention_heads":      <int>      // Number of attention heads per layer
  "intermediate_size":        <int>      // Dimensionality of feed-forward sublayer
  "hidden_act":               <string>   // Activation function (e.g. "gelu", "relu")
  "hidden_dropout_prob":      <float>    // Dropout probability for hidden layers
  "attention_probs_dropout_prob": <float> // Dropout probability for attention
  "max_position_embeddings":  <int>      // Maximum sequence length supported
  "vocab_size":               <int>      // Size of the token vocabulary
  "initializer_range":        <float>    // Std-dev for weight initialization
  "layer_norm_eps":           <float>    // Epsilon for layer normalization
  "pad_token_id":             <int>      // Token ID used for padding
  "position_embedding_type":  <string>   // "absolute" | "relative_key" | …
  "use_cache":                <bool>     // Whether to cache key/value states
  "transformers_version":     <string>   // Library version that saved the config
  // … additional architecture-specific fields
}
```

**Real example** (config.json):
```json
{
  "_name_or_path": "bert-base-uncased",
  "architectures": ["BertForSequenceClassification"],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.30.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

### Model Card Format (Markdown)

**Abstract pseudo-structure** — logical skeleton of a model card:
```
README.md
---                                    # YAML front-matter start
language:    [<iso-639-1>, …]          # Supported languages
license:     <spdx-id>                 # SPDX license identifier
library_name: <string>                 # Framework (transformers, diffusers, …)
tags:        [<string>, …]             # Searchable tags
widget:                                # Interactive demo inputs
  - text: <string>                     #   sample input text
    example_title: <string>            #   label shown in widget
model-index:                           # Benchmark results block
  - name: <string>                     #   results group name
    results:
      - task:
          name: <string>               #   human-readable task name
          type: <task-type>            #   canonical task identifier
        dataset:
          name: <string>               #   dataset display name
          type: <dataset-id>           #   canonical dataset identifier
        metrics:
          - name: <string>             #   metric display name
            type: <metric-type>        #   canonical metric identifier
            value: <number>            #   achieved score
---                                    # YAML front-matter end

# <Model Title>
<Short description of the model>

## Model Details
  - Developed by / Model type / License / …

## Intended Use
  - Primary use cases and audiences

## Out-of-Scope
  - Known inappropriate or unsupported uses

## Training Data
  - Dataset description and size

## Results
  - Evaluation metrics table

## How to Use
  - Code snippet for inference

## Limitations
  - Known weaknesses and caveats

## Environmental Impact & Bias
  - Carbon footprint, fairness notes
```

**Real example** (README.md):
```markdown
---
language:
  - en
license: mit
library_name: transformers
tags:
  - text-classification
  - bert
widget:
  - text: "This movie is absolutely fantastic!"
    example_title: "Positive example"
  - text: "Terrible movie, waste of time."
    example_title: "Negative example"
model-index:
  - name: Results
    results:
      - task:
          name: Text Classification
          type: text-classification
        dataset:
          name: IMDb
          type: imdb
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.947
---

# bert-imdb-classifier

This model is a BERT model fine-tuned on the IMDb dataset for sentiment classification.

## Model Details

### Model Description

- **Developed by**: Hugging Face
- **Model type**: Transformer
- **Library**: Transformers
- **License**: MIT

### Intended Use

Primary use cases for this model:
- Sentiment analysis on movie reviews
- Text classification tasks

### Out-of-Scope

- Medical/legal document analysis
- Languages other than English

## Training Data

Trained on the IMDb movie reviews dataset with 50,000 labeled examples.

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 94.7% |
| F1 Score | 0.944 |
| AUC-ROC | 0.978 |

## How to Use

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="username/bert-imdb-classifier")
result = classifier("This movie was amazing!")
print(result)
```

## Limitations

- Trained only on English text
- Best performance on movie reviews (domain-specific)
- May not generalize well to other domains

## Environmental Impact

Model inference generates approximately 0.012kg CO2.

## Bias and Fairness

Tested on demographic groups with similar performance across genders and ages.
```

---

## Procedural Use-Cases

### Use-Case 1: Upload Model to Hub

**Abstract pseudocode workflow**:
```
PROCEDURE UploadModelToHub(token, repo_id, model_path):
    1. AUTHENTICATE with Hub using <token>
    2. CREATE remote repository <repo_id>
         - set visibility (public / private)
         - set repo_type = "model"
    3. LOAD model weights from <model_path>
    4. PUSH model weights  → remote <repo_id>
    5. LOAD tokenizer from <model_path>
    6. PUSH tokenizer      → remote <repo_id>
    7. COMPOSE model card (README.md) with YAML front-matter
    8. UPLOAD model card   → remote <repo_id>/README.md
    9. RETURN hub URL for the published model
END PROCEDURE
```

**Workflow** (Python SDK):
```python
from huggingface_hub import HfApi, HfFolder
from transformers import AutoModel, AutoTokenizer
import json

# Authenticate
HfFolder.save_token("your_hf_token")
api = HfApi()

# Create repository
repo_id = "username/my-model"
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,
    exist_ok=True
)

# Upload model files
model = AutoModel.from_pretrained("bert-base-uncased")
model.push_to_hub(repo_id)

# Upload tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.push_to_hub(repo_id)

# Upload model card
model_card = """---
language: en
license: mit
tags:
  - bert
---

# My BERT Model

This is my fine-tuned BERT model for [your task].
"""

api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model"
)

print(f"Model uploaded to https://huggingface.co/{repo_id}")
```

### Use-Case 2: Download and Use Model

**Abstract pseudocode workflow**:
```
PROCEDURE DownloadAndInfer(model_id, input_text):
    1. RESOLVE model_id → Hub URL  (e.g. "user/model-name")
    2. DOWNLOAD tokenizer artefacts from Hub (cached locally)
    3. DOWNLOAD model weights from Hub     (cached locally)
    4. TOKENIZE input_text → tensor of token IDs
    5. RUN forward pass: outputs = model(tokens)
    6. RETURN outputs (embeddings, logits, etc.)
END PROCEDURE
```

**Workflow** (Python):
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Download model and tokenizer
model_id = "username/my-model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# Use for inference
text = "This is a sample text."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
```

### Use-Case 3: Model Card Creation

**Step-by-Step**:
```python
from huggingface_hub import ModelCard, ModelCardData
from datetime import datetime

# Create model card metadata
card_data = ModelCardData(
    language="en",
    license="mit",
    library_name="transformers",
    tags=["text-classification", "bert"],
    model_name="BERT Classifier",
    model_id="username/bert-classifier"
)

# Create model card
card = ModelCard.from_template(
    card_data,
    model_id="username/bert-classifier",
    model_description="Fine-tuned BERT for sentiment analysis",
    intended_use="Sentiment analysis on product reviews",
    limitations="English text only, trained on product reviews",
    how_to_use="""
    ```python
    from transformers import pipeline
    classifier = pipeline("text-classification", model="username/bert-classifier")
    result = classifier("Great product!")
    print(result)
    ```
    """
)

# Save and upload
card.save("README.md")
```

---

## Examples

### Example 1: Hub API REST Requests

**Abstract endpoint pattern summary**:
```
Base URL:  https://huggingface.co

┌─────────────────────────────────────────────────────────────────────┐
│ Verb   Endpoint                          Purpose                  │
├─────────────────────────────────────────────────────────────────────┤
│ GET    /api/models                       List / search models     │
│          ?search=<query>                   full-text search       │
│          &sort=likes|downloads|modified    sort field             │
│          &direction=-1|1                   sort direction         │
│          &limit=<n>                        max results            │
│ GET    /api/models/<owner>/<model>        Single model metadata   │
│ GET    /<owner>/<model>/resolve/<rev>/    Download a repo file    │
│          <filepath>                        at a given revision    │
│ POST   /api/repos/create                  Create a new repo      │
│ DELETE /api/repos/delete                  Delete a repo          │
│ PUT    /api/<owner>/<model>/upload/<rev>  Upload file(s)         │
└─────────────────────────────────────────────────────────────────────┘

Authentication:  Bearer token in header  →  Authorization: Bearer <HF_TOKEN>
Response format: JSON (lists or objects)
```

**List Models**:
```bash
curl "https://huggingface.co/api/models?search=bert&sort=likes&direction=-1&limit=10"
```

**Real Response** (JSON):
```json
[
  {
    "id": "bert-base-uncased",
    "author": "google",
    "modelId": "google/bert-base-uncased",
    "private": false,
    "pipeline_tag": "fill-mask",
    "tags": ["transformers", "tensorflow", "pytorch"],
    "lastModified": "2023-12-04T10:30:00.000Z",
    "siblings": [
      {"rfilename": "config.json"},
      {"rfilename": "pytorch_model.bin"},
      {"rfilename": "tokenizer.json"}
    ],
    "downloads": 50000000,
    "likes": 15000
  }
]
```

**Download File**:
```bash
curl "https://huggingface.co/google/bert-base-uncased/resolve/main/config.json" \
  -o config.json
```

### Example 2: Python API Usage

```python
from huggingface_hub import hf_hub_download, model_info, list_models

# Get model info
info = model_info("google/bert-base-uncased")
print(f"Model: {info.modelId}")
print(f"Downloads: {info.downloads}")
print(f"Likes: {info.likes}")
print(f"Tags: {info.tags}")

# Download specific file
model_file = hf_hub_download(
    repo_id="google/bert-base-uncased",
    filename="config.json",
    cache_dir="/tmp/models"
)

# List models by task
models = list_models(
    task="text-classification",
    sort="downloads",
    direction=-1
)

for model in models[:5]:
    print(f"{model.modelId} - {model.downloads} downloads")
```

### Example 3: Complete Upload Workflow

```python
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModel, AutoTokenizer
import os

def upload_model_to_hub(
    local_model_path: str,
    repo_name: str,
    private: bool = False
):
    """Upload local model to Hugging Face Hub"""
    
    api = HfApi()
    
    # Create repository
    repo_url = create_repo(
        repo_id=repo_name,
        repo_type="model",
        private=private,
        exist_ok=True
    )
    
    # Prepare model card
    model_card = """---
    language: en
    license: mit
    ---
    
    # My Model
    
    This is my fine-tuned model.
    """
    
    # Write model card
    with open(os.path.join(local_model_path, "README.md"), "w") as f:
        f.write(model_card)
    
    # Upload entire folder
    api.upload_folder(
        folder_path=local_model_path,
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"Model uploaded to {repo_url}")
    return repo_url

# Usage
upload_model_to_hub(
    "/path/to/my/model",
    "username/my-awesome-model",
    private=False
)
```

---

## Safetensors Format

Modern format for storing model weights (replaces pickle):

**Advantages**:
- Secure (no code execution)
- Fast loading
- Smaller file size
- Language-agnostic

**Usage**:
```python
import torch
from safetensors.torch import load_file, save_file

# Save
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")

# Load
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
```

---

## Tools & Ecosystem

| Tool | Description | Homepage |
|------|-------------|----------|
| **huggingface_hub** | Python SDK | https://github.com/huggingface/huggingface_hub |
| **Transformers** | Model library | https://huggingface.co/docs/transformers |
| **Diffusers** | Image generation | https://huggingface.co/docs/diffusers |
| **Datasets** | Dataset library | https://huggingface.co/docs/datasets |
| **Spaces** | Demo hosting | https://huggingface.co/spaces |
| **Gradio** | Interface builder | https://www.gradio.app/ |

---

**Navigation**: [Back to Index](../INDEX.md) | [Previous: MLflow](./06-MLflow.md) | [Next: gRPC for ML Serving](./08-gRPC-ML-Serving.md)
