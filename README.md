# 🤖 Question Answering System with ALBERT

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tuning **ALBERT-base** for extractive Question Answering on **SQuAD v1.1** with robust preprocessing, post-processing, and early stopping to prevent overfitting.

## 📊 Results

| Metric | Score | Notes |
|--------|-------|-------|
| **Exact Match (EM)** | **56.8%** | Subset demo (2k train / 500 val) |
| **F1 Score** | **70.8%** | ALBERT-base, 5 epochs, early stopping |

> 💡 **Note**: Using the full SQuAD train set (87k examples) and/or stronger models (RoBERTa/BERT-large) typically yields **F1 85-90+**.

### 📈 Training Progress
```
Epoch  Training Loss  Validation Loss  Exact Match  F1
1      4.277300       1.966378         46.400000    59.443828
2      1.713600       1.549081         55.400000    69.436697
3      1.168400       1.478653         56.800000    70.802003
4      0.604700       1.637864         56.600000    70.867851
5      0.469100       1.681083         57.000000    71.109433
```

## 🏗️ Project Structure

```
question-answering-system/
├── src/
│   ├── data.py              # 📦 Load SQuAD JSON → DataFrame → HF Dataset
│   ├── postprocess.py       # 🔄 prepare_features + postprocess predictions
│   ├── callbacks.py         # ⏹️  EarlyStoppingCallback
│   ├── train.py             # 🚀 Training script with Trainer
│   ├── eval.py              # 📊 Evaluation script
│   └── inference.py         # 🔍 Single-question inference helper
├── configs/                 # ⚙️  Configuration files (optional)
├── results/                 # 📈 Logs and metrics outputs
├── scripts/                 # 🛠️  Helper scripts
│   ├── train.sh
│   └── eval.sh
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download SQuAD v1.1 from [Kaggle](https://www.kaggle.com/datasets/stanfordnlp/squad) or [official website](https://rajpurkar.github.io/SQuAD-explorer/):

```bash
# Option 1: Using Kaggle CLI
kaggle datasets download -d stanfordnlp/squad -f train-v1.1.json
unzip train-v1.1.json.zip -d data/

# Option 2: Direct download
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O data/train-v1.1.json
```

### 3. Training

```bash
python -m src.train \
  --train_file data/train-v1.1.json \
  --output_dir ./qa_model \
  --train_size 2000 \
  --valid_size 500 \
  --model_name albert-base-v2
```

**Key Anti-Overfitting Settings:**
- ✅ Learning rate: `2e-5` (reduced from 3e-5)
- ✅ Batch size: `16` (increased from 8)
- ✅ Weight decay: `0.1` (increased from 0.01)
- ✅ LR scheduler: `cosine` decay
- ✅ Early stopping: `patience=2`, `min_delta=0.001`
- ✅ Metric for best model: `eval_loss` (not F1)

### 4. Evaluation

```bash
python -m src.eval \
  --train_file data/train-v1.1.json \
  --model_path ./qa_model \
  --train_size 2000 \
  --valid_size 500
```

### 5. Inference

```python
from src.inference import load_qa_model, predict_answer

# Load model
model, tokenizer, device = load_qa_model("./qa_model")

# Predict
question = "Who wrote the novel 1984?"
context = "The novel 1984 was written by George Orwell in 1949."
result = predict_answer(model, tokenizer, question, context, device)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 🎯 Features

- ✅ **Robust preprocessing**: Handles overflow tokens, answer position mapping
- ✅ **Anti-overfitting**: Early stopping, weight decay, cosine LR scheduling
- ✅ **Proper post-processing**: Top-k predictions, context filtering, answer_start in references
- ✅ **Production-ready**: Clean code structure, error handling, logging

## 📚 Key Improvements Over Baseline

| Aspect | Baseline | This Project |
|--------|----------|--------------|
| **Answer Finding** | Simple `find()` | 3-tier fallback (exact → case-insensitive → normalized) |
| **Overflow Tokens** | Not handled | Proper mapping with `overflow_to_sample_mapping` |
| **Post-processing** | Argmax only | Top-k predictions with filtering |
| **Overfitting** | No early stopping | Early stopping + stronger regularization |
| **Metrics** | Basic | SQuAD-compliant with `answer_start` |

## 🔧 Configuration

Key hyperparameters (in `src/train.py`):

```python
learning_rate = 2e-5
per_device_train_batch_size = 16
weight_decay = 0.1
num_train_epochs = 5
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_grad_norm = 1.0
```

## 📈 Tips to Improve Scores

1. **Use full dataset**: Train on all 87k SQuAD examples instead of 2k subset
2. **Stronger models**: Try `deepset/roberta-base-squad2` or `bert-large-uncased-whole-word-masking-finetuned-squad`
3. **Hyperparameter tuning**:
   - Learning rate: `1e-5` to `2e-5`
   - Batch size: `16` to `32` (if GPU allows)
   - Weight decay: `0.1` to `0.2`
   - Doc stride: `128` to `192`
   - Max answer length: `30` to `45`
4. **Regularization**: Freeze early layers or add dropout if still overfitting
5. **Data augmentation**: Paraphrasing, back-translation

## 🐛 Troubleshooting

### Issue: Validation loss increases while training loss decreases
**Solution**: This is overfitting. The project already includes early stopping, but you can:
- Increase `weight_decay` to `0.2`
- Reduce `learning_rate` to `1e-5`
- Freeze early ALBERT layers

### Issue: KeyError 'answer_start' in evaluation
**Solution**: Ensure `postprocess_predictions_fixed_final` is used (includes `answer_start` in references)

### Issue: Results don't match training metrics
**Solution**: Use same tokenization settings (max_length=512, stride=128) and post-processing pipeline

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/) by Stanford NLP
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [ALBERT Model](https://github.com/google-research/albert)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

⭐ **Star this repo if you find it helpful!**
