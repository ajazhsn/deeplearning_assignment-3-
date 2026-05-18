# DA6401 Assignment 3 — Transformer for Machine Translation

**German → English Neural Machine Translation** using the architecture from ["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Vaswani et al., 2017).

## Links

- **GitHub Repository:** https://github.com/ajazhsn/deeplearning_assignment-3-
- **W&B Report:** https://api.wandb.ai/links/da25m006-iitmaana/9qcpkrx3

## Results

| Metric | Score |
|---|---|
| Test BLEU | **37.63** |
| MHA Tests | 10/10 |
| Positional Encoding Tests | 10/10 |
| Noam Scheduler Tests | 10/10 |
| Translation Performance | 20/20 |

## Model Architecture

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| N (layers) | 3 |
| num_heads | 8 |
| d_ff | 512 |
| dropout | 0.1 |
| warmup_steps | 2000 |
| Positional Encoding | Sinusoidal |
| Loss | Label Smoothing (ε=0.1) |

## Project Structure

```
├── model.py          # Transformer architecture (MHA, PE, Encoder, Decoder)
├── dataset.py        # Multi30k dataset loading and preprocessing
├── train.py          # Training pipeline, greedy decode, BLEU evaluation
├── lr_scheduler.py   # Noam learning rate scheduler
├── ablations.py      # W&B experiments (Sections 2.1–2.5)
└── setup.sh          # Dependency installation
```

## Setup

```bash
bash setup.sh
```

Or manually:

```bash
pip install torch datasets spacy wandb nltk gdown matplotlib
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Training

```python
from train import run_training_experiment
run_training_experiment()
```

## Inference

```python
from model import Transformer
model = Transformer()  # auto-downloads checkpoint
print(model.infer("Eine Gruppe von Männern arbeitet an einem Lastwagen."))
# → "a group of men are working on a truck ."
```

## W&B Experiments

| Section | Experiment | Key Finding |
|---|---|---|
| 2.1 | Noam vs Fixed LR | Noam achieves 37.52 BLEU vs lower with fixed LR |
| 2.2 | Scaling Factor 1/√dk | Without scaling: 29.98 BLEU vs 35.68 with scaling |
| 2.3 | Attention Head Heatmaps | Head redundancy observed in heads 2, 6 and 3, 8 |
| 2.4 | Sinusoidal vs Learned PE | 37.68 vs 37.47 BLEU — sinusoidal extrapolates better |
| 2.5 | Label Smoothing | ε=0.1 prevents overconfidence; higher training perplexity but better generalisation |

## Dataset

[Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) — 29,000 train / 1,014 val / 1,000 test sentence pairs.
