#!/bin/bash
# setup.sh — Run this ONCE to install all dependencies

echo "=== Installing Python packages ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# If no GPU / Colab CPU: pip install torch
pip install datasets spacy tqdm wandb nltk matplotlib scikit-learn

echo "=== Downloading spaCy language models ==="
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

echo "=== Setup complete! ==="
