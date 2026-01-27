# CSE156 PA1: Neural Networks for Text Classification

Assignment for CSE156 - Neural Networks for Text Classification using PyTorch.

## Project Overview

This project implements neural network models for binary sentiment classification on movie reviews:
- **Bag-of-Words (BOW) Models**: 2-layer and 3-layer fully connected networks
- **Deep Averaging Network (DAN)**: Word embedding-based model (to be implemented)

## Setup

### Prerequisites
- Python 3.11+
- PyTorch
- NumPy, scikit-learn, matplotlib

### Installation

1. **Using uv (recommended)**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Using conda**:
   ```bash
   conda env create -f environment.yml
   conda activate cse156_pa1
   ```

3. **Using pip**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Run BOW Models
```bash
python main.py --model BOW
```

This will:
- Train 2-layer and 3-layer neural networks
- Generate training and dev accuracy plots
- Save plots as `train_accuracy.png` and `dev_accuracy.png`

### Run DAN Models
```bash
python main.py --model DAN
```

## Project Structure

```
CSE156_PA1_WI26/
├── data/
│   ├── train.txt              # Training data
│   ├── dev.txt                # Development data
│   ├── glove.6B.50d-relativized.txt    # 50-dim GloVe embeddings
│   └── glove.6B.300d-relativized.txt   # 300-dim GloVe embeddings
├── main.py                    # Main training script
├── BOWmodels.py               # Bag-of-Words model implementations
├── DANmodels.py               # Deep Averaging Network models (TODO)
├── sentiment_data.py          # Data loading utilities
├── utils.py                   # Utility functions
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment file
└── PROJECT_GUIDE.md           # Detailed implementation guide
```

## Data Format

The data files contain newline-separated sentiment examples:
- Format: `[label]\t[sentence]`
- Label: `0` (negative) or `1` (positive)
- Sentences are tokenized but not lowercased

## Implementation Status

- ✅ BOW Models (2-layer and 3-layer)
- ⏳ DAN Model (in progress)
- ⏳ Skip-gram conceptual exercises

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

## Notes

- This assignment can be completed using CPU only (no GPU required)
- The GloVe embeddings are already relativized to the dataset
- See `PROJECT_GUIDE.md` for detailed implementation instructions

## License

This is a course assignment for CSE156.
