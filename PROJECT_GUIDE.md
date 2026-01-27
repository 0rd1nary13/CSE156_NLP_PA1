# CSE156 PA1 Project Guide

## Overview
This assignment focuses on training neural networks for text classification (sentiment analysis) on movie reviews. You'll work with:
1. **Bag-of-Words (BOW) Models** - Already implemented
2. **Deep Averaging Network (DAN) Models** - Need to implement
3. **Skip-gram Word Embeddings** - Conceptual exercises

## Project Structure

```
CSE156_PA1_WI26/
├── data/
│   ├── train.txt              # Training data (label \t sentence)
│   ├── dev.txt                # Development/validation data
│   ├── glove.6B.50d-relativized.txt    # 50-dim GloVe embeddings
│   └── glove.6B.300d-relativized.txt   # 300-dim GloVe embeddings
├── main.py                    # Main training script
├── BOWmodels.py               # BOW model implementations (DONE)
├── DANmodels.py               # DAN model implementations (TODO)
├── sentiment_data.py          # Data loading utilities
└── utils.py                   # Utility functions (Indexer)
```

## Part 1: Bag-of-Words Models (Already Implemented)

The BOW models are already working. You can run them with:

```bash
source .venv/bin/activate
python main.py --model BOW
```

This will:
- Train 2-layer and 3-layer neural networks
- Generate accuracy plots
- Save `train_accuracy.png` and `dev_accuracy.png`

### What's Already Done:
- ✅ `SentimentDatasetBOW` class - loads and vectorizes data using CountVectorizer
- ✅ `NN2BOW` - 2-layer fully connected network
- ✅ `NN3BOW` - 3-layer fully connected network
- ✅ Training and evaluation loops
- ✅ Plotting functionality

## Part 2: Deep Averaging Network (DAN) - TO IMPLEMENT

### What You Need to Do:

1. **Create `SentimentDatasetDAN` class** in `DANmodels.py`:
   - Load sentences and labels
   - Use word embeddings (GloVe) instead of CountVectorizer
   - Convert words to indices using the embedding's word_indexer
   - Pad/truncate sequences to a fixed length
   - Return (sentence_indices, label) pairs

2. **Implement the DAN architecture**:
   - Embedding layer (use `WordEmbeddings.get_initialized_embedding_layer()`)
   - Average the word embeddings for each sentence
   - Pass through fully connected layers
   - Output layer for binary classification

3. **Update `main.py`**:
   - Add DAN model training in the `elif args.model == "DAN"` section
   - Load GloVe embeddings using `read_word_embeddings()`
   - Create `SentimentDatasetDAN` instances
   - Train and evaluate the model

### Key Concepts:

**Deep Averaging Network (DAN)**:
1. Convert words to embeddings
2. Average all word embeddings in a sentence → single vector
3. Pass through neural network layers
4. Output sentiment prediction

**Example Architecture:**
```
Input: "this movie is great"
  ↓
Word Embeddings: [emb(this), emb(movie), emb(is), emb(great)]
  ↓
Average: (emb(this) + emb(movie) + emb(is) + emb(great)) / 4
  ↓
Fully Connected Layer(s)
  ↓
Output: [log_prob_negative, log_prob_positive]
```

### Implementation Steps:

1. **Read the GloVe embeddings**:
   ```python
   from sentiment_data import read_word_embeddings
   embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
   ```

2. **Create dataset class**:
   ```python
   class SentimentDatasetDAN(Dataset):
       def __init__(self, infile, word_embeddings, max_length=50):
           # Read examples
           # Convert words to indices
           # Pad/truncate to max_length
           # Store as tensors
   ```

3. **Implement DAN model**:
   ```python
   class DAN(nn.Module):
       def __init__(self, embedding_layer, hidden_size, num_layers=2):
           # Initialize embedding layer
           # Create fully connected layers
       
       def forward(self, x):
           # x shape: (batch_size, seq_length)
           # Get embeddings: (batch_size, seq_length, emb_dim)
           # Average: (batch_size, emb_dim)
           # Pass through FC layers
           # Return log probabilities
   ```

4. **Handle padding**:
   - Use PAD token (index 0) for padding
   - When averaging, mask out padding tokens (don't include them in average)

## Part 3: Conceptual Exercises

Based on typical assignments, you'll likely need to:

1. **Understand Skip-gram**:
   - How it learns word embeddings
   - The relationship between context and target words
   - Why it's useful for NLP tasks

2. **Analyze Embeddings**:
   - Use `sentiment_data.py` to load embeddings
   - Compute cosine similarity between words
   - Find similar words to given examples

3. **Compare Models**:
   - BOW vs DAN performance
   - Effect of embedding dimensions (50d vs 300d)
   - Effect of network depth

## Implementation Checklist

### For DAN Model:

- [ ] Create `SentimentDatasetDAN` class
  - [ ] Load data using `read_sentiment_examples()`
  - [ ] Convert words to indices using `word_embeddings.word_indexer`
  - [ ] Handle UNK tokens (words not in vocabulary)
  - [ ] Pad sequences to fixed length
  - [ ] Convert to PyTorch tensors

- [ ] Implement `DAN` model class
  - [ ] Initialize embedding layer (frozen or trainable)
  - [ ] Create averaging mechanism (with padding mask)
  - [ ] Add fully connected layers
  - [ ] Add activation functions (ReLU)
  - [ ] Output log probabilities

- [ ] Update `main.py`
  - [ ] Load word embeddings
  - [ ] Create DAN dataset
  - [ ] Initialize DAN model
  - [ ] Train and evaluate
  - [ ] Generate plots

- [ ] Test and debug
  - [ ] Check data loading
  - [ ] Verify model forward pass
  - [ ] Monitor training loss
  - [ ] Compare with BOW results

## Tips

1. **Start Simple**: Begin with a basic DAN (1 hidden layer, frozen embeddings)
2. **Debug Incrementally**: Test data loading, then model forward pass, then training
3. **Use the BOW code as reference**: The structure is similar, just different input format
4. **Padding**: Make sure to mask padding tokens when averaging (don't divide by sequence length, divide by actual word count)
5. **Embedding Dimensions**: Start with 50d embeddings (smaller, faster), then try 300d
6. **Hyperparameters**: Learning rate ~0.0001, batch size ~16-32, hidden size ~100-200

## Running Your Code

```bash
# Activate environment
source .venv/bin/activate

# Run BOW models (already working)
python main.py --model BOW

# Run DAN models (after implementation)
python main.py --model DAN
```

## Common Issues & Solutions

1. **Out of Memory**: Reduce batch size or use smaller embeddings
2. **Poor Performance**: Try different learning rates, more epochs, or larger hidden layers
3. **Padding Issues**: Make sure to mask padding when averaging embeddings
4. **UNK Tokens**: Handle words not in vocabulary using the UNK embedding

## Next Steps

1. Read through `BOWmodels.py` to understand the pattern
2. Read `sentiment_data.py` to understand data structures
3. Implement `SentimentDatasetDAN` first
4. Implement `DAN` model class
5. Update `main.py` to use DAN
6. Test and iterate

Good luck!
