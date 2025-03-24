# Name Entity Recognition(NER)
This repository demonstrates the implementation of a Named Entity Recognition (NER) model using CoNLL-2003 dataset.

## Prerequisites

Before using this repository, ensure the following:
- Python 3.9.19 is installed.
- TensorFlow 2.5.0 is installed.
- Required dependencies are installed using
  ```pip install -r requirements.txt```.

---
## Steps to Use the Repository

### Step 1: Clone the Repository
Clone the repository to your local machine:
```
git clone https://github.com/TheAnujBaliyan/NERProj.git
```

### Step 2: Install Dependencies
Install all required Python packages:

### Step 3: Train the Model
Train the model using the command:
```
python train.py
```
### step 4: Run the App 
```
python app.py
```
Now, use the app on the local port.

# Report

## Steps for Data Preprocessing and Feature Engineering

### 1. Dataset Preparation
- The CoNLL-2003 dataset is used, containing sentences tagged with named entities.
- Sentences are split into arrays of words and their corresponding entity labels, e.g., `[ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'] ]`.

### 2. Vocabulary Building
- Unique words and labels are identified across the dataset.
- Mappings are created to associate each word and label with a unique index.
- Special tokens like `PADDING_TOKEN` (for padding) and `UNKNOWN_TOKEN` (for out-of-vocabulary words) are included.

### 3. Feature Encoding
- Words and labels are converted to indices based on vocabulary mappings.
- Sentences are padded to a maximum sequence length of 128 using `pad_sequences`.

### 4. Pre-trained Word Embeddings
- GloVe embeddings (`glove.6B.100d.txt`) are used to create an embedding matrix for all words in the vocabulary.
- Words not found in GloVe embeddings are assigned zero vectors.

### 5. Data Pipeline
- TensorFlow’s `tf.data.Dataset.from_tensor_slices` is utilized for batching and shuffling data.
- Batch sizes: 
  - Training: 32
  - Validation: 64
  - Testing: 64

---

## Model Selection and Optimization Approach

### 1. Model Architecture
The model (`TFNer`) consists of:
- **Embedding Layer**: Initialized with pre-trained GloVe embeddings (non-trainable).
- **Bidirectional LSTM Layer**: Contains 128 units for sequential data processing.
- **Dense Layer**: Maps LSTM outputs to label predictions.

### 2. Loss Function and Optimizer
- **Loss Function**: Sparse categorical cross-entropy for multi-class classification tasks.
- **Optimizer**: Adam optimizer with a learning rate of 0.01.

### 3. Training Strategy
- A custom training loop is implemented using `tf.GradientTape` for backpropagation.
- Training loss is tracked using TensorFlow metrics, while validation loss is monitored after each epoch.

### 4. Evaluation Metrics
- Precision, recall, and F1-score metrics are calculated using the `seqeval` library.
- Classification reports are generated to assess prediction accuracy on test data.

---

## Deployment Strategy and API Usage Guide

### Deployment Steps:
1. Save model weights after training:
2. Load saved weights into the `TFNer` model instance:

### API Usage:
To use the trained model for predictions:
1. Pass input sentences through the model to obtain logits:
2. Apply softmax activation to logits:
3. Extract label predictions using `tf.argmax`:




