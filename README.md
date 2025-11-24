# NLP Disaster Tweets – RNN, LSTM & GRU Classification

This project is a course assignment based on the Kaggle **“Natural Language Processing with Disaster Tweets”** competition.  
The goal is to build machine learning models that predict whether a tweet is about a real disaster (`1`) or not (`0`).

I start with a simple baseline model using TF-IDF + Logistic Regression, and then build sequential neural network models using GRU and LSTM to see if deep learning can improve performance.

---

## 1. Dataset

- Source: Kaggle – Disaster Tweets dataset (~10,000 tweets).
- Main columns used:
  - `id`: unique tweet ID  
  - `text`: tweet text  
  - `target`: label (`1` = real disaster, `0` = not a disaster)  
- Extra columns:
  - `keyword`, `location` – contain many missing values; filled with `"unknown"` when used.

---

## 2. Problem & Approach

**Task:** Binary classification of tweets into *disaster* vs *non-disaster*.

### Preprocessing

- Checked label distribution with a bar chart of the `target` column.
- Checked for duplicate rows and missing values.
- Filled missing `keyword` and `location` values with `"unknown"`.
- Turned raw tweet text into numeric features in two different ways:
  1. **TF-IDF** vectors for the baseline model.
  2. **Tokenized + padded sequences** for GRU/LSTM models using Keras:
     - `Tokenizer` to map words → integer IDs  
     - `pad_sequences` to make all tweets the same length (max_len = 40).

---

## 3. Models

### 3.1 Baseline: TF-IDF + Logistic Regression

- Used `TfidfVectorizer` with:
  - `max_features = 10000`
  - `ngram_range = (1, 2)` (unigrams + bigrams)
  - English stopword removal
- Trained a Logistic Regression classifier on TF-IDF features.
- This model is used as the **baseline** to compare with deep models.

**Validation F1:** `0.7648`

---

### 3.2 GRU Model

- Text pipeline: tokenized + padded sequences → embedding layer.
- Model architecture:
  - `Embedding(max_words, embedding_dim)`
  - `GRU(gru_units, return_sequences=True)`
  - `GlobalMaxPooling1D`
  - `Dropout`
  - `Dense(32, relu)`
  - `Dense(1, sigmoid)`
- Hyperparameters tuned:
  - `embedding_dim`, `gru_units`, `dropout_rate`, `learning_rate`, `batch_size`
- Used **EarlyStopping** on validation loss to avoid overfitting.

**Best GRU configuration:**  
`embedding_dim=64, gru_units=64, dropout=0.2, lr=0.001, batch_size=64`  
**Best Validation F1:** `0.7688`

Training curves show that training accuracy keeps increasing, while validation accuracy peaks around ~0.80 and validation loss starts to rise after a few epochs, indicating mild overfitting. Early stopping helps stop training before it gets worse.

---

### 3.3 LSTM Model

- Similar text pipeline as GRU (tokenized + padded + embeddings).
- Model architecture:
  - `Embedding(max_words, embedding_dim)`
  - `Bidirectional(LSTM(lstm_units))`
  - `Dropout`
  - `Dense(32, relu)`
  - `Dropout`
  - `Dense(1, sigmoid)`
- Hyperparameters tuned:
  - `embedding_dim`, `lstm_units`, `dropout_rate`, `learning_rate`

**Best LSTM configuration:**  
`embedding_dim=128, lstm_units=128, dropout=0.5, lr=0.0005`  
**Best Validation F1:** `0.7700`

The LSTM training curves look very similar to the GRU: strong training performance with validation metrics flattening and then slightly degrading after a few epochs.

---

## 4. Results Summary

| Model                                | Validation F1 |
|-------------------------------------|--------------:|
| TF-IDF + Logistic Regression (base) | **0.7648**    |
| GRU (best)                          | **0.7688**    |
| LSTM (best)                         | **0.7700**    |

The TF-IDF + Logistic Regression model provides a strong baseline with an F1 score of 0.7648.  
Both deep learning models slightly improve over this baseline, with the **LSTM model performing best overall**, followed closely by the GRU.  
This suggests that sequence-aware models with learned word embeddings can capture a bit more signal from the tweet text than the simpler TF-IDF representation, although the gains are modest.

Overall, the LSTM model achieved the highest validation F1 score (≈0.77), narrowly outperforming both the GRU model and the TF-IDF + Logistic Regression baseline, showing that a well-tuned LSTM provides a small but meaningful improvement for this disaster tweet classification task.

---
