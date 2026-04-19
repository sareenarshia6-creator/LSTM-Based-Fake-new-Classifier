# 📰 Fake News Classifier using LSTM

**A Natural Language Processing (NLP) deep learning project that classifies news articles as "Fake" or "Real" using Long Short-Term Memory (LSTM) networks.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Enabled-red)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)

---

## 📝 Project Overview

Fake news detection is a critical application of text classification in the modern digital age. The goal of this project is to build a robust sequential neural network that can accurately predict the authenticity of a news article based strictly on its **Title**. 

Traditional machine learning models often struggle with text because they fail to capture the sequential context and relationships between words in a sentence. To solve this, this project implements a **Long Short-Term Memory (LSTM)** network. LSTMs are a specialized type of Recurrent Neural Network (RNN) capable of learning long-term dependencies, making them highly effective for natural language understanding.

### Key Features & Architecture:
* **Advanced Text Preprocessing:** Cleans raw text data using NLTK for Stopwords removal and Porter Stemmer to reduce vocabulary noise.
* **Dense Word Embeddings:** Converts sparse One-Hot encoded vectors into dense vector representations, allowing the model to understand semantic relationships between words.
* **Deep Learning Architecture:** Utilizes a custom TensorFlow/Keras sequential model featuring Embedding Layers, LSTM layers for sequence memory, and Dropout layers to actively mitigate overfitting during training.

## 💻 Tech Stack & Dependencies

**Core Language:**
* **Python 3.8+**: The primary programming language used for the entire project.

**Data Processing & NLP:**
* **Pandas**: Used for data ingestion, dataframe manipulation, and cleaning (dropping missing or `NaN` values).
* **NumPy**: Used for efficient array operations, matrix transformations, and handling the final input data shapes required by the neural network.
* **NLTK (Natural Language Toolkit)**: Essential for text preprocessing. Specifically used for importing English Stopwords and applying the `PorterStemmer` to reduce words to their root structure.
* **RegEx (`re`)**: Built-in Python library used to filter out punctuation, numbers, and special characters from the raw textual data.

**Deep Learning & Modeling:**
* **TensorFlow & Keras**: The core deep learning ecosystem used to architect the model. Keras provides the high-level API to assemble the sequential network.
  * *Specific Keras modules used:* `Sequential`, `Embedding`, `LSTM`, `Dense`, `Dropout`, `pad_sequences`, and `one_hot`.

**Evaluation & Metrics:**
* **Scikit-Learn (`sklearn`)**: Utilized for splitting the dataset (`train_test_split`) and for generating crucial evaluation metrics, including `confusion_matrix`, `classification_report`, and `accuracy_score`.

## 📊 Dataset

The dataset used for this project originates from the Kaggle Fake News competition. 

| Column | Description |
| :--- | :--- |
| `id` | Unique identification number for the news article |
| `title` | The headline/title of the news article |
| `author` | The author of the news article |
| `text` | The full body text of the article |
| `label` | Target variable: `1` indicates Fake/Unreliable, `0` indicates Real/Reliable |

*Note: To optimize training time and focus on high-impact features, this specific model is trained exclusively on the `title` feature to predict the `label`.*

## ⚙️ Detailed Workflow & Steps

### 1. Data Loading & Cleaning
Before feeding data into a neural network, it must be clean and structured.
* **Load Data:** Imported the `train.csv` file using Pandas.
* **Handle Missing Values:** Dropped `NaN` values. Text data cannot be logically imputed with means or medians, so dropping incomplete records ensures the model doesn't crash during training.
* **Feature Selection:** Extracted the `title` column as the independent variable (`X`) and `label` as the dependent variable (`y`).

### 2. Text Preprocessing
Raw text contains grammar, punctuation, and common words that don't add predictive value.
* **Regex Filtering:** Removed all special characters, numbers, and punctuation using regular expressions, keeping only alphabetical characters.
* **Lowercasing:** Standardized all text to lowercase to prevent the model from treating "News" and "news" as two different words.
* **Stopwords Removal:** Filtered out common English words (e.g., "the", "is", "in") that provide no contextual meaning for classification.
* **Stemming:** Applied NLTK's `PorterStemmer` to chop off word suffixes, reducing words to their root forms (e.g., "running" becomes "run").

### 3. Sequence Preparation
Neural networks require numerical input of a fixed size, not raw text strings.
* **One-Hot Encoding:** Converted the cleaned text into numerical indexes based on a predefined dictionary/vocabulary size of `5,000` most frequent words.
* **Padding Sequences:** Sentences have varying lengths, but the LSTM requires a uniform input shape. We applied **Pre-Padding** using Keras's `pad_sequences` to ensure every title is exactly `20` words long, adding zeros to the beginning of shorter titles.

### 4. Model Building (LSTM)
The core Deep Learning architecture was built using the Keras `Sequential` API:
* **Embedding Layer:** Transforms the padded integer sequences into dense vectors of fixed size (`40` features per word). This is where the model learns word semantics.
* **Dropout Layer (0.3):** Randomly deactivates 30% of neurons during training to prevent the model from memorizing the training data (overfitting).
* **LSTM Layer:** A layer with `100` memory units that processes the word embeddings sequentially, carrying context from the beginning of the title to the end.
* **Dense Output Layer:** A single neuron with a `sigmoid` activation function that squashes the final output into a probability score between `0` and `1`.
* **Compilation:** Used the `adam` optimizer and `binary_crossentropy` loss function (standard for binary classification tasks).

### 5. Training & Evaluation
* **Data Split:** Divided the processed arrays into Training (80%) and Testing (20%) sets.
* **Training:** Fit the model over `10 epochs` passing the data in batches of `64`.
* **Prediction:** Generated predictions on the unseen test set, establishing a threshold of `> 0.5` to classify a result as `1` (Fake).
* **Results:** The model successfully learned to distinguish real from fake headlines, achieving an overall accuracy score of **~90-91%**.

## 🚀 How to Run

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/your-username/fake-news-lstm.git](https://github.com/your-username/fake-news-lstm.git)
