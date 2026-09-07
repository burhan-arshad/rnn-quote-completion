# RNN Quote Completion

A Natural Language Processing project that uses a Recurrent Neural Network (RNN) to learn patterns from a collection of quotes and predict the next word in a given sequence.

The project demonstrates how text can be transformed into numerical sequences, processed through an embedding layer and RNN, and used for next-word prediction and quote completion.

## Live Demo

🚀 **Try the application live:**
https://rnn-quote-completion-burhan.streamlit.app/

## Features

* Text preprocessing and cleaning
* Tokenization
* Sequence generation
* Padding of input sequences
* Word embeddings
* Recurrent Neural Network (RNN)
* Next-word prediction
* Quote completion
* Temperature-based text generation
* Interactive Streamlit interface

## Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Streamlit

## How It Works

The model follows this text-generation pipeline:

```text
Quote Dataset
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Sequence Generation
     ↓
Padding
     ↓
Embedding Layer
     ↓
RNN Layer
     ↓
Dense + Softmax
     ↓
Next Word Prediction
     ↓
Quote Completion
```

The model learns relationships between words from the training data. Given a sequence of words, it predicts the most probable next word and uses the prediction to generate a continuation.

## Model Architecture

The neural network consists of:

```text
Input Sequence
      ↓
Embedding Layer
      ↓
RNN Layer
      ↓
Dense Layer
      ↓
Softmax
      ↓
Next Word
```

### Embedding Layer

The embedding layer converts tokenized words into dense numerical vectors, allowing the neural network to learn meaningful relationships between words.

### RNN Layer

The Recurrent Neural Network processes the input sequence while maintaining information from previous words, allowing the model to learn sequential patterns in text.

### Dense + Softmax

The final layer produces probability scores for possible next words. The model uses these probabilities to select a likely next word and continue generating text.

## Example

Given an input such as:

```text
Life is
```

the model predicts a likely continuation based on patterns learned from the training dataset.

The generated output depends on the model's learned vocabulary and training data.

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd rnn-quote-completion
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Concepts Demonstrated

This project demonstrates practical implementation of:

* Natural Language Processing
* Text preprocessing
* Tokenization
* Vocabulary creation
* Sequence modeling
* Word embeddings
* Recurrent Neural Networks
* Next-word prediction
* Text generation
* Temperature-based sampling
* Neural network training
* Model inference

## Future Improvements

* Replace the basic RNN with LSTM or GRU
* Train on a larger and more diverse dataset
* Improve text generation quality
* Implement Top-K sampling
* Implement Top-P sampling
* Add beam search
* Deploy the model as an API
* Create a production-ready web interface

## Author

**Burhan Arshad**

Computer Science Student | Machine Learning & AI
