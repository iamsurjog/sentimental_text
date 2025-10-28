# Sentimental LLM

A modular CLI tool for sentiment analysis. This tool processes raw text through a pipeline of preprocessing, feature extraction, semantic analysis, and deep learning models to produce various text analysis predictions.

## Usage

To use the tool, run the `main.py` script from your terminal and provide a string of text as an argument.

```bash
python3 main.py "Your text goes here."
```

## Example

Running the tool with the sample sentence: "This is a sample sentence, showing off the stop words filtration."

```bash
python3 main.py "This is a sample sentence, showing off the stop words filtration."
```

### Output

```
Processed Tokens: ['this', 'is', 'sample', 'sentence', 'showing', 'off', 'stop', 'words', 'filtration']
Generating Word2Vec embeddings...
Word2Vec Embeddings (placeholder): []
Generating GloVe embeddings...
GloVe Embeddings (placeholder): []
Generating BERT embeddings...
BERT Embeddings (placeholder): []

--- Semantic Analysis Module ---
Extracting semantic features...
Semantic Features (placeholder): {}
Understanding context...
Context Understanding (placeholder): {}
Performing syntactic analysis...
Syntactic Analysis (placeholder): {}

--- Deep Learning Models ---
Applying attention mechanism...
Attention Features (placeholder): {}
Running LSTM model...
LSTM Output (placeholder): LSTM output
Running RNN model...
RNN Output (placeholder): RNN output
Running Transformer model...
Transformer Output (placeholder): Transformer output

--- Prediction Module ---
Classifying text...
Predicting next word...
Analyzing sentiment...

--- Output Layer ---
Text Classification: Text classification result
Next Word Prediction: Next word prediction
Sentiment Analysis: Sentiment analysis result
```

