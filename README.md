# Sentimental LLM

A modular CLI tool for sentiment analysis. This tool processes raw text through a pipeline of preprocessing, feature extraction, semantic analysis, and deep learning models to produce various text analysis predictions.

## Usage

To use the tool, run the `main.py` script from your terminal and provide a path to a text file as an argument. You can also specify the model type and output type using optional arguments.

```bash
python3 main.py <file_path> [--model_type <model>] [--output_type <output>]
```

### Arguments

*   `file_path`: The path to the input text file to analyze.
*   `--model_type`: The type of model to use. Choices are `lstm`, `rnn`, `transformer`, and `attention`. Default is `lstm`.
*   `--output_type`: The type of output to get. Choices are `text_classification`, `next_word_prediction`, and `sentiment_analysis`. Default is `sentiment_analysis`.

## Example

Create a file named `input.txt` with the following content:
```
This is a sample sentence, showing off the stop words filtration.
```

Now, run the tool with the `input.txt` file:
```bash
python3 main.py input.txt --model_type transformer --output_type sentiment_analysis
```

### Output

```
{
    "Sentiment Analysis": "Sentiment analysis result",
    "Syntactic Analysis": [
        ("This", "DT"),
        ("is", "VBZ"),
        ("a", "DT"),
        ("sample", "JJ"),
        ("sentence", "NN"),
        (",", ","),
        ("showing", "VBG"),
        ("off", "RP"),
        ("the", "DT"),
        ("stop", "NN"),
        ("words", "NNS"),
        ("filtration", "NN"),
        (".", ".")
    ],
    "Context Understanding": "Context understanding placeholder"
}
```

## Syntactic Analysis Labels

The syntactic analysis is performed using the NLTK library, which uses the Penn Treebank tag set. Here is a list of the most common tags and their meanings:

| Tag | Meaning | Example(s) |
|:---:|---|---|
| **CC** | Coordinating conjunction | and, or, but |
| **CD** | Cardinal number | one, two, 1, 2 |
| **DT** | Determiner | a, an, the |
| **EX** | Existential there | there |
| **FW** | Foreign word | bonjour, schadenfreude |
| **IN** | Preposition or subordinating conjunction | in, on, of, while |
| **JJ** | Adjective | big, tall, beautiful |
| **JJR** | Adjective, comparative | bigger, taller |
| **JJS** | Adjective, superlative | biggest, tallest |
| **LS** | List item marker | 1), a) |
| **MD** | Modal | can, could, will, may |
| **NN** | Noun, singular or mass | dog, cat, tree |
| **NNS** | Noun, plural | dogs, cats, trees |
| **NNP** | Proper noun, singular | John, London, Google |
| **NNPS** | Proper noun, plural | Americans, Kardashians |
| **PDT** | Predeterminer | all, both, half |
| **POS** | Possessive ending | 's, ' |
| **PRP** | Personal pronoun | I, you, he, she, it |
| **PRP$** | Possessive pronoun | my, your, his, her, its |
| **RB** | Adverb | quickly, very, silently |
| **RBR** | Adverb, comparative | faster, better |
| **RBS** | Adverb, superlative | fastest, best |
| **RP** | Particle | up, down, off |
| **SYM** | Symbol | +, -, =, * |
| **TO** | to | to go to the store |
| **UH** | Interjection | uh, um, oh |
| **VB** | Verb, base form | go, take, eat |
| **VBD** | Verb, past tense | went, took, ate |
| **VBG** | Verb, gerund or present participle | going, taking, eating |
| **VBN** | Verb, past participle | gone, taken, eaten |
| **VBP** | Verb, non-3rd person singular present | go, take, eat |
| **VBZ** | Verb, 3rd person singular present | goes, takes, eats |
| **WDT** | Wh-determiner | which, that |
| **WP** | Wh-pronoun | who, what |
| **WP$** | Possessive wh-pronoun | whose |
| **WRB** | Wh-adverb | where, when, why |
