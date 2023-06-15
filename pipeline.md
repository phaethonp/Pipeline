# Pipeline

## Rationale
by structuring the process as independent and modular functions, we can turn each function into a FastAPI endpoint or an AWS Lambda function, and deploy the whole set as a scalable, serverless API using AWS API Gateway.

We use the code we have implemented in this notebook: https://colab.research.google.com/drive/15Ra6OuNE9fcjI8eVg9OrqxE25-gAv2n2#scrollTo=ckWRI8e5n2TB


## 1. Creating and saving embeddings

The pseudocode:


```python

def load_corpus():
    # Load the corpus from a source (a file, a database, etc.)
    ...

def load_model(model_name):
    # Load the Transformer model
    ...

def encode_corpus(corpus, model):
    # Encode the entire corpus and return the embeddings
    ...

def save_embeddings(embeddings):
    # Save embeddings to a database
    ...

def main():
    corpus = load_corpus()
    model = load_model('model_name')
    embeddings = encode_corpus(corpus, model)
    save_embeddings(embeddings)
    
  ```
    
This approach separates each step into its own function for better code readability and maintainability. It also allows for flexibility in where the corpus is loaded from and how the embeddings are saved.

## 2. Creating question + Index embeddings
#### TO DO : clarify index type selection if it is included in this API

The pseudocode:

```python

def load_question():
    # Load the question from a source (user input, a file, a database, etc.)
    ...

def encode_question(question, model):
    # Encode the question and return the embedding
    ...

def main():
    question = load_question()
    model = load_model('model_name')
    question_embedding = encode_question(question, model)
    
```
    
    
This revision allows the question to come from any source, not just direct user input.

## 3. Using ChatGPT as a predictor and saving response to ChromaDB

```python
def load_chatgpt_model():
    # Load the ChatGPT model
    ...

def generate_response(question, doc, chatgpt_model):
    # Generate a response using the ChatGPT model
    ...

def save_response(response):
    # Save the response to a database
    ...

def main():
    question = load_question()
    doc = get_related_doc(question)
    chatgpt_model = load_chatgpt_model()
    response = generate_response(question, doc, chatgpt_model)
    save_response(response)
    
```













