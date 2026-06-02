# Mini FAISS Semantic Search Engine

## Overview

This project demonstrates how semantic search works using sentence embeddings and FAISS.

Instead of matching exact keywords, semantic search understands the meaning of a query and retrieves the most relevant results from a knowledge base.

The project uses:

* Sentence Transformers for embedding generation
* FAISS for vector storage and similarity search
* NumPy for vector processing

---

## Features

* Generate sentence embeddings using all-MiniLM-L6-v2
* Store embeddings in a FAISS vector index
* Perform semantic similarity search
* Retrieve Top-3 most relevant results
* Interactive command-line interface
* Demonstrates concepts used in Retrieval-Augmented Generation (RAG) systems

---

## Project Structure

```text
MINI_FAISS_SEMANTIC_SEARCH_ENGINE/
│
├── faiss_semantic_search.py
├── requirements.txt
├── reflection_answers.md
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd MINI_FAISS_SEMANTIC_SEARCH_ENGINE
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python faiss_semantic_search.py
```

---

## Sample Queries

```text
How do I recover my password?

When will my refund be processed?

Why am I unable to sign in?
```

---

## Example Output

```text
Rank      Score          Matched Sentence

1         0.1234         Forgotten passwords can be recovered through the password reset portal.
2         0.2781         Sign-in issues may result from invalid login credentials.
3         0.4125         A verified email address is required for account activation.
```

---

## Concepts Demonstrated

### Embeddings

Text is converted into numerical vectors using the Sentence Transformer model.

### Vector Search

FAISS stores vectors and performs similarity search efficiently.

### Semantic Search

Queries are matched based on meaning rather than exact keyword overlap.

### Cosine Similarity

Embeddings are normalized before indexing to enable cosine similarity behavior.

### Retrieval-Augmented Generation (RAG)

Semantic retrieval is one of the core building blocks of modern RAG systems.

---

## Technologies Used

* Python
* NumPy
* Sentence Transformers
* FAISS

---

## Author

Amogh Sumbad
