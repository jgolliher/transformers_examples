# Transformers Examples

An introduction to HuggingFace's `transformers` package. This includes BERT embedddings, BERT as a classifier, and running LLMs on your local machine.

# 1: Encoders

Provides an overview of encoder-only models (i.e., BERT-based models). These models excel at text classification and are the backbone for retrieval-augmented generation (RAG). This section deals with how to load a BERT model from HuggingFace and obtain embeddings.

# 2: Encoder Classification

Provides an example of using BERT for a downstream classification task. This example uses BERT for entity resolution (i.e., determining if two records refer to the same thing). This example builds upon the [Ditto](https://github.com/megagonlabs/ditto) architecture and uses PyTorch.

# 3: Local LLM

Provides an example of running LLMs on your local machine using the transformers library.