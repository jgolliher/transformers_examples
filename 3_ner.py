"""

From HuggingFace: https://huggingface.co/urchade/gliner_medium-v2 
GLiNER is a Named Entity Recognition (NER) model capable of identifying any 
entity type using a bidirectional transformer encoder (BERT-like). It provides a 
practical alternative to traditional NER models, which are limited to predefined 
entities, and Large Language Models (LLMs) that, despite their flexibility, 
are costly and large for resource-constrained scenarios.

This version has been trained on the NuNER dataset (commercially permissive)"""

import gliner
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

###############
# Basic example
###############

text = "Millie is a beagle who loves to eat."
labels = ["dog"]
entities = model.predict_entities(text, labels)

for entity in entities:
    print(entity["text"], "=>", entity["label"])


###############
# Another Example
###############

text = """Apple announced the new iPhone 15 at their headquarters in 
Cupertino, California."""

labels = ["organization", "product", "location"]
entities = model.predict_entities(text, labels)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
