"""
pip install torch
pip install transformers
pip install 'accelerate>=0.26.0' # To run on CPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pprint
from time import time

model_name = "h2oai/h2o-danube3-500m-chat"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

# Check if MPS is available and use it, otherwise default to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Choose a smaller model or use CPU if memory is a constraint
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16 if it is not supported
    device_map={
        "": device
    },  # this makes sure the entire model is mapped to a single device
    trust_remote_code=True,
)

t0 = time()
message_content = """
What is the main point of the below summarization of a research paper?

The research paper introduces a new neural long-term memory module designed to 
enhance the capabilities of attention-based deep learning models, especially for 
processing very long sequences.  The authors argue that existing models, such as 
Transformers, have limitations in handling long sequences due to the quadratic 
cost of attention mechanisms. They propose a solution where a neural memory module 
learns to memorize the historical context, allowing the attention mechanism to focus 
on the current context while still utilizing past information.

This neural memory module is inspired by the human brain's memory system, which 
effectively memorizes surprising or unexpected events. The module incorporates a 
"surprise metric" to identify and store important information from the sequence.  
The authors also introduce a decaying mechanism that considers the memory size and 
the level of surprise, enabling efficient memory management.

The paper presents a family of architectures called Titans, which incorporate 
this neural memory module in three different ways. The first variant treats memory 
as a context for the current information. The second variant uses a gated memory 
approach, combining short-term memory from attention with long-term memory from 
the neural memory module. The third variant incorporates memory as a layer within 
the deep learning network. 

Experimental results across various tasks, including language modeling, common-sense 
reasoning, and DNA modeling, demonstrate that Titans outperform existing models, 
especially with long sequences. The authors show that Titans can effectively scale 
to context window sizes larger than 2 million tokens, exceeding the capabilities of 
current Transformer models.
"""

messages = [
    {"role": "user", "content": message_content},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
    device
)  # To make sure that the input is on the correct device

# generate configuration can be modified to your needs
tokens = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    min_new_tokens=2,
    max_new_tokens=256,
)[0]

tokens = tokens[inputs["input_ids"].shape[1] :]
answer = tokenizer.decode(tokens, skip_special_tokens=True)
t1 = time()
print(f"Generation Time: {t1-t0:2f}")
print(t1-t0:)
pprint.pprint(answer)
