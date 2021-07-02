from transformers import GPT2Tokenizer, GPT2Model
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')  # 345B parameters GPT-2 model.
model = GPT2Model.from_pretrained('gpt2-medium')

def generate_text(prompt, model, tokenizer, score_function):
    inputs = tokenizer("The potato", return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    decoded_output = tokenizer.decode(outputs)
    return decoded_output