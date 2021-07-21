from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
# Different types of classic decoding:

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # 345B parameters GPT-2 model.
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

inputs = tokenizer.encode("The potato", return_tensors="pt")
#outputs = model(**inputs)

#last_hidden_states = outputs.last_hidden_state

# greedy decoding
greedy_output = model.generate(inputs, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# beam search decoding
beam_output = model.generate(
    inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# Sampling decoding.
sample_output = model.generate(
    inputs,
    do_sample=True,
    max_length=50,
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# temperature sampling.
sample_output = model.generate(
    inputs,
    do_sample=True,
    max_length=50,
    top_k=0,
    temperature=0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# top k sampling
sample_output = model.generate(
    inputs,
    do_sample=True,
    max_length=50,
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# top-p sampling.
sample_output = model.generate(
    inputs,
    do_sample=True,
    max_length=50,
    top_p=0.92,
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))