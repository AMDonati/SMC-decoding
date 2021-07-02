import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch.nn.functional as F


# You can get around that behavior by passing add_prefix_space=True when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.
def get_tokens(tokenizer, word_list):
    token_list = []
    for word in word_list:
        token = tokenizer.encode(" " + word)
        if token:
            token_list.append(token[0])
    return token_list

def read_topic_BoW(tokenizer=GPT2Tokenizer.from_pretrained('gpt2') , topic="science", data_path="../../data/wordlists/"):
    file_name = data_path + topic + ".txt"
    with open(file_name, "r") as f:
        words = f.read()
        words = words.split('\n')
    words_tokens = get_tokens(tokenizer=tokenizer, word_list=words)
    return words_tokens

def topic_BoW_function(inputs, bow_tokens, model=GPT2LMHeadModel.from_pretrained('gpt2'), return_logits=False):
    outputs = model(inputs)
    logits = outputs.logits
    if return_logits:
        last_probs = logits[:, -1, :]
    else:
        last_probs = F.softmax(logits[:,-1,:], dim=-1)
    bow_logprobs = last_probs[:,bow_tokens]
    BoW_score = bow_logprobs.sum(-1).log()
    return BoW_score


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # 345B parameters GPT-2 model.
    model = GPT2Model.from_pretrained('gpt2')
    bow_tokens = read_topic_BoW()
    inputs = tokenizer.encode("The potato", return_tensors="pt")
    BoW_score = topic_BoW_function(inputs=inputs, bow_tokens=bow_tokens)
    print('done')