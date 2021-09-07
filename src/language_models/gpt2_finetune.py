from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch.nn as nn
import torch.nn.functional as F

class GPT2FTModel(nn.Module):
    def __init__(self, vocab_size, device, hidden_size=768):
        super(GPT2FTModel, self).__init__()
        self.configuration = GPT2Config(use_cache=True, output_hidden_states=True, vocab_size=vocab_size)
        self.model = GPT2LMHeadModel(self.configuration).from_pretrained("cache/gpt2")
        for param in self.model.parameters():
            param.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.trainable_layer = nn.Linear(hidden_size, vocab_size).to(device)

    def forward(self, input, attn_mask=None):
        if attn_mask is None:
            outputs = self.model(input_ids=input, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input, attention_mask=attn_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] # shape (B,S,hidden_size)
        logits = self.trainable_layer(last_hidden_state)
        log_probas = F.log_softmax(logits, dim=-1)
        log_probas = log_probas.view(log_probas.size(0)*log_probas.size(1), -1)
        return log_probas, logits

# to compute the number of trainable parameters;
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])