from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch.nn as nn
import torch.nn.functional as F
import torch
from smc.utils import constant_noise, decreasing_noise_with_time

class GPT2FTModel(nn.Module):
    def __init__(self, vocab_size, device, hidden_size=768, init_weight=True):
        super(GPT2FTModel, self).__init__()
        self.configuration = GPT2Config(use_cache=True, output_hidden_states=True, vocab_size=vocab_size)
        self.model = GPT2LMHeadModel(self.configuration).from_pretrained("cache/gpt2").to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.trainable_layer = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
        if init_weight:
            self.trainable_layer.weight.data = self.model.lm_head.weight.data
        self.device = device

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

    def get_hidden_from_input(self, input, attn_mask=None, sigma=0.5, noise_function=constant_noise):
        if attn_mask is None:
            outputs = self.model(input_ids=input, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input, attention_mask=attn_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1].squeeze(-2) # shape (B,S,hidden_size)
        last_hidden_state = last_hidden_state[:,-1,:].unsqueeze(1)
        if sigma is not None:
            seq_len = last_hidden_state.shape[1]
            std_tensor = noise_function(sigma=sigma, seq_len=seq_len)
            last_hidden_state = self.add_noise(last_hidden_state, std_tensor) #TODO: check hidden_states meaning.
        return last_hidden_state

    def predict_from_hidden(self, hidden):
        logits = self.trainable_layer(hidden) # (P,S,V)
        all_probas = F.softmax(logits, dim=-1)
        probas = all_probas[:,-1,:]
        return probas, all_probas

    def get_new_hidden(self, hidden, observation, sigma=0.5, noise_function=constant_noise):
        outputs = self.model(input_ids=observation, output_hidden_states=True)
        current_hidden = outputs.hidden_states[-1] # shape (1, S, hidden_size)
        seq_len = current_hidden.shape[1]
        std_tensor = noise_function(sigma=sigma, seq_len=seq_len)
        current_hidden = self.add_noise(current_hidden, std_tensor)
        new_hidden = torch.cat([hidden, current_hidden], dim=-2)
        return new_hidden, current_hidden

    def add_noise(self, params, std_tensor):
        '''
        :param params: tensor to which noise should be added.
        :param sigma: covariance matrix.
        :return:
        '''
        std_tensor_ = std_tensor.view(1,std_tensor.size(0),1).repeat(params.size(0), 1, params.size(-1)).to(self.device)
        gaussian_noise = torch.normal(mean=params.new_zeros(params.size()), std=std_tensor_)
        return params + gaussian_noise

    def generate_input_word_sequences(self, prompt, max_length=50, top_k=0, seed=None, temperature=1.0):
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        len_prompt = inputs.shape[-1]
        if temperature != "greedy":
            # Sampling decoding.
            sample_output = self.model.generate(
                inputs,
                do_sample=True,
                max_length=max_length,
                top_k=top_k,
                temperature=temperature
            )
        else:
            # Sampling decoding.
            sample_output = self.model.generate(
                inputs,
                do_sample=False,
                max_length=max_length,
                top_k=top_k)
        print("Output:\n" + 100 * '-')
        decoded_output = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
        print(decoded_output)
        sample_output = sample_output[:,len_prompt:]
        return sample_output.unsqueeze(-1), decoded_output

# to compute the number of trainable parameters;
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])