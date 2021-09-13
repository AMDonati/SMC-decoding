from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch.nn as nn
import torch.nn.functional as F
import torch

class GPT2FTModel(nn.Module):
    def __init__(self, vocab_size, device, hidden_size=768):
        super(GPT2FTModel, self).__init__()
        self.configuration = GPT2Config(use_cache=True, output_hidden_states=True, vocab_size=vocab_size)
        self.model = GPT2LMHeadModel(self.configuration).from_pretrained("cache/gpt2").to(device)
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
        return log_probas, logits, last_hidden_state

    def get_hidden_from_input(self, input, attn_mask=None, sigma=0.5):
        if attn_mask is None:
            outputs = self.model(input_ids=input, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input, attention_mask=attn_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1].squeeze() # shape (B,S,hidden_size)
        if sigma is not None:
            last_hidden_state = self.add_noise(last_hidden_state, sigma) #TODO: check hidden_states meaning.
        return last_hidden_state

    def predict_from_hidden(self, hidden):
        logits = self.trainable_layer(hidden) # (P,S,V)
        all_probas = F.softmax(logits, dim=-1)
        probas = all_probas[:,-1,:]
        return probas, all_probas

    def get_new_hidden(self, hidden, observation, sigma=0.5):
        outputs = self.model(input_ids=observation, output_hidden_states=True)
        current_hidden = outputs.hidden_states[-1] # shape (1, S, hidden_size)
        current_hidden = self.add_noise(current_hidden, sigma)
        new_hidden = torch.cat([hidden, current_hidden], dim=-2)
        return new_hidden, current_hidden

    def add_noise(self, params, sigma):
        '''
        :param params: tensor to which noise should be added.
        :param sigma: covariance matrix.
        :return:
        '''
        gaussian_noise = torch.normal(mean=params.new_zeros(params.size()), std=params.new_ones(params.size()))
        noise = (sigma) ** (1 / 2) * gaussian_noise #here sigma is a variance.
        return params + noise

    def generate_input_word_sequences(self, prompt, max_length=50, top_k=0):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        # Sampling decoding.
        sample_output = self.model.generate(
            inputs,
            do_sample=True,
            max_length=max_length,
            top_k=top_k
        )
        print("Output:\n" + 100 * '-')
        print(self.tokenizer.decode(sample_output[0], skip_special_tokens=True))
        return sample_output.unsqueeze(-1)

# to compute the number of trainable parameters;
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])