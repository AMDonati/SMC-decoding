import torch
import transformers

class LanguageModel:
    def __init__(self, pretrained_lm, dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 tokenizer=None, prefix_tokenizer="", lm_path=None):
        self.device = device
        self.tokenizer = tokenizer
        self.language_model = pretrained_lm.to(self.device)
        self.dataset = dataset
        self.prefix_tokenizer = prefix_tokenizer
        self.init_text = None
        self.init_text_short = None
        self.lm_path = lm_path

    def encode(self, **kwargs):
        return self.tokenizer.encode(**kwargs)

    def decode(self, **kwargs):
        return self.tokenizer.decode(**kwargs)