from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
import matplotlib.pyplot as plt
from pprint import pprint
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence
import os
import json
from attribute_models.sst_tokenizer import SSTTokenizer


class SSTDataset():
    def __init__(self, data_path="../../data/sst", tokenizer=GPT2Tokenizer.from_pretrained("gpt2")):
        self.tokenizer = tokenizer
        #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.vocab = None
        # self.vocab_path = os.path.join(data_path, "vocab.json")
        # self.SPECIAL_TOKENS = {
        #     '<PAD>': 0,
        #     '<UNK>': 1,
        # }

    def visualize_labels(self, dataset):
        plt.hist(dataset['label'], 30, density=True, facecolor='g', alpha=0.75)
        plt.show()

    def load_sst_dataset(self):
        dataset = load_dataset('sst')
        train_set, val_set, test_set = dataset["train"], dataset["validation"], dataset["test"]
        return train_set, val_set, test_set

    def tokenize(self, dataset):
        if self.tokenizer.__class__ == GPT2Tokenizer:
            encoded_dataset = dataset.map(lambda example: self.tokenizer(example['sentence']), batched=True)
        elif self.tokenizer.__class__ == SSTTokenizer:
            encoded_dataset = dataset.map(lambda example: self.tokenizer.encode(example['sentence']))
        print("encoded_dataset[0]")
        pprint(encoded_dataset[0], compact=True)
        return encoded_dataset

    def get_tokens(self, dataset, add_start_token=False, add_end_token=False, punct_to_remove=[]):
        def tokenize_sentence(example):
            example["tokens"] = example["tokens"].split("|")
            return example
        processed_dataset = dataset.map(tokenize_sentence)
        return processed_dataset

    def build_vocab(self, dataset, min_token_count=2, add_start_token=False, add_end_token=False, punct_to_remove=[]):
        token_to_count = {}
        start_tokens = []
        for seq_tokens in dataset["tokens"]:
            start_tokens.append(seq_tokens[0])
            for token in seq_tokens:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1
        # remove "" token
        if "" in list(token_to_count.keys()):
            del token_to_count[""]
        token_to_idx = {}
        for token, idx in self.SPECIAL_TOKENS.items():
            token_to_idx[token] = idx
        for token, count in sorted(token_to_count.items()):
            if count >= min_token_count:
                token_to_idx[token] = len(token_to_idx)
        # getting the unique starting words.
        start_tokens = list(set(start_tokens))
        # saving vocab:
        with open(self.vocab_path, 'w') as f:
            json.dump(token_to_idx, f)
        self.vocab = token_to_idx
        return token_to_idx, start_tokens, token_to_count


    def get_binary_label(self, dataset):
        def binarize_label(example):
            label = example['label']
            if label > 0.5:
                example["binary_label"] = 1
            else:
                example["binary_label"] = 0
            return example
        processed_dataset = dataset.map(binarize_label)
        return processed_dataset

    def remove_neutral_labels(self, dataset, label_min=0.4, label_max=0.6):
        trimmed_dataset = dataset.filter(lambda example: example['label'] < label_min or example['label'] > label_max)
        return trimmed_dataset

    def filter_per_label(self, dataset, label=1):
        dataset_with_label = dataset.filter(lambda example: example['binary_label']==label)
        return dataset_with_label

    def get_input_target_sequences(self, dataset):
        def split_input_target(example):
            example['target_ids'] = example['input_ids'][1:]
            example['input_ids'] = example['input_ids'][:-1]
            return example
        processed_dataset = dataset.map(split_input_target)
        return processed_dataset

    def get_torch_dataset(self, dataset, columns=['input_ids', 'target_ids', 'attention_mask']):
        dataset.set_format(type='torch', columns=columns)
        return dataset

    #Instantiate a PyTorch Dataloader around our dataset
    #Let's do dynamic batching (pad on the fly with our own collate_fn)

    def collate_fn(self, examples):
        inputs_ids = [ex["input_ids"] for ex in examples]
        targets_ids = [ex["target_ids"] for ex in examples]
        attn_mask = [ex["attention_mask"][:-1] for ex in examples]
        inputs_ids = pad_sequence(inputs_ids, batch_first=True, padding_value=50257)
        targets_ids = pad_sequence(targets_ids, batch_first=True, padding_value=50257)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        return inputs_ids, targets_ids, attn_mask
    #
    def pad_sequences(self, dataset):
        def pad(example):
            example["input_ids"] = pad_sequence([torch.tensor(e) for e in example["input_ids"]], batch_first=True, padding_value=50257)
            example["target_ids"] = pad_sequence([torch.tensor(e) for e in example["target_ids"]], batch_first=True, padding_value=50257)
            example["attention_mask"] = pad_sequence([torch.tensor(e) for e in example["attention_mask"]], batch_first=True, padding_value=50257)
            return example
        processed_dataset = dataset.map(pad)
        return processed_dataset

    def create_data_loader(self, dataset, batch_size):
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=self.collate_fn, batch_size=batch_size)
        #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return dataloader

    def preprocess_dataset(self, dataset, label=1):
        dataset = self.tokenize(dataset)
        dataset = self.get_tokens(dataset)
        dataset = self.get_binary_label(dataset)
        dataset = self.get_input_target_sequences(dataset)
        dataset = self.remove_neutral_labels(dataset)
        if label is not None:
            dataset = self.filter_per_label(dataset, label=label)
        print("visualizing labels distribution")
        self.visualize_labels(dataset)
        return dataset

    def prepare_data_for_torch(self, dataset, batch_size=32):
        dataset = self.get_torch_dataset(dataset)
        dataloader = self.create_data_loader(dataset, batch_size)
        return dataset, dataloader


if __name__ == '__main__':
    print("SST dataset with GPT2 tokenizer")
    sst_dataset = SSTDataset()
    train_set, val_set, test_set = sst_dataset.load_sst_dataset()
    train_set = sst_dataset.preprocess_dataset(train_set)
    #vocab, start_tokens, tokens_to_count = sst_dataset.build_vocab(train_set)
    train_set, train_dataloader = sst_dataset.prepare_data_for_torch(train_set)
    train_set.__getitem__(0)
    batch = next(iter(train_dataloader))
    print("-----------------------------------------------------------------------------")
    print("SST dataset with SST tokenizer")
    dataset = load_dataset("sst", split='train+validation+test')
    sst_tokenizer = SSTTokenizer(dataset=dataset)
    sst_dataset_sst = SSTDataset(tokenizer=sst_tokenizer)
    train_set, val_set, test_set = sst_dataset_sst.load_sst_dataset()
    train_set = sst_dataset_sst.preprocess_dataset(train_set)
    # vocab, start_tokens, tokens_to_count = sst_dataset.build_vocab(train_set)
    train_set, train_dataloader = sst_dataset.prepare_data_for_torch(train_set)