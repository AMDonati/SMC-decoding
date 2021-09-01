from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
from pprint import pprint
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from attribute_models.sst_tokenizer import SSTTokenizer


class SSTDataset():
    def __init__(self, tokenizer=GPT2Tokenizer.from_pretrained("gpt2")):
        self.tokenizer = tokenizer
        self.len_vocab = self.get_len_vocab()

    def get_len_vocab(self):
        if self.tokenizer.__class__ == GPT2Tokenizer:
            len_vocab = len(self.tokenizer.decoder)
        elif self.tokenizer.__class__ == SSTTokenizer:
            len_vocab = len(self.tokenizer.vocab)
        return len_vocab

    def get_PAD_IDX(self, args):
        if args.tokenizer == "gpt2":
            PAD_IDX = 50256
        elif args.tokenizer == "sst":
            PAD_IDX = 0
        return PAD_IDX

    def visualize_labels(self, dataset):
        plt.hist(dataset['label'], 30, density=True, facecolor='g', alpha=0.75)
        plt.show()

    def load_sst_dataset(self):
        dataset = load_dataset('sst')
        train_set, val_set, test_set = dataset["train"], dataset["validation"], dataset["test"]
        return train_set, val_set, test_set

    def tokenize(self, dataset):
        def tokenize_example(example):
            example["input_ids"] = self.tokenizer.encode(example['sentence'])
            example["attention_mask"] = [1] * len(example["input_ids"])
            return example
        if self.tokenizer.__class__ == GPT2Tokenizer:
            encoded_dataset = dataset.map(lambda example: self.tokenizer(example['sentence']), batched=True)
        elif self.tokenizer.__class__ == SSTTokenizer:
            encoded_dataset = dataset.map(tokenize_example)
        print("encoded_dataset[0]")
        pprint(encoded_dataset[0], compact=True)
        return encoded_dataset

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

    def collate_fn(self, examples):
        inputs_ids = [ex["input_ids"] for ex in examples]
        targets_ids = [ex["target_ids"] for ex in examples]
        attn_mask = [ex["attention_mask"][:-1] for ex in examples]
        inputs_ids = pad_sequence(inputs_ids, batch_first=True, padding_value=50256)
        targets_ids = pad_sequence(targets_ids, batch_first=True, padding_value=50256)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        return inputs_ids, targets_ids, attn_mask

    def create_data_loader(self, dataset, batch_size):
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=self.collate_fn, batch_size=batch_size)
        return dataloader

    def preprocess_dataset(self, dataset, label=1):
        dataset = self.tokenize(dataset)
        dataset = self.get_binary_label(dataset)
        dataset = self.get_input_target_sequences(dataset)
        dataset = self.remove_neutral_labels(dataset)
        if label is not None:
            dataset = self.filter_per_label(dataset, label=label)
        #print("visualizing labels distribution")
        #self.visualize_labels(dataset)
        return dataset

    def check_number_unk_tokens(self, dataset):
        inputs_ids = dataset["input_ids"]
        num_unk = 0
        num_tokens = 0
        for inp in inputs_ids:
            len_ = len(inp)
            num_tokens += len_
            mask = inp == 1
            unk = mask.sum().item()
            num_unk += unk
        return num_unk / num_tokens, num_unk

    def prepare_data_for_torch(self, dataset, batch_size=32):
        dataset = self.get_torch_dataset(dataset)
        dataloader = self.create_data_loader(dataset, batch_size)
        return dataset, dataloader


if __name__ == '__main__':
    print("SST dataset with GPT2 tokenizer")
    sst_dataset = SSTDataset()
    train_set, val_set, test_set = sst_dataset.load_sst_dataset()
    train_set = sst_dataset.preprocess_dataset(train_set)
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
    percent_unk, num_unk = sst_dataset_sst.check_number_unk_tokens(train_set)
    print("done")