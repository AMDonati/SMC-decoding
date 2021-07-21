from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
import matplotlib.pyplot as plt
from pprint import pprint
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence


class SSTDataset():
    def __init__(self, tokenizer=GPT2Tokenizer.from_pretrained("gpt2")):
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def visualize_labels(self, dataset):
        plt.hist(dataset['label'], 30, density=True, facecolor='g', alpha=0.75)
        plt.show()

    def load_sst_dataset(self):
        dataset = load_dataset('sst')
        train_set, val_set, test_set = dataset["train"], dataset["validation"], dataset["test"]
        return train_set, val_set, test_set

    def tokenize(self, dataset):
      encoded_dataset = dataset.map(lambda example: self.tokenizer(example['sentence']), batched=True)
      #encoded_dataset = dataset.map(lambda example: self.tokenizer(example['sentence'], padding='max_length'),
      #batched=True)
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
        # for i, ex in enumerate(examples):
        #     ex["input_ids"] = inputs_ids[i]
        #     ex["target_ids"] = targets_ids[i]
        # return examples
        #return self.tokenizer.pad(examples, padding='max_length', return_tensors='pt') #TODO: bug here.
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

    def preprocess_dataset(self, dataset, batch_size=32, label=1):
        dataset = self.tokenize(dataset)
        dataset = self.get_binary_label(dataset)
        dataset = self.get_input_target_sequences(dataset)
        dataset = self.remove_neutral_labels(dataset)
        if label is not None:
            dataset = self.filter_per_label(dataset, label=label)
        print("visualizing labels distribution")
        self.visualize_labels(dataset)
        #dataset = self.pad_sequences(dataset)
        dataset = self.get_torch_dataset(dataset)
        #dataset = self.pad_sequences(dataset)
        dataloader = self.create_data_loader(dataset, batch_size)
        return dataset, dataloader


if __name__ == '__main__':
    sst_dataset = SSTDataset()
    train_set, val_set, test_set = sst_dataset.load_sst_dataset()
    train_set, train_dataloader = sst_dataset.preprocess_dataset(train_set)
    train_set.__getitem__(0)
    batch = next(iter(train_dataloader))
    print("done")