from attribute_models.sst_sentiment import SSTDataset
from transformers import GPT2Tokenizer

if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")
    sst_dataset = SSTDataset(tokenizer=tokenizer)
    train_set, val_set, test_set = sst_dataset.load_sst_dataset()
    for (split, dataset) in zip(["train", "val", "test"],[train_set, val_set, test_set]):
        sst_dataset.plot_most_frequent_words(dataset, split=split)
        sst_dataset.plot_len_reviews(train_set, split=split)



