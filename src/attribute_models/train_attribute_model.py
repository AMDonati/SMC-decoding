import time
import torch
import torch.nn.functional as F
import math
from attribute_models.utils_train import write_to_csv, create_logger
from language_models.LSTM import LSTMModel
from language_models.gpt2_finetune import GPT2FTModel
from attribute_models.sst_sentiment import SSTDataset
from transformers import GPT2Tokenizer
from attribute_models.sst_tokenizer import SSTTokenizer
from datasets import load_dataset
import os
import argparse
import numpy as np


def assert_correctness_batch(inputs, targets):
    assert torch.all(torch.eq(inputs[:, 1:], targets[:, :-1])) == 1, "error in inputs/targets"


def train_one_epoch(model, train_generator, optimizer, criterion, device, grad_clip=None, print_interval=10):
    model.train()  # Turns on train mode which enables dropout.
    total_loss = 0.
    start_time = time.time()
    start_time_epoch = time.time()
    for batch, (inputs, targets, attn_mask) in enumerate(train_generator):
        inputs = inputs.to(device)
        #targets = targets.to(device)
        targets = targets.view(targets.size(1) * targets.size(0)).to(device)  # targets (S*B)
        model.zero_grad()
        if model.__class__ == GPT2FTModel:
            output, hidden = model(inputs, attn_mask)
        else:
            output, hidden = model(inputs)  # output (S * B, V), hidden (num_layers,B,1)
        loss = criterion(output, targets)
        loss.backward()
        # clip grad norm:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        # print loss every number of batches
        if (batch + 1) % print_interval == 0:
            print('loss for batch {}: {:5.3f}'.format(batch + 1, total_loss / (batch + 1)))
            print('time for {} training steps: {:5.2f}'.format(print_interval, time.time() - start_time))
            start_time = time.time()

    curr_loss = total_loss / (batch + 1)
    elapsed = time.time() - start_time_epoch

    return curr_loss, elapsed


def evaluate(model, val_generator, criterion, device):
    model.eval()  # turn on evaluation mode which disables dropout.
    total_loss = 0.
    start_time = time.time()
    with torch.no_grad():
        for batch, (inputs, targets, attn_mask) in enumerate(val_generator):
            inputs = inputs.to(device)
            targets = targets.view(targets.size(1) * targets.size(0)).to(device)
            if model.__class__ == GPT2FTModel:
                output, hidden = model(inputs, attn_mask)
            else:
                output, hidden = model(inputs)
            total_loss += criterion(output, targets).item()
    print("Evaluation time {:5.2f}".format(time.time() - start_time))
    return total_loss / (batch + 1)


def generate_text_lm(model, tokenizer, device, out_path, temperatures=["greedy", 0.7, 1, 2], num_words=50, prompt="The"):
    dict_words = {k: [] for k in temperatures}
    for temp in temperatures:
        input_idx = tokenizer.encode(prompt, return_tensors="pt")
        input_idx = input_idx.view(1, 1).to(device)
        with torch.no_grad():
            for i in range(num_words):
                _, logits = model(input_idx)  # output (S, num_tokens)
                if temp != "greedy":
                    word_weights = logits[-1].squeeze().div(
                        temp).exp()  # (exp(1/temp * logits)) = (p_i^(1/T))
                    word_weights = word_weights / word_weights.sum(dim=-1).cpu()
                    word_idx = torch.multinomial(word_weights, num_samples=1)[
                        0]  # [0] to have a scalar tensor.
                else:
                    word_idx = logits[-1].squeeze().argmax()
                input_idx = torch.cat([input_idx, word_idx.view(1, 1)], dim=-1)
            words = tokenizer.decode(input_idx.squeeze().cpu())
        dict_words[temp] = words
        out_file_generate = os.path.join(out_path,
                                         'generate_words_temp_{}_prompt_{}.txt'.format(temp, prompt))
        with open(out_file_generate, 'w') as f:
            f.write(dict_words[temp])
            f.close()


def train(model, train_generator, val_generator, optimizer, criterion, device, out_path, logger=None, grad_clip=None,
          EPOCHS=1, print_interval=10):
    train_loss_history, train_ppl_history, val_loss_history, val_ppl_history = [], [], [], []
    out_path_model = os.path.join(out_path, "model.pt")
    best_val_loss = None
    for epoch in range(EPOCHS):
        print('epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_loss, elapsed = train_one_epoch(model=model,
                                              train_generator=train_generator,
                                              optimizer=optimizer,
                                              criterion=criterion,
                                              device=device,
                                              grad_clip=grad_clip,
                                              print_interval=print_interval)
        print('train loss {:5.3f} - train perplexity {:8.3f}'.format(train_loss, math.exp(train_loss)))
        print('time for one epoch...{:5.2f}'.format(elapsed))
        val_loss = evaluate(model=model, val_generator=val_generator, criterion=criterion,
                            device=device)
        print('val loss: {:5.3f} - val perplexity: {:8.3f}'.format(val_loss, math.exp(val_loss)))

        # saving loss and metrics information.
        train_loss_history.append(np.round(train_loss,3))
        train_ppl_history.append(np.round(math.exp(train_loss), 2))
        val_loss_history.append(np.round(val_loss, 2))
        val_ppl_history.append(np.round(math.exp(val_loss), 2))
        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(out_path_model, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss

    print("saving loss and metrics information...")
    hist_keys = ['train_loss', 'train_ppl', 'val_loss', 'val_ppl']
    hist_dict = dict(zip(hist_keys, [train_loss_history, train_ppl_history, val_loss_history, val_ppl_history]))
    write_to_csv(os.path.join(out_path, "train_history.csv"), hist_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-out_path", type=str, default="../../output/sst_attribute_model", help="out path ")
    # model params.
    parser.add_argument("-model", type=str, default="gpt2", help="lstm or gpt-2 fine-tune model")
    parser.add_argument("-tokenizer", type=str, default="gpt2", help="using gpt2 tokenizer or sst vocab.")
    parser.add_argument("-label", type=int, default=1, help="train on positive or negative label.")
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-emb_size", type=int, default=32, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=64, help="dimension of the hidden state")
    # SL algo args.
    parser.add_argument("-p_drop", type=float, default=0.1, help="dropout rate")
    parser.add_argument("-grad_clip", type=float)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=1, help="number of epochs")
    parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader") #TODO: add this in the loader.

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # out files
    out_path = os.path.join(args.out_path, "{}_tok-{}-{}L_{}E_{}H_drop{}_gradclip-{}_bs{}".format(args.model,
                                                                                                       args.tokenizer,
                                                                                                args.num_layers,
                                                                                                args.emb_size,
                                                                                                args.hidden_size,
                                                                                                args.p_drop,
                                                                                                args.grad_clip, args.bs))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file_log = os.path.join(out_path, "training_log.log")
    # logger = create_logger(out_file_log)

    # build tokenizer:
    if args.tokenizer == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif args.tokenizer == "sst":
        dataset = load_dataset("sst", split='train+validation+test') #TODO: add label argument and choose between positive / negative tokenizer.
        tokenizer = SSTTokenizer(dataset, label=args.label)

    # load dataset
    sst_dataset = SSTDataset()
    train_set, val_set, test_set = sst_dataset.load_sst_dataset()
    train_set = sst_dataset.preprocess_dataset(train_set)
    val_set = sst_dataset.preprocess_dataset(val_set)
    train_set, train_dataloader = sst_dataset.prepare_data_for_torch(train_set, batch_size=args.bs)
    val_set, val_dataloader = sst_dataset.prepare_data_for_torch(val_set, batch_size=args.bs)

    # Build model
    if args.model == "lstm":
        model = LSTMModel(num_tokens=sst_dataset.len_vocab,
                          emb_size=args.emb_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          p_drop=args.p_drop).to(device)
    elif args.model == "gpt2":
        model = GPT2FTModel(vocab_size=sst_dataset.len_vocab)

    # train parameters
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    PAD_IDX = sst_dataset.get_PAD_IDX(args)
    criterion = torch.nn.NLLLoss(ignore_index=PAD_IDX)

    # train model
    train(model=model, train_generator=train_dataloader, val_generator=val_dataloader, criterion=criterion,
          optimizer=optimizer, device=device, out_path=out_path, EPOCHS=args.ep)
    # generate text post-training:
    generate_text_lm(model=model, tokenizer=sst_dataset.tokenizer, device=device, out_path=out_path)