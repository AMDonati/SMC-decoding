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
from datasets import load_from_disk
import os
import argparse
import numpy as np
import datetime
from scripts.decode_with_attribute import save_hparams


def assert_correctness_batch(inputs, targets):
    assert torch.all(torch.eq(inputs[:, 1:], targets[:, :-1])) == 1, "error in inputs/targets"


def train_one_epoch(model, train_generator, optimizer, criterion, kl_criterion, device, grad_clip=None, print_interval=10, KL_coeff=0.0, kl_model=None):
    model.train()  # Turns on train mode which enables dropout.
    total_loss = 0.
    start_time = time.time()
    start_time_epoch = time.time()
    for batch, (inputs, targets, attn_mask) in enumerate(train_generator):
        inputs = inputs.to(device)
        attn_mask = attn_mask.to(device)
        targets = targets.view(targets.size(1) * targets.size(0)).to(device)  # targets (S*B)
        model.zero_grad()
        if model.__class__ == GPT2FTModel:
            output, hidden = model(inputs, attn_mask)
        else:
            output, hidden = model(inputs)  # output (S * B, V), hidden (num_layers,B,1)
        ce_loss = criterion(output, targets)
        if KL_coeff > 0 and kl_model is not None:
            gpt_output, _ = kl_model(inputs, attn_mask)
            kl_loss = KL_coeff * kl_criterion(output, gpt_output.detach())
        else:
            kl_loss = 0.
        loss = ce_loss + kl_loss
        loss.backward()
        # clip grad norm:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += ce_loss.item()
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
            attn_mask = attn_mask.to(device)
            inputs = inputs.to(device)
            targets = targets.view(targets.size(1) * targets.size(0)).to(device)
            if model.__class__ == GPT2FTModel:
                output, hidden = model(inputs, attn_mask)
            else:
                output, hidden = model(inputs)
            total_loss += criterion(output, targets).item()
    print("Evaluation time {:5.2f}".format(time.time() - start_time))
    return total_loss / (batch + 1)


def generate_text_lm(model, tokenizer, device, out_path, kl_model, temperatures=["greedy", 0.7], num_words=50, prompt="The", num=5):
    model.eval()
    kl_model.eval()
    dict_words = {k: "" for k in temperatures}
    dict_words_gpt = {k: "" for k in temperatures}
    for temp in temperatures:
        for n in range(num):
            input_idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if len(input_idx.size()) == 1:
                input_idx = input_idx.view(1, input_idx.shape[0])
            with torch.no_grad():
                for i in range(num_words):
                    _, logits = model(input_idx)  # output (S, num_tokens)
                    if temp != "greedy":
                        logits_with_temp = logits[:,-1,:].squeeze().div(
                            temp)  # (exp(1/temp * logits)) = (p_i^(1/T))
                        word_weights = F.softmax(logits_with_temp, dim=-1)
                        word_idx = torch.multinomial(word_weights, num_samples=1)[
                            0]  # [0] to have a scalar tensor.
                    else:
                        word_idx = logits[:,-1,:].squeeze().argmax()
                    input_idx = torch.cat([input_idx, word_idx.view(1, 1)], dim=-1)
                words = tokenizer.decode(input_idx.squeeze().cpu().numpy())
            dict_words[temp] = dict_words[temp] + '\n' + '\n' + words
            _, words_gpt = kl_model.generate_input_word_sequences(prompt=prompt, temperature=temp, max_length=num_words)
            dict_words_gpt[temp] = dict_words_gpt[temp] + '\n' + '\n' + words_gpt
        out_file_generate = os.path.join(out_path,
                                         'generate_words_temp_{}_prompt_{}.txt'.format(temp, prompt))
        with open(out_file_generate, 'w') as f:
            f.write(dict_words[temp])
            f.write('\n -----------------------------GPT2 TEXT GENERATION-------------------------- \n')
            f.write(dict_words_gpt[temp])
            f.close()


def train(model, train_generator, val_generator, optimizer, criterion, device, out_path, logger=None, grad_clip=None,
          EPOCHS=1, print_interval=10, KL_coeff=0.0, kl_model=None):
    train_loss_history, train_ppl_history, val_loss_history, val_ppl_history = [], [], [], []
    out_path_model = os.path.join(out_path, "model.pt")
    best_val_loss = None
    # for the KL regularization term
    kl_loss = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
    for epoch in range(EPOCHS):
        print('epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_loss, elapsed = train_one_epoch(model=model,
                                              train_generator=train_generator,
                                              optimizer=optimizer,
                                              criterion=criterion,
                                              kl_criterion=kl_loss,
                                              device=device,
                                              grad_clip=grad_clip,
                                              print_interval=print_interval,
                                              KL_coeff=KL_coeff,
                                              kl_model=kl_model)
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


def get_sst_tokenizer_vocab(args):
    if args.label_vocab is None:
        vocab_path = "data/sst/vocab_mincount{}.json".format(args.min_count)
    else:
        vocab_path = "data/sst/vocab_mincount{}_label{}.json".format(args.min_count, args.label_vocab)
    return vocab_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-out_path", type=str, default="output/temp", help="out path ")
    # model params.
    parser.add_argument("-model", type=str, default="gpt2", help="lstm or gpt-2 fine-tune model")
    parser.add_argument("-model_path", type=str, default="output/sst_attribute_model_new/gpt2_lr5e-05_gradclip-None_bs16_KLcoeff0.5_initweight1/1/model.pt", help="path if starting with a trained_model.")
    parser.add_argument("-tokenizer", type=str, default="gpt2", help="using gpt2 tokenizer or sst vocab.")
    parser.add_argument("-min_count", type=int, default=2, help="for choosing sst tokenizer vocab.")
    parser.add_argument("-label_vocab", type=int, help="for choosing sst tokenizer vocab (all words or positive/negative.)")
    parser.add_argument("-label", type=int, default=1, help="train on positive or negative label.")
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-emb_size", type=int, default=32, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=64, help="dimension of the hidden state")
    parser.add_argument("-init_weight", type=int, default=1, help="for GPT2FT Model, if last layer weight is initialized with GPT2 weight")
    # SL algo args.
    parser.add_argument("-KL_coeff", type=float, default=0., help='KL regularization term with GPT2 in the loss.')
    parser.add_argument("-p_drop", type=float, default=0., help="dropout rate")
    parser.add_argument("-grad_clip", type=float)

    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=0, help="number of epochs")
    parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader") #TODO: add this in the loader.

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # out files
    tok_string = args.tokenizer if args.tokenizer == "gpt2" else "{}-count{}-lv{}".format(args.tokenizer, args.min_count, args.label_vocab)
    if args.model_path is not None:
        out_folder = os.path.dirname(args.model_path)
        out_path = os.path.join(out_folder, "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    else:
        if args.model == "lstm":
            out_path = os.path.join(args.out_path, "{}_tok-{}_{}E_{}H_p{}_lr{}_gradclip-{}_bs{}_KLcoeff{}".format(args.model,
                                                                                                        tok_string,
                                                                                                        args.emb_size,
                                                                                                        args.hidden_size,
                                                                                                        args.p_drop,
                                                                                                        args.lr,
                                                                                                        args.grad_clip, args.bs, args.KL_coeff))
        elif args.model == "gpt2":
            out_path = os.path.join(args.out_path, "{}_lr{}_gradclip-{}_bs{}_KLcoeff{}_initweight{}".format(args.model, args.lr,
                                                                                                    args.grad_clip,
                                                                                                    args.bs,
                                                                                                    args.KL_coeff,
                                                                                                  args.init_weight))
        out_path = os.path.join(out_path, "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file_log = os.path.join(out_path, "training_log.log")
    # logger = create_logger(out_file_log)
    save_hparams(args, out_path)

    # build tokenizer:
    if args.tokenizer == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif args.tokenizer == "sst":
        dataset = load_from_disk("cache/sst/all_data")
        vocab_path = get_sst_tokenizer_vocab(args)
        tokenizer = SSTTokenizer(dataset, vocab_path=vocab_path)

    # load dataset
    sst_dataset = SSTDataset(tokenizer=tokenizer)
    train_set, val_set, test_set = sst_dataset.load_sst_dataset()
    train_set = sst_dataset.preprocess_dataset(train_set, label=args.label)
    val_set = sst_dataset.preprocess_dataset(val_set, label=args.label)
    train_set, train_dataloader = sst_dataset.prepare_data_for_torch(train_set, batch_size=args.bs)
    val_set, val_dataloader = sst_dataset.prepare_data_for_torch(val_set, batch_size=args.bs)
    print("len train set", len(train_set)) # 3610 samples
    print("len val set", len(val_set))  # 444 samples

    # Build model
    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=device)
    else:
        if args.model == "lstm":
            model = LSTMModel(num_tokens=sst_dataset.len_vocab,
                          emb_size=args.emb_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          p_drop=args.p_drop).to(device)
        elif args.model == "gpt2":
            model = GPT2FTModel(vocab_size=sst_dataset.len_vocab, device=device, init_weight=args.init_weight)

    # for the kl regularization:
    kl_model = GPT2FTModel(vocab_size=sst_dataset.len_vocab, device=device, init_weight=True)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # train parameters
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    PAD_IDX = sst_dataset.get_PAD_IDX()
    criterion = torch.nn.NLLLoss(ignore_index=PAD_IDX)

    if args.ep > 0:
        # train model
        train(model=model, train_generator=train_dataloader, val_generator=val_dataloader, criterion=criterion,
              optimizer=optimizer, device=device, out_path=out_path, EPOCHS=args.ep, KL_coeff=args.KL_coeff, kl_model=kl_model)

    prompts = ['The potato', 'The movie', 'I think that', 'I liked the movie.', 'I disliked badly the movie.']
    for prompt in prompts:
        # generate text post-training:
        generate_text_lm(model=model, tokenizer=sst_dataset.tokenizer, kl_model=kl_model, device=device, out_path=out_path, prompt=prompt)
