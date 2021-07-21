import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch.nn.functional as F
from old_scripts.score_functions.topic_BoW import topic_BoW_function, read_topic_BoW
from old_scripts.smc.utils import resample_all_seq
import os



def smc_decoding(prompt,  num_sequences=20, max_length=50, score_function=topic_BoW_function, model=GPT2LMHeadModel.from_pretrained('gpt2'), decoding_mode='sampling', bow_tokens=None, resample=True, start_sample_step=5, sample_step=1):
    # INITIALIZATION:
    #TODO: add a torch witn no grad here ?
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.repeat(num_sequences, 1)
    indices_matrix = []
    sequences_genealogy = []
    for t in range(max_length):
        inputs = generate_next_word(inputs=inputs, model=model, decoding=decoding_mode)
        print(tokenizer.decode(inputs[0], skip_special_tokens=True))
        scores = score_function(inputs=inputs, bow_tokens=bow_tokens, model=model)
        # RESAMPLE PAST SEQUENCES:
        # get indices from normalized scores:
        normalized_scores = F.softmax(scores, dim=0)
        i_t = torch.multinomial(normalized_scores.squeeze(), num_samples=num_sequences, replacement=True)
        indices_matrix.append(i_t.cpu().squeeze())
        if resample and t>= start_sample_step and t % sample_step == 0:
            inputs = resample_all_seq(inputs, i_t=i_t)
        sequences_genealogy.append([tokenizer.decode(inputs[s], skip_special_tokens=True) for s in range(num_sequences)])
    return inputs, sequences_genealogy[-1], sequences_genealogy, indices_matrix

def generate_next_word(inputs, model, decoding='sampling'):
    if decoding == 'sampling':
        output = model.generate(
            inputs,
            do_sample=True,
            max_length=1,
            top_k=0)

    elif decoding == 'temp-sampling':
        output = model.generate(
            inputs,
            do_sample=True,
            max_length=1,
            top_k=0,
            temperature=0.7
        )

    elif decoding == 'bs':
        output = model.generate(
            inputs,
            max_length=1,
            num_beams=5,
            early_stopping=True
        )

    elif decoding == 'top-k':
        output = model.generate(
            inputs,
            do_sample=True,
            max_length=1,
            top_k=50
        )

    elif decoding == 'top-p':
        output = model.generate(
            inputs,
            do_sample=True,
            max_length=50,
            top_p=0.92,
            top_k=0
        )
    return output

def save_generated_text(prompt, generated_sequence, topic, out_file):
    generated_sequence = '\n'.join(generated_sequence)
    with open(out_file, 'w') as f:
        f.write(topic + "\n" + prompt + "\n" + "\n" + generated_sequence)
        f.close()


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # 345B parameters GPT-2 model.
    model = GPT2Model.from_pretrained('gpt2')

    topic = 'science'
    bow_tokens = read_topic_BoW(topic=topic)

    out_path = "../../../output/temp"
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    inputs = tokenizer("The potato", return_tensors="pt")
    print("-"*50 + "The potato" + "-"*50)
    for i in range(4):
        torch.manual_seed(i)
        outputs, decoded_sequences, sequences_genealogy, indices_matrix = smc_decoding(prompt="The potato", resample=True)
        for s in decoded_sequences:
            print('-'*100)
            print(s)
        out_file=os.path.join(out_path, "{}__The_potato__generated_sequences_s{}.txt".format(topic,i))
        save_generated_text(prompt="The potato", generated_sequence=decoded_sequences, topic=topic, out_file=out_file)
    print('-' * 100)
    print("-" * 50 + "The scientist" + "-" * 50)
    for i in range(4):
        torch.manual_seed(i)
        outputs, decoded_sequences, sequences_genealogy, indices_matrix = smc_decoding(prompt="The scientist", resample=True)
        for s in decoded_sequences:
            print('-'*100)
            print(s)
            out_file = os.path.join(out_path, "{}__The_scientist__generated_sequences_s{}.txt".format(topic, i))
            save_generated_text(prompt="The scientist", generated_sequence=decoded_sequences, topic=topic,
                                out_file=out_file)