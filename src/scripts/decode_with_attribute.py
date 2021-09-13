from smc.PoorManSmoother import PoorManSmoothing
from smc.BootstrapFilter import BootstrapFilter
import torch
import argparse
from language_models.gpt2_finetune import GPT2FTModel
import os

def sample_new_observations(predictions, num=1, select='sampling'):
    if select == 'topk':
        values, indices = torch.topk(input=predictions, k=num, dim=-1)
    elif select == 'sampling':
        indices = torch.multinomial(predictions, num_samples=num)
        indices = indices.view(indices.size(-1), indices.size(0))
    return indices

def decode_with_attribute(prompt, model, out_folder, max_length=50, num_particles=10, num_iterations=10, num_trajectories=1, num_observations=1, select='sampling'):

    out_txt_file = os.path.join(out_folder, '{}_word_sequences.txt'.format(prompt)) #TODO, replace this by a logger.

    # init Bootstrap Filter & SMC Smoother
    bootstrap_filter = BootstrapFilter(num_particles=num_particles, transition_model=model)
    # Get init word sequences
    init_observations = model.generate_input_word_sequences(prompt=prompt, max_length=max_length)
    observations = init_observations

    seq_of_observations, seq_of_hidden = [init_observations], []

    for iter in range(num_iterations):
        smc_smoother = PoorManSmoothing(bootstrap_filter=bootstrap_filter, observations=observations, out_folder=out_folder)
        _, _ = smc_smoother.run_PMS()
        # select trajectories
        selected_trajectories, _ = smc_smoother.select_trajectories(num=num_trajectories, select=select)
        seq_of_hidden.append(selected_trajectories)
        # sample new observations from trajectories
        _, predictions = model.predict_from_hidden(selected_trajectories) # shape (num_traj, S, V)
        new_observations = []
        for traj in range(num_trajectories):
            new_observation = sample_new_observations(predictions=predictions[traj], num=num_observations, select=select)
            new_observations.append(new_observation)
        new_observations = torch.stack(new_observations, dim=0) # shape (num_trajectories, num_observations, S)
        decoded_observations = decode_and_write_observations(observations=new_observations, tokenizer=model.tokenizer, txt_file=out_txt_file, num_trajectories=num_trajectories, num_observations=num_observations)
        seq_of_observations.append(new_observations)
        observations = new_observations[0,0,:].unsqueeze(-1) #TODO: change this to get the argmax of the traj & observations ?
        observations = observations.unsqueeze(0)
    return seq_of_hidden, seq_of_observations, decoded_observations


def decode_and_write_observations(observations, tokenizer, txt_file, num_trajectories, num_observations):
    decoded_observations = []
    with open(txt_file, 'w') as f:
        for traj in range(num_trajectories):
            f.write("Generated sequences for hidden state #{}:".format(traj+1) + "\n" + "\n")
            for obs in range(num_observations):
                decoded_observation = tokenizer.decode(observations[traj, obs].squeeze(), skip_special_tokens=True)
                f.write("SEQ {}:".format(obs+1) + decoded_observation + "\n")
                decoded_observations.append(decoded_observation)
        f.close()
    return decoded_observations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-out_path", type=str, default="output/temp/sst_decoding", help="out path")
    parser.add_argument("-model_path", type=str, help="path for the pretrained attribute model")
    parser.add_argument("-max_length", type=int, default=50, help='length maximal for word sequence')
    parser.add_argument("-num_particles", type=int, default=50, help='number of particles for the smc algo.')
    parser.add_argument("-num_trajectories", type=int, default=5, help='number of trajectories to display for the smc algo.')
    parser.add_argument("-num_observations", type=int, default=5, help='number of observations to display for the smc algo.')
    parser.add_argument("-num_iterations", type=int, default=10, help='number of iterations for the decoding algo.')
    parser.add_argument("-select", type=str, default='sampling', help='selection method for the hidden states & observations.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=device)
    else:
        model = GPT2FTModel(vocab_size=50257, device=device)

    out_folder = os.path.join(args.out_path, "{}particles_{}iter_{}".format(args.num_particles, args.num_iterations, args.select))
    if args.model_path is not None:
        out_folder = out_folder + '_random'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    prompts = ["The movie is"]
    #prompts = ["The movie is", "I disliked the movie", "I liked the movie.", "The potato", "This man is very ugly.", "This man is awesome."]
    for prompt in prompts:
        seq_of_hidden, seq_of_observations, decoded_observations = decode_with_attribute(prompt=prompt, model=model, out_folder=args.out_path, max_length=50, num_particles=10, num_iterations=10, num_trajectories=3, num_observations=2, select='sampling')
    print("done")


