from smc.PoorManSmoother import PoorManSmoothing
from smc.BootstrapFilter import BootstrapFilter
from smc.utils import create_logger, decreasing_noise_with_time, constant_noise, sqrt_decreasing_noise_with_time
import torch
import argparse
from language_models.gpt2_finetune import GPT2FTModel
import os
import numpy as np
import json


def save_hparams(args, out_folder):
    dict_hparams = vars(args)
    dict_hparams = {key: str(value) for key, value in dict_hparams.items()}
    config_path = os.path.join(out_folder, "config.json")
    with open(config_path, 'w') as fp:
        json.dump(dict_hparams, fp, sort_keys=True, indent=4)


def sample_new_observations(predictions, num=1, select='sampling', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if select == 'topk':
        values, indices = torch.topk(input=predictions, k=num, dim=-1)
    elif select == 'sampling':
        indices = torch.multinomial(predictions, num_samples=num)
        indices = indices.view(indices.size(-1), indices.size(0))
    return indices


def decode_with_attribute(prompt, model, out_folder, sigma, noise_function, max_length=50, num_particles=10,
                          num_iterations=10, num_trajectories=1, num_observations=1, select='sampling', seed=None):
    out_file_log = os.path.join(out_folder, '{}_word_sequences.log'.format(prompt))
    logger = create_logger(out_file_log)

    # init Bootstrap Filter & SMC Smoother
    bootstrap_filter = BootstrapFilter(num_particles=num_particles, transition_model=model, sigma=sigma,
                                       noise_function=noise_function)
    # Get init word sequences
    init_observations = model.generate_input_word_sequences(prompt=prompt, max_length=max_length, seed=seed)
    observations = init_observations

    seq_of_observations, seq_of_hidden, seq_of_decoded_observations = [init_observations], [], [model.tokenizer.decode(init_observations.squeeze(), skip_special_tokens=True)]
    seq_of_best_hidden, seq_of_best_observations, seq_of_best_hidden_norm = [], [], []
    logger.info('INIT OBSERVATION:')
    logger.info(model.tokenizer.decode(init_observations.squeeze(), skip_special_tokens=True))

    for iter in range(num_iterations):
        logger.info(
            '-----------------------------------------iter #{} --------------------------------------------------'.format(
                iter))
        smc_smoother = PoorManSmoothing(bootstrap_filter=bootstrap_filter, observations=observations,
                                        out_folder=out_folder)
        _, _ = smc_smoother.run_PMS()
        # select trajectories
        selected_trajectories, _ = smc_smoother.select_trajectories(num=num_trajectories, select=select, seed=seed)
        seq_of_hidden.append(selected_trajectories)
        # sample new observations from trajectories
        _, predictions = model.predict_from_hidden(selected_trajectories)  # shape (num_traj, S, V)
        new_observations = []
        for traj in range(num_trajectories):
            new_observation = sample_new_observations(predictions=predictions[traj], num=num_observations,
                                                      select=select, seed=seed)
            new_observations.append(new_observation)
        new_observations = torch.stack(new_observations, dim=0)  # shape (num_trajectories, num_observations, S)
        decoded_observations = decode_and_write_observations(observations=new_observations, trajectories=selected_trajectories, tokenizer=model.tokenizer,
                                                             logger=logger, num_trajectories=num_trajectories,
                                                             num_observations=num_observations)
        seq_of_observations.append(new_observations)
        seq_of_decoded_observations.append(decoded_observations)
        seq_of_best_hidden.append(selected_trajectories[0].cpu().numpy())
        seq_of_best_hidden_norm.append(np.array([round(torch.linalg.norm(selected_trajectories[0,t]).cpu().item(),1) for t in range(selected_trajectories[0].shape[0])]))
        seq_of_best_observations.append(new_observations[0, 0, :].cpu().numpy())
        observations = new_observations[0, 0, :].unsqueeze(
            -1)  # TODO: change this to get the argmax of the traj & observations ?
        observations = observations.unsqueeze(0)
    logger.info(
        '---------------------------------------------------------------------------------------------------------------')
    np.save(os.path.join(out_folder, "seq_of_best_observations.npy"), np.stack(seq_of_best_observations))
    np.save(os.path.join(out_folder, "seq_of_best_hidden.npy"), np.stack(seq_of_best_hidden))
    np.save(os.path.join(out_folder, "seq_of_best_hidden_norm.npy"), np.stack(seq_of_best_hidden_norm))
    return seq_of_hidden, seq_of_observations, seq_of_decoded_observations


def decode_and_write_observations(observations, trajectories, tokenizer, logger, num_trajectories, num_observations):
    decoded_observations = []
    for traj in range(num_trajectories):
        logger.info("---------------Generated sequences for hidden state #{}-----------------".format(traj + 1) + "\n")
        current_traj = trajectories[traj]
        seq_norm = [round(torch.linalg.norm(current_traj[t]).item(),1) for t in range(current_traj.shape[0])]
        logger.info("hidden states norms:{}".format(seq_norm))
        for obs in range(num_observations):
            decoded_observation = tokenizer.decode(observations[traj, obs].squeeze(), skip_special_tokens=True)
            logger.info("SEQ {}:".format(obs + 1) + decoded_observation + "\n")
            decoded_observations.append(decoded_observation)
        logger.info('-' * 50)
    return decoded_observations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-out_path", type=str, default="output/temp/sst_decoding", help="out path")
    parser.add_argument("-model_path", type=str, default='output/sst_attribute_model/best_models/gpt2_ft/1/model.pt', help="path for the pretrained attribute model")
    parser.add_argument("-max_length", type=int, default=50, help='length maximal for word sequence')
    parser.add_argument("-num_particles", type=int, default=50, help='number of particles for the smc algo.')
    parser.add_argument("-num_trajectories", type=int, default=5,
                        help='number of trajectories to display for the smc algo.')
    parser.add_argument("-num_observations", type=int, default=5,
                        help='number of observations to display for the smc algo.')
    parser.add_argument("-num_iterations", type=int, default=10, help='number of iterations for the decoding algo.')
    parser.add_argument("-select", type=str, default='sampling',
                        help='selection method for the hidden states & observations.')
    parser.add_argument("-std", type=int, default=0.5,
                        help='sigma constant for the noise added to the transition model.')
    parser.add_argument("-noise_function", type=str, default="constant",
                        help='sigma function for the noise added to the transition model.')
    parser.add_argument("-seed", type=str, default="123",
                        help='sigma function for the noise added to the transition model.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=device)
    else:
        model = GPT2FTModel(vocab_size=50257, device=device)

    out_folder = os.path.join(args.out_path,
                              "{}particles_{}iter_{}_std{}_noisefn-{}".format(args.num_particles, args.num_iterations, args.select, args.std, args.noise_function))
    if args.model_path is None:
        out_folder = out_folder + '_random'
    out_folder = os.path.join(out_folder, "seed_{}".format(args.seed))
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    save_hparams(args=args, out_folder=out_folder)

    if args.noise_function == 'constant':
        noise_function = constant_noise
    elif args.noise_function == 'decreasing':
        noise_function = decreasing_noise_with_time
    elif args.noise_function == 'sqrt_decreasing':
        noise_function = sqrt_decreasing_noise_with_time

    #prompts = ["The movie is"]
    prompts = ["The movie is", "I disliked the movie", "I liked the movie.", "The potato", "This man is very ugly.", "This man is awesome."]
    for prompt in prompts:
        out_file_log = '{}_word_sequences.log'.format(prompt)
        seq_of_hidden, seq_of_observations, seq_decoded_observations = decode_with_attribute(prompt=prompt, model=model,
                                                                                         out_folder=out_folder,
                                                                                         sigma=args.std,
                                                                                         noise_function=noise_function,
                                                                                         max_length=args.max_length,
                                                                                         num_particles=args.num_particles,
                                                                                         num_iterations=args.num_iterations,
                                                                                         num_trajectories=args.num_trajectories,
                                                                                         num_observations=args.num_observations,
                                                                                         select=args.select,
                                                                                         seed=args.seed)
    print("done")
