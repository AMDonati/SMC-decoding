import torch
from smc.BootstrapFilter import BootstrapFilter
from smc.utils import resample, resample_all_seq, create_logger
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import torch.nn.functional as F


class SmoothingAlgo:
    def __init__(self, bootstrap_filter, observations, prompt="The", out_folder=None,
                logger=None, init_prompt=None):
        # '''
        # :param bootstrap_filter: Class Implementing a Bootstrap filter algorithm.
        # :param observations: sequence of observations generated by a stochastic RNN: tensor of shape (num_samples=B, num_particles, seq_len, output_size)
        # :param states: sequence of hidden states generated by a stochastic RNN: tensor of shape (num_samples, num_particles, seq_len, hidden_size)
        # :param backward_samples: number of backward_samples for the Backward IS Smoothing algo.
        # :param estimation_function: Fonction to estimate: in our case $mathbb[E][X_0|Y_{0:n}]$
        # '''
        self.bootstrap_filter = bootstrap_filter
        self.transition_model = bootstrap_filter.transition_model
        self.device = self.transition_model.device
        self.num_particles = self.bootstrap_filter.num_particles
        self.sigma = self.bootstrap_filter.sigma
        self.noise_function = self.bootstrap_filter.noise_function
        self.observations = observations.repeat(self.num_particles, 1, 1)  # Tensor of shape (particles, seq_len, 1)
        self.seq_len = self.observations.size(-2)
        if logger is None:
            self.logger = self.create_logger(out_folder)
        else:
            self.logger = logger
        self.init_prompt = init_prompt
        if self.init_prompt is None:
            self.encoded_prompt = self.transition_model.tokenizer.encode(prompt, return_tensors="pt").squeeze().to(self.device)
        else:
            prompt_ = "." + prompt
            self.encoded_prompt = self.transition_model.tokenizer.encode(prompt_, return_tensors="pt").squeeze().to(self.device)

    def create_logger(self, out_folder):
        out_file_log = os.path.join(out_folder, 'debug_log.log')
        logger = create_logger(out_file_log=out_file_log)
        return logger

    def init_particles(self, hidden_size=768):
        #self.trajectories = self.transition_model.get_hidden_from_input(input=self.observations, sigma=self.sigma, noise_function=self.noise_function)
        #self.trajectories = init_trajectories.repeat(self.num_particles, 1, 1) # shape (P,S,hidden_size)
        # self.ancestors = torch.normal(
        #     mean=torch.zeros(self.states.size(0), self.num_particles, self.states.size(-1)),
        #     std=sigma_init ** (1 / 2) * torch.ones(self.states.size(0), self.num_particles,
        #                                            self.states.size(-1)))
        #trajectories = torch.zeros(size=(self.observations.size(0), 1, hidden_size)).to(self.device) #TODO: initialize instead with hidden state from prompt.
        #std_tensor = self.noise_function(sigma=self.sigma, seq_len=trajectories.size(1))
        #self.trajectories = self.transition_model.add_noise(params=trajectories, std_tensor=std_tensor)
        if self.init_prompt is not None:
            input_prompt = torch.cat(self.init_prompt, self.encoded_prompt)
        else:
            input_prompt = self.encoded_prompt
        trajectories = input_prompt.view(1,input_prompt.shape[0], 1).repeat(self.num_particles, 1, 1)

        self.trajectories = self.transition_model.get_hidden_from_input(input=trajectories, sigma=self.sigma, noise_function=self.noise_function) # (P,1,hidden_size)
        #self.ancestors = self.trajectories[:,0,:].unsqueeze(1) # shape (P,1,hidden_size)
        self.filtering_weights = self.bootstrap_filter.compute_filtering_weights(hidden=self.trajectories,
                                                                                 observations=self.observations[:, 0, :])  # decide if take $Y_0 of $Y_1$

    def plot_trajectories_pms(self, trajectories, out_folder):
        trajectories = trajectories.squeeze().cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        x = np.linspace(1, trajectories.shape[1], trajectories.shape[1])
        num_part = [np.unique(trajectories[:, k, :], axis=0).shape[0] for k in range(trajectories.shape[1])]
        for p in range(trajectories.shape[0]):
            label1 = "trajectory for  dim 0" if p == 0 else None
            ax1.scatter(x, trajectories[p,:,0], label=label1, s=7)
        ax2.plot(x, num_part, label='number of unique particles')
        ax1.legend(loc='upper center')
        ax2.legend(loc='upper center')
        out_file = "pms_trajectories_{}particles".format(self.num_particles)
        fig.savefig(os.path.join(out_folder, out_file))
        plt.close()


class PoorManSmoothing(SmoothingAlgo):
    def __init__(self, bootstrap_filter, observations, out_folder, logger=None, init_prompt=None, prompt="The"):
        # '''
        # :param bootstrap_filter: Class Implementing a Bootstrap filter algorithm.
        # :param observations: sequence of observations generated by a stochastic NN: tensor of shape (1, num_particles, seq_len, V)
        # '''
        super(PoorManSmoothing, self).__init__(bootstrap_filter=bootstrap_filter, observations=observations,
                                               out_folder=out_folder, logger=logger, init_prompt=init_prompt, prompt=prompt)

    def get_observation_sequence_from_probas(self, particles_probas):
        tokens_oneparticle = torch.argmax(particles_probas[0], dim=-1)
        tokens_oneparticle_sampling = torch.multinomial(particles_probas[0], num_samples=1).squeeze()
        words_oneparticle = self.transition_model.tokenizer.decode(tokens_oneparticle, skip_special_tokens=True)
        words_oneparticle_sampling = self.transition_model.tokenizer.decode(tokens_oneparticle_sampling,
                                                                            skip_special_tokens=True)

        return words_oneparticle, words_oneparticle_sampling


    def check_trajectories(self):
        # Selection: resample all past trajectories with current indice i_t
        print("length of trajectories:", self.trajectories.size(1))
        _, particles_probas = self.transition_model.predict_from_hidden(hidden=self.trajectories)
        words_oneparticle, words_oneparticle_sampling = self.get_observation_sequence_from_probas(particles_probas)
        predictions2 = self.transition_model.model(
            input_ids=self.encoded_prompt.view(1, self.encoded_prompt.shape[0], 1).repeat(self.num_particles, 1, 1))
        particles_logits2 = predictions2.logits.squeeze()
        particles_probas2 = F.softmax(particles_logits2, dim=-1)
        words_oneparticle2, words_oneparticle_sampling2 = self.get_observation_sequence_from_probas(
            particles_probas2)
        print(
            " ------------------------------------current trajectories output words ------------------------------------------ ")
        print("GREEDY")
        print(words_oneparticle)
        print("SAMPLING")
        print(words_oneparticle_sampling)
        print(
            "------------------------------------------------------------------------------------------------------------------")

    def run_PMS(self):
        with torch.no_grad():
            self.init_particles()
            # for loop on time
            indices_matrix, particles_seq = [], []
            particles_seq.append(self.trajectories)
            for k in range(self.seq_len - 1): #TODO: check if we go until seq_len or seq_len - 1.
                self.check_trajectories()
                self.old_filtering_weights = self.filtering_weights
                i_t = torch.multinomial(self.old_filtering_weights, num_samples=self.num_particles, replacement=True)
                indices_matrix.append(i_t.cpu().squeeze())
                resampled_trajectories = resample_all_seq(self.trajectories, i_t=i_t) #TODO: pad with zeros here? cf SMC-T code.
                # Mutation: Run bootstrap filter at time k to get new particle without resampling
                (self.trajectories, self.particles), self.filtering_weights = self.bootstrap_filter.get_new_particle(
                    observation=self.observations[:, k, :], next_observation=self.observations[:, k + 1, :],
                 hidden=resampled_trajectories, weights=self.old_filtering_weights, resampling=False) #TODO: add sigma here.
                # append resampled trajectories to new particle
                #self.trajectories = torch.cat([resampled_trajectories, self.particles.unsqueeze(-2)], dim=-2) #TODO: useless ?
            indices_matrix = torch.stack(indices_matrix, dim=0) # (seq_len, P)
            particles_seq = torch.stack(particles_seq, dim=0)
            return particles_seq.squeeze().cpu().numpy(), indices_matrix.cpu().numpy()

    def get_genealogy(self, indices_matrix):
        n_particles = indices_matrix.shape[-1]
        n_times = indices_matrix.shape[0]
        particle_indices = np.arange(n_particles, dtype=int)
        # Array contanant la genealogie
        # La genealogie est un array n_times * n_particles
        # A chaque ligne on a l'indice de particule en lequel passe la trajectoire
        # Au debut tout le monde passe par sa particule associÃ©e.
        genealogy = np.repeat([particle_indices], n_times+1, axis=0)
        # Maintenant on actualise
        for t in range(0, n_times):
            old_genealogy = genealogy  # A chaque debut, on stocke la "vieille genealogie"
            # Ici, pour l'exemple, un resampling uniforme
            indice_resampling = indices_matrix[t]
            # Maintenant, pour chaque colonne, la colonne entiere est remplacee par l'ancienne associÃ©e Ã  la particule
            genealogy = old_genealogy[:, indice_resampling]
            # Attention, Ã  chaque fois on restipule bien qu'au temps final, on passe par le bon indice de particule
            genealogy[t + 1:, :] = particle_indices
        return genealogy

    def resample_trajectories(self, trajectories, genealogy):
        n_particles = genealogy.shape[-1]
        n_times = trajectories.shape[0]
        num_dim = trajectories.shape[-1]
        resampled_trajectories = np.zeros(shape=(n_times, n_particles, num_dim))
        for t in reversed(range(n_times)):
            resampled_trajectories[t, :, :] = trajectories[t, genealogy[t, :], :] # (seq_len, P, hidden_size)
        return np.transpose(resampled_trajectories, axes=[1,0,2])

    def select_trajectories(self, num=1, select='topk', seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        if select == 'topk':
            values, indices = torch.topk(input=self.filtering_weights, k=num)
        elif select == 'sampling':
            indices = torch.multinomial(self.filtering_weights, num_samples=num)
        indices = indices.view(indices.shape[0],1,1).repeat(1, self.trajectories.shape[1], self.trajectories.shape[-1])
        selected_traj = torch.gather(input=self.trajectories, index=indices, dim=0)  # shape (num, S, hidden_size)
        return selected_traj, indices




if __name__ == '__main__':
    from language_models.gpt2_finetune import GPT2FTModel
    from smc.BootstrapFilter import BootstrapFilter
    import os
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch

    # Different types of classic decoding:
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # 345B parameters GPT-2 model.
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=gpt_tokenizer.eos_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2FTModel(vocab_size=50256, device=device)
    out_folder = os.path.join("output", "temp", "pms")
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Get an observation
    prompt = "The movie is"
    inputs = gpt_tokenizer.encode(prompt, return_tensors="pt")

    # Sampling decoding.
    sample_output = gpt_model.generate(
        inputs,
        do_sample=True,
        max_length=50,
        top_k=0
    )
    print("Output:\n" + 100 * '-')
    print(gpt_tokenizer.decode(sample_output[0], skip_special_tokens=True))

    # Build the bootstrap filter:
    bootstrap_filter = BootstrapFilter(transition_model=model, num_particles=10)
    pms_smoother = PoorManSmoothing(bootstrap_filter=bootstrap_filter, observations=sample_output.unsqueeze(-1), out_folder=out_folder)
    particles_seq, genealogy_indices = pms_smoother.run_PMS()
    print("done")