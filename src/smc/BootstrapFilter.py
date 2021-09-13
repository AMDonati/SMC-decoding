from smc.utils import resample
import torch
import torch.nn.functional as F

class BootstrapFilter:
    def __init__(self, num_particles, transition_model):
        self.num_particles = num_particles
        self.transition_model = transition_model

    def compute_filtering_weights(self, hidden, observations):
        '''
             # FORMULA
             # logw = -0.5 * mu_t ^ T * mu_t / sigma; sigma=scalar covariance.
             #  w = softmax(log_w)
             :param hidden: hidden state at timestep k: tensor of shape (B,num_particles,1,hidden_size)
             :param observations: current target element > shape (B,num_particles,1,F_y).
             :return:
             resampling weights of shape (B,P=num_particles).
        '''
        # get current prediction from hidden state.
        predictions = self.transition_model.predict_from_hidden(hidden) # (P,V) #TODO: check this function.
        #observations = observations.repeat(1, self.num_particles, 1)
        w = torch.gather(input=predictions, index=observations, dim=-1).squeeze() # (P)
        w = F.softmax(w, dim=0) # dim (P,1)
        return w

    def get_new_particle(self, observation, next_observation, hidden, weights, sigma=0.5, resampling=True):
        '''
        :param observation:
        :param next_observation:
        :param hidden:
        :param weights:
        :param sigma:
        :param resampling:
        :return:
        '''
        # '''
        # :param observation: current observation $Y_{k}$: tensor of shape (B, 1, input_size)
        # :param next_observation $Y_{k+1}$: tensor of shape (B, P, input_size)
        # :param hidden $\xi_k^l(h_k)$: current hidden state: tensor of shape (B, P, hidden_size)
        # :param $\w_{k-1}^l$: previous resampling weights: tensor of shape (B, P)
        # :return new_hidden state $\xi_{k+1}^l$: tensor of shape (B, P, hidden_size), new_weights $w_k^l$: shape (B,P).
        # '''
        if resampling:
            # Mutation: compute $I_t$ from $w_{t-1}$ and resample $h_{t-1}$ = \xi_{t-1}^l
            It = torch.multinomial(weights, num_samples=self.num_particles, replacement=True)
            resampled_h = resample(hidden, It)
        else:
            resampled_h = hidden
        # Selection : get $h_t$ = \xi_t^l
        #observation = observation.repeat(1, self.num_particles, 1)
        new_hidden, current_hidden = self.transition_model.get_new_hidden(hidden=resampled_h, observation=observation, sigma=sigma) #TODO: check this function. Ok add noise here.
        # compute $w_t$
        new_weights = self.compute_filtering_weights(hidden=new_hidden, observations=next_observation)
        return (new_hidden, current_hidden), new_weights