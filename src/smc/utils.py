import torch
import torch.nn.functional as F
import logging

def resample(params, I):
    '''
    :params: shape (B,P,H)
    :I: shape (B,P)
    '''
    I = I.unsqueeze(-1).repeat(1,1,params.size(-1))
    resampled_params = torch.gather(input=params, index=I, dim=1)
    return resampled_params

def resample_all_seq(params, i_t):
    '''
    :param params > shape (B,S,D)
    :param i_t: current indice > shape (P)
    '''
    I = i_t.view(i_t.size(0), 1, 1).repeat(1, params.size(-2), params.size(-1)) # shape (P,S,hidden_size)
    resampled_params = torch.gather(input=params, index=I, dim=0)
    return resampled_params

def create_logger(out_file_log):
    logging.basicConfig(filename=out_file_log, level=logging.INFO)
    # create logger
    logger = logging.getLogger('training log')
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger

def decreasing_noise_with_time(sigma, seq_len):
    times = [sigma/(t+1) for t in range(seq_len)]
    std_tensor = torch.tensor(times)
    return std_tensor

def sqrt_decreasing_noise_with_time(sigma, seq_len):
    times = [sigma/(t+1)**(1/2) for t in range(seq_len)]
    std_tensor = torch.tensor(times)
    return std_tensor

def constant_noise(sigma, seq_len):
    std_tensor = torch.tensor([sigma]*seq_len)
    return std_tensor


if __name__ == "__main__":

    # ---------- test of resample function -----------------------------------------------------------------------------------------------------------
    B = 2
    P = 4
    D = 2

    ind_matrix = torch.tensor([[1,1,2,3],[0,1,3,2]])

    K = torch.tensor([[[1,5],[2,6],[3,7],[4,8]],[[13,17], [14,18], [15,19], [16,20]]])

    truth = torch.tensor([[[2,6], [2,6], [3,7], [4,8]], [[13,17], [14,18], [16,20], [15,19]]])

    resampled_K = resample(K, ind_matrix)

    print("TRUTH:{}", truth)
    print("resampled_K:{}", resampled_K)

    # ok, test passed.

    # ----------------------- test of resample all past trajectories function ---------------------------------------------------------

    B = 2
    S = 3
    P = 4
    D = 1

    ind_matrix = torch.tensor([[[1, 1, 2, 2], [0, 0, 0, 0], [1, 1, 1, 0]],
                              [[0, 1, 3, 2], [3, 3, 2, 0], [1, 2, 3, 1]]])
    ind_matrix = ind_matrix.permute(0, 2, 1)
    # ind_matrix = tf.tile(tf.expand_dims(ind_matrix, axis=0), multiples=[B, 1, 1])  # (B,P,S)

    print('indices_matrices', ind_matrix[0, :, :].numpy())

    K = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    K = K.permute(0, 2, 1)
    K = K.unsqueeze(-1) # (B,P,S,D=1)
    print('init K', K[0, :, :, 0])

    truth_t0_b1 = torch.tensor([[2], [2], [3], [3]])
    truth_t1_b1 = torch.tensor([[2, 5], [2, 5], [2, 5], [2, 5]])
    truth_t2_b1 = torch.tensor([[2, 5, 10], [2, 5, 10], [2, 5, 10], [2, 5, 9]])
    truth_t0_b2 = torch.tensor([[13], [14], [16], [15]])
    truth_t1_b2 = torch.tensor([[15, 20], [15, 20], [16, 19], [13, 17]])
    truth_t2_b2 = torch.tensor([[15, 20, 22], [16, 19, 23], [13, 17, 24], [15, 20, 22]])

    truth_t0 = torch.stack([truth_t0_b1, truth_t0_b2])
    truth_t1 = torch.stack([truth_t1_b1, truth_t1_b2])
    truth_t2 = torch.stack([truth_t2_b1, truth_t2_b2])

    new_K = K
    for t in range(S):
        i_t = ind_matrix[:, :, t]
        Kt = resample_all_seq(params=new_K[:, :, :t + 1, :], i_t=i_t)
        if t < (S-1):
            new_K = torch.cat([Kt, K[:,:,t+1,:].unsqueeze(dim=-2)], dim=-2)
        print('new K at time_step {} for batch 0 {}:'.format(t, Kt[0, :, :, 0]))
        print('new K at time_step {} for batch 1 {}:'.format(t, Kt[1, :, :, 0]))

    # ------------- test of inv tanh x ----------------------------------------------------------------------
    X = torch.ones(size=(8,10,32))
    XX = F.tanh(inv_tanh(X))
    print(XX)