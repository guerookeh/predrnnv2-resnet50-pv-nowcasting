from args import configs
import math
import numpy as np

def reserve_schedule_sampling_exp(itr):
    if itr < configs.r_sampling_step_1:
        r_eta = 0.5
    elif itr < configs.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - configs.r_sampling_step_1) / configs.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < configs.r_sampling_step_1:
        eta = 0.5
    elif itr < configs.r_sampling_step_2:
        eta = 0.5 - (0.5 / (configs.r_sampling_step_2 - configs.r_sampling_step_1)) * (itr - configs.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (configs.batch_size, configs.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (configs.batch_size, configs.total_length - configs.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((configs.img_width // configs.patch_size,
                    configs.img_width // configs.patch_size,
                    configs.patch_size ** 2 * configs.img_channel))
    zeros = np.zeros((configs.img_width // configs.patch_size,
                      configs.img_width // configs.patch_size,
                      configs.patch_size ** 2 * configs.img_channel))

    real_input_flag = []
    for i in range(configs.batch_size):
        for j in range(configs.total_length - 2):
            if j < configs.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (configs.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (configs.batch_size,
                                  configs.total_length - 2,
                                  configs.img_width // configs.patch_size,
                                  configs.img_width // configs.patch_size,
                                  configs.patch_size ** 2 * configs.img_channel))
    return real_input_flag


def schedule_sampling(eta, itr):
    zeros = np.zeros((configs.batch_size,
                      configs.total_length - configs.input_length - 1,
                      configs.img_width // configs.patch_size,
                      configs.img_width // configs.patch_size,
                      configs.patch_size ** 2 * configs.img_channel))
    if not configs.scheduled_sampling:
        return 0.0, zeros

    if itr < configs.sampling_stop_iter:
        eta -= configs.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (configs.batch_size, configs.total_length - configs.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((configs.img_width // configs.patch_size,
                    configs.img_width // configs.patch_size,
                    configs.patch_size ** 2 * configs.img_channel))
    zeros = np.zeros((configs.img_width // configs.patch_size,
                      configs.img_width // configs.patch_size,
                      configs.patch_size ** 2 * configs.img_channel))
    real_input_flag = []
    for i in range(configs.batch_size):
        for j in range(configs.total_length - configs.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (configs.batch_size,
                                  configs.total_length - configs.input_length - 1,
                                  configs.img_width // configs.patch_size,
                                  configs.img_width // configs.patch_size,
                                  configs.patch_size ** 2 * configs.img_channel))
    return eta, real_input_flag