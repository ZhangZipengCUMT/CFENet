import numpy as np
import torch
import random

def one_hot_encoding(X, n_values=4):
    X = [int(x) for x in X]
    # n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def DataTransform(sample, config):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # weak_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug

# def DataTransform_TD(sample, config):
#     """Weak and strong augmentations"""
#     weak_aug = sample
#     strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio) #masking(sample)
#     return weak_aug, strong_aug
#
# def DataTransform_FD(sample, config):
#     """Weak and strong augmentations in Frequency domain """
#     # weak_aug =  remove_frequency(sample, 0.1)
#     strong_aug = add_frequency(sample, 0.1)
#     return weak_aug, strong_aug
def DataTransform_TD(sample, jitter_ratio):
    """Simplely use the jittering augmentation. Feel free to add more autmentations you want,
    but we noticed that in TF-C framework, the augmentation has litter impact on the final tranfering performance."""
    aug = jitter(sample, jitter_ratio)
    return aug


def DataTransform_TD_bank(sample, jitter_ratio, jitter_scale_ratio, max_seg):
    """Augmentation bank that includes four augmentations and randomly select one as the positive sample.
    You may use this one the replace the above DataTransform_TD function."""
    aug_1 = jitter(sample, jitter_ratio)
    aug_2 = scaling(sample, jitter_scale_ratio)
    aug_3 = permutation(sample, max_segments=max_seg)
    aug_4 = masking(sample, keepratio=0.9)

    li = np.random.randint(0, 4, size=[1])
    li_onehot = one_hot_encoding(li)
    aug_1 = aug_1 * li_onehot[:, 0]#[:, None, None]  # the rows that are not selected are set as zero.
    aug_2 = aug_2 * li_onehot[:, 1]#[:, None, None]
    aug_3 = aug_3 * li_onehot[:, 2]#[:, None, None]
    aug_4 = aug_4 * li_onehot[:, 3]#[:, None, None]
    aug_T = aug_1 + aug_2 + aug_3 + aug_4
    return aug_T

def DataTransform_FD(sample):
    """Weak and strong augmentations in Frequency domain """
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    return aug_F

def remove_frequency(x, pertub_ratio=0.0):
    mask = np.random.uniform(0, 1, x.shape) > pertub_ratio
    #torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    # mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0.0):
    mask = np.random.uniform(0, 1, x.shape) > (1 - pertub_ratio)
    #torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    # mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = np.random.random(x.shape)*(max_amplitude*0.1)
    #torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix


def generate_binomial_mask(L, p=0.5): # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(L))).to(torch.bool)


def masking(x, keepratio=0.9, mask= 'binomial'):
    global mask_id
    nan_mask = ~np.isnan(x).any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.shape[0], p=keepratio)#.to(x.device)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0]))
    ai = []
    # for i in range(x.shape[1]):
    #     xi = x[:, i, :]
    x = np.multiply(x, factor)
    return x #np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[0])

    num_segs = np.random.randint(1, max_segments)

    ret = np.zeros_like(x)
    # for i, pat in enumerate(x):
    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(x.shape[0] - 2, num_segs - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)
        warp = np.concatenate(np.random.permutation(splits)).ravel()
        ret = x[warp]
    else:
        ret = x
    return ret  # torch.from_numpy(ret)

def spec_augment(mel_spectrogram, max_f_num=2, max_t_num=3,
                 freq_mask_rate=0.3, time_mask_rate=0.05,
                 random_move=False, max_move_num=3):
    freq_mask_num = np.random.randint(0, max_f_num + 1)
    time_mask_num = np.random.randint(0, max_t_num + 1)
    move_num = np.random.randint(0, max_move_num)
    t_num = mel_spectrogram.shape[0]
    f_num = mel_spectrogram.shape[1]
    warped_mel_spectrogram = mel_spectrogram
    freq_mask_max_width = int(f_num * freq_mask_rate) // 2
    time_mask_max_width = int(t_num * time_mask_rate) // 2
    if freq_mask_num > 0:
        for _ in range(freq_mask_num):
            f_c = np.random.randint(0, f_num)
            f_w_ = np.random.randint(1, freq_mask_max_width)
            warped_mel_spectrogram[:, max(f_c - f_w_, 0):min(f_c + f_w_, f_num)] = 0
    if time_mask_num > 0:
        for _ in range(time_mask_num):
            t_c = np.random.randint(0, t_num)
            t_w_ = np.random.randint(1, time_mask_max_width)
            warped_mel_spectrogram[max(t_c - t_w_, 0):min(t_c + t_w_, t_num), :] = 0
    if random_move:
        for _ in range(move_num):
            f_w_ = np.random.randint(1, freq_mask_max_width)
            t_w_ = np.random.randint(1, time_mask_max_width)
            f_c_s = np.random.randint(f_w_, f_num - f_w_)
            t_c_s = np.random.randint(t_w_, t_num - t_w_)
            f_c_t = np.random.randint(f_w_, f_num - f_w_)
            t_c_t = np.random.randint(t_w_, t_num - t_w_)
            warped_mel_spectrogram[(t_c_t - t_w_):(t_c_t + t_w_), (f_c_t - f_w_):(f_c_t + f_w_)] = \
                mel_spectrogram[(t_c_s - t_w_):(t_c_s + t_w_), (f_c_s - f_w_):(f_c_s + f_w_)]
    return warped_mel_spectrogram
