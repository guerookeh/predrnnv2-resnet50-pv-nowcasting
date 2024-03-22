# =====================================
# INSTALL REQUIRED PACKAGES
# =====================================

requirements = [
    "numpy",
    "torch",
    "torchvision",
    "matplotlib",
    "xarray",
    "ipykernel",
    "gcsfs",
    "fsspec",
    "dask",
    "cartopy",
    "ocf-blosc2",
    "torchinfo",
    "pyarrow"
]

for package in requirements:
    os.system(f"pip install {package}")

# =====================================
# IMPORTS
# =====================================

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as Adam
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import math
import xarray as xr
from datetime import datetime, timedelta, time
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

# =====================================
# CONFIGURATION CLASS
# =====================================

# Model Configuration
class Config:
    def __init__(self, hrv_path, year):
        # Data and File Paths
        self.hrv_path = hrv_path
        self.year = year
        self.sites_path = ''
        self.pretrained_model = ''
        self.save_dir = ''
        
        # Training Parameters
        self.max_iterations = 500
        self.is_training = 1
        self.batch_size = 1
        self.lr = 0.001
        self.display_interval = 10
        self.test_interval = round(0.0625 * self.max_iterations)
        self.snapshot_interval = 50
        self.num_save_samples = 10
        
        # Model Parameters
        self.model_name = 'predrnn_v2'
        self.num_hidden = '64,64,64,64'
        self.filter_size = 5
        self.stride = 1
        self.patch_size = 1
        self.layer_norm = 1
        self.decouple_beta = 0.1
        self.input_length = 12
        self.total_length = 60
        self.img_width = 128
        self.img_channel = 1
        self.injection_action = 'concat'
        self.conv_on_input = 0
        self.res_on_conv = 0
        self.num_action_ch = 4

        # Scheduled Sampling Parameters
        self.reverse_scheduled_sampling = 1
        self.r_sampling_step_1 = round(0.3125 * self.max_iterations)
        self.r_sampling_step_2 = round(0.625 * self.max_iterations)
        self.r_exp_alpha = round(0.0625 * self.max_iterations)
        self.scheduled_sampling = 1
        self.sampling_stop_iter = round(0.625 * self.max_iterations)
        self.sampling_start_value = 1.0
        self.sampling_changing_rate = 0.00002
        self.reverse_input = 1

        # Device Configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = 1

        # Visualization Parameters
        self.visual = 0
        self.visual_path = './decoupling_visual'

        # Training Date Range
        self.train_min_date = datetime(2020, 7, 1)
        self.train_max_date = datetime(2020, 7, 28)

configs = Config()

# =====================================
# SITE LOCATIONS
# =====================================

# Site Locations JSON
with open(configs.sites_path) as f:
    site_locations = {
        data_source: {
            int(site): (int(location[0]), int(location[1]))
            for site, location in locations.items()
        }
        for data_source, locations in json.load(f).items()
    }

# =====================================
# DATA TRANSFORMS
# =====================================

# Random Seeds
torch.manual_seed(7)
np.random.seed(7)
random.seed(7)

# Transforms
MEAN_PIXEL = 0.35
STD_PIXEL = 0.01
tsfm = transforms.Compose([
    transforms.Normalize(MEAN_PIXEL, STD_PIXEL),
])

# =====================================
# MODEL COMPONENTS
# =====================================

# SpatioTemporalLSTMCell
class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        # Checkpointing the convolutional operations
        x_concat = checkpoint(self.conv_x, x_t, use_reentrant=False)
        h_concat = checkpoint(self.conv_h, h_t, use_reentrant=False)
        m_concat = checkpoint(self.conv_m, m_t, use_reentrant=False)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + checkpoint(self.conv_o, mem, use_reentrant=False))
        h_new = o_t * torch.tanh(checkpoint(self.conv_last, mem, use_reentrant=False))

        return h_new, c_new, m_new, delta_c, delta_m

# PredRNNv2
class PredRNNv2(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNNv2, self).__init__()

        self.configs = configs
        self.visual = self.configs.visual
        self.visual_path = self.configs.visual_path

        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        if self.visual:
            delta_c_visual = []
            delta_m_visual = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):

            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = checkpoint(self.cell_list[0], net, h_t[0], c_t[0], memory, use_reentrant=False)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
            if self.visual:
                delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = checkpoint(self.cell_list[i], h_t[i - 1], h_t[i], c_t[i], memory, use_reentrant=False)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                if self.visual:
                    delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                    delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous() # 1x59x128x128x1
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + self.configs.decouple_beta * decouple_loss
        return next_frames, loss

# =====================================
# DATASET CLASS
# =====================================

# RNNDataset
class RNNDataset(Dataset):
    def __init__(self, hrv, min_date, max_date, site_locations=site_locations, tsfm=tsfm):
        self.hrv = hrv
        self.site_locations = site_locations
        self.sites = list(site_locations['hrv'].keys())
        self.tsfm = tsfm
        self.min_date = min_date
        self.max_date = max_date
        self.image_times = self._get_image_times()
        self.time_site_pairs = [(time, site) for time in self.image_times for site in self.sites]
        random.shuffle(self.time_site_pairs)

    def _get_image_times(self):
        start_time = time(8)
        end_time = time(17)
        possible_times = []
        date = self.min_date
        while date <= self.max_date:
            current_time = datetime.combine(date, start_time)
            while current_time.time() <= end_time:
                possible_times.append(current_time)
                current_time += timedelta(hours=1)
            date += timedelta(days=1)
        return possible_times

    def __getitem__(self, idx, retry=0, max_retries=20):
        if retry > max_retries:
            raise ValueError(f"Unable to find a valid data item after {max_retries} retries.")
        time, site = self.time_site_pairs[idx]
        try:
          five_hour = slice(str(time), str(time + timedelta(hours=4, minutes=55)))
          hrv_data = self.hrv['data'].sel(time=five_hour).to_numpy()

          x, y = self.site_locations['hrv'][site]
          hrv_features = hrv_data[:, y - 64: y + 64, x - 64: x + 64, 0]

          if hrv_features.shape == (60, 128, 128):
              hrv_features_tensor = torch.tensor(hrv_features, dtype=torch.float32)
              hrv_features_tensor = hrv_features_tensor.unsqueeze(-1)
              hrv_features_tensor = self.tsfm(hrv_features_tensor)
              return hrv_features_tensor
          else:
              raise ValueError('Invalid data shape encountered.')
        except (KeyError, ValueError) as e:
            # If shape mismatch, retry with a different index
            new_idx = np.random.choice([i for i in range(len(self.time_site_pairs)) if i != idx])
            return self.__getitem__(new_idx, retry=retry+1, max_retries=max_retries)

    def __len__(self):
        return len(self.time_site_pairs)

# =====================================
# MODEL WRAPPER CLASS
# =====================================

# Model wrapper
class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn_v2': PredRNNv2,
        }
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("Saved model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask):
        frames_tensor = torch.tensor(frames, dtype=torch.float32, device=self.configs.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.configs.device)
        self.optimizer.zero_grad()
        next_frames, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()

# =====================================
# TRAINING AND TESTING FUNCTIONS
# =====================================

# Training and testing functions
def train_model(model):
    if configs.pretrained_model:
        model.load(configs.pretrained_model)

    eta = configs.sampling_start_value
    train_iter = iter(train_loader)

    for itr in range(1, configs.max_iterations + 1):

        try:
            inputs = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs = next(train_iter)

        inputs_device = inputs.to(configs.device)

        if configs.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(itr)
        else:
            eta, real_input_flag = schedule_sampling(eta, itr)

        cost = model.train(inputs_device, real_input_flag)

        if configs.reverse_input:
            inputs_rev = torch.flip(inputs, dims=[1])
            inputs_rev_device = inputs_rev.to(configs.device)
            rev_cost = model.train(inputs_rev_device, real_input_flag)
            cost += rev_cost
            cost = cost / 2

        if itr % configs.display_interval == 0:
          print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
          print('Training Loss: ' + str(cost))

        if itr % configs.snapshot_interval == 0:
          model.save(itr)

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

# =====================================
# MAIN SCRIPT
# =====================================

# Main script
if __name__ == "__main__":
    for year in [2020, 2021]:
        for month in range(1, 13):  # Assuming you have zarr files for months 1 to 12
            hrv_path = f'/content/drive/MyDrive/ch24/dunet/data/satellite-hrv/{year}/{month}.zarr.zip'
            configs = Config(hrv_path=hrv_path, year=year)
            hrv = xr.open_dataset(configs.hrv_path, engine="zarr", chunks="auto")

            train_ds = RNNDataset(hrv=hrv, min_date=configs.train_min_date, max_date=configs.train_max_date)
            train_loader = DataLoader(train_ds, batch_size=configs.batch_size, shuffle=True)

            model = Model(configs)

            if configs.is_training:
                train_model(model)

