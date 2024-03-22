import torch
import torchvision.transforms as transforms
import numpy as np
import time
import json
import random
from datetime import datetime, timedelta, time
from torch.utils.data import Dataset

# Site Locations JSON
with open("/content/drive/MyDrive/ch24/dunet/indices.json") as f:
    site_locations = {
        data_source: {
            int(site): (int(location[0]), int(location[1]))
            for site, location in locations.items()
        }
        for data_source, locations in json.load(f).items()
    }

# Image Transforms (useful?)
MEAN_PIXEL = 0.35
STD_PIXEL = 0.01
torch.manual_seed(7)
tsfm = transforms.Compose([
    transforms.Normalize(MEAN_PIXEL, STD_PIXEL),
])

# PredRNN Dataset
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
              hrv_features_tensor = self.tsfm(hrv_features_tensor)
              return hrv_features_tensor, hrv_targets_tensor
          else:
              raise ValueError('Invalid data shape encountered.')
        except (KeyError, ValueError) as e:
            # If shape mismatch, retry with a different index
            new_idx = np.random.choice([i for i in range(len(self.time_site_pairs)) if i != idx])
            return self.__getitem__(new_idx, retry=retry+1, max_retries=max_retries)

    def __len__(self):
        return len(self.time_site_pairs)