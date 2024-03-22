import os
import json
import random
import subprocess
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2
from datetime import datetime, timedelta, time
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

with open('indices.json') as f:
    site_locations = {
        data_source: {
            int(site): (int(location[0]), int(location[1]))
            for site, location in locations.items()
        }
        for data_source, locations in json.load(f).items()
    }

torch.manual_seed(7)
MEAN_PIXEL = 0.35
STD_PIXEL = 0.01
tsfm = transforms.Compose([
    transforms.Normalize(MEAN_PIXEL, STD_PIXEL),
])

class PVDataset(Dataset):
    def __init__(self, pv, hrv, min_date, max_date, site_locations=site_locations, tsfm=tsfm):
        self.pv = pv
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
        end_time = time(16)
        possible_times = []
        date = self.min_date
        while date <= self.max_date:
            current_time = datetime.combine(date, start_time)
            while current_time.time() <= end_time:
                possible_times.append(current_time)
                current_time += timedelta(hours=1)
            date += timedelta(days=1)
        return possible_times

    def __getitem__(self, idx, retry=0, max_retries=10000):
        try:
            time, site = self.time_site_pairs[idx]

            start_time = time
            hour_range = slice(str(start_time), str(start_time))

            pv_target_data = self.pv.loc[(slice(str(start_time), str(start_time)), site), :]
            pv_target_data = pv_target_data.values.flatten()

            x, y = self.site_locations['hrv'][site]
            hrv_feature_data = self.hrv['data'].sel(time=hour_range).to_numpy()
            hrv_features = hrv_feature_data[:, y - 64: y + 64, x - 64: x + 64, 0]

            if hrv_features.shape != (1, 128, 128) or pv_target_data.shape != (1,):
              raise ValueError('Invalid data shape encountered')

            hrv_features_tensor = torch.tensor(hrv_features, dtype=torch.float32)
            pv_targets_tensor = torch.tensor(pv_target_data, dtype=torch.float32)
            hrv_features_tensor = self.tsfm(hrv_features_tensor)
            return hrv_features_tensor, pv_targets_tensor

        except (KeyError, ValueError) as e:
            if retry < max_retries:
                new_idx = (idx + np.random.randint(1, len(self.time_site_pairs))) % len(self.time_site_pairs)
                return self.__getitem__(new_idx, retry=retry+1, max_retries=max_retries)
            else:
                print(f"Max retries reached for index {idx}.")
                raise e

    def __len__(self):
        return len(self.time_site_pairs)
    

class ResNet50(nn.Module):
    def __init__(self, num_input_channels=1, num_outputs=1):
        super(ResNet50, self).__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        x = self.resnet(x)
        return x
    

def run_shell_command(command):
    subprocess.run(command, shell=True, check=True)

def download_files(year, month):
    # Satellite HRV data
    hrv_directory = f"resnet_data/satellite-hrv/{year}"
    os.makedirs(hrv_directory, exist_ok=True)

    hrv_file_path = f"{hrv_directory}/{month}.zarr.zip"
    if not os.path.exists(hrv_file_path):
        hrv_url = f"https://huggingface.co/datasets/climatehackai/climatehackai-2023/resolve/main/satellite-hrv/{year}/{month}.zarr.zip"
        run_shell_command(f"curl -L {hrv_url} --output {hrv_file_path}")

    # PV data
    pv_directory = f"resnet_data/pv/{year}"
    os.makedirs(pv_directory, exist_ok=True)

    pv_file_path = f"{pv_directory}/{month}.parquet"
    if not os.path.exists(pv_file_path):
        pv_url = f"https://huggingface.co/datasets/climatehackai/climatehackai-2023/resolve/main/pv/{year}/{month}.parquet"
        run_shell_command(f"curl -L {pv_url} --output {pv_file_path}")

def delete_files(year, month):
    hrv_file_path = f"resnet_data/satellite-hrv/{year}/{month}.zarr.zip"
    if os.path.exists(hrv_file_path):
        os.remove(hrv_file_path)

    pv_file_path = f"resnet_data/pv/{year}/{month}.parquet"
    if os.path.exists(pv_file_path):
        os.remove(pv_file_path)

def train_model(model, dataloader, optimizer, criterion, num_epochs=25, device='cuda', checkpoint_interval=5, checkpoint_dir='checkpoints'):
    model.train()  # Set the model to training mode
    model.to(device)  # Move the model to the specified device

    os.makedirs(checkpoint_dir, exist_ok=True)  # Create the checkpoint directory if it doesn't exist

    best_loss = float('inf')

    max_batches_per_epoch = 100

    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_counter = 0

        for inputs, targets in dataloader:

            if batch_counter >= max_batches_per_epoch:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_loss += loss.item() * inputs.size(0)

            batch_counter += 1

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Checkpointing
        if epoch % checkpoint_interval == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved at {checkpoint_path}')

        # Update best loss if the current loss is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved at {best_model_path}')

    print('Training complete')


# Prepare dataset and dataloader
if __name__ == "__main__":
    batch_size = 8
    pretrained_model_path = None # none yet
    is_training = True
    num_epochs = 30
    learning_rate = 0.001
    checkpoint_interval = 5  # Save a checkpoint every 5 epochs
    checkpoint_dir = 'resnet_checkpoints'  # Directory to save checkpoints

    model = ResNet50(num_input_channels=1, num_outputs=1)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Change to nn.L1Loss() if you prefer

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for year in [2020, 2021]:
        for month in range(5, 13):

            download_files(year, month)

            pv_path = f"resnet_data/pv/{year}/{month}.parquet"
            pv = pd.read_parquet(pv_path).drop("generation_wh", axis=1)

            hrv_path = f"resnet_data/satellite-hrv/{year}/{month}.zarr.zip"
            hrv = xr.open_dataset(hrv_path, engine="zarr", chunks="auto")

            train_min_date = datetime(year, month, 1)
            train_max_date = datetime(year, month, 27)  # Adjust as needed

            train_ds = PVDataset(pv=pv, hrv=hrv, min_date=train_min_date, max_date=train_max_date)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            if is_training:
                train_model(model, train_loader, optimizer, criterion, num_epochs=num_epochs, device=device, checkpoint_interval=checkpoint_interval, checkpoint_dir=checkpoint_dir)

            delete_files(year, month)



