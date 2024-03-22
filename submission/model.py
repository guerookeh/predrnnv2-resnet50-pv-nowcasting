import torch
import numpy as np
from submission.predrnn import PredRNNv2
from submission.resnet import ResNet50

class Config:
    def __init__(self):
        # Data and File Paths
        self.sites_path = None
        self.pretrained_model = 'checkpoints/model.ckpt-1300' # define yourself
        self.save_dir = None
        
        # Training Parameters
        self.max_iterations = None
        self.is_training = 0
        self.batch_size = 1
        self.lr = None
        self.display_interval = None
        self.test_interval = None
        self.snapshot_interval = None
        self.num_save_samples = None
        self.test_iterations = None
        
        # Model Parameters
        self.model_name = 'predrnn_v2'
        self.num_hidden = '64,64,64,64'
        self.num_layers = 4
        self.filter_size = 5
        self.stride = 1
        self.patch_size = 1
        self.layer_norm = 1
        self.decouple_beta = 0.1
        self.input_length = 12
        self.total_length = 60
        self.output_length = self.total_length - self.input_length
        self.img_width = 128
        self.img_channel = 1
        self.injection_action = 'concat'
        self.conv_on_input = 0
        self.res_on_conv = 0
        self.num_action_ch = 4

        # Scheduled Sampling Parameters
        self.reverse_scheduled_sampling = 1
        self.r_sampling_step_1 = None
        self.r_sampling_step_2 = None
        self.r_exp_alpha = None
        self.scheduled_sampling = 1
        self.sampling_stop_iter = None
        self.sampling_start_value = 1.0
        self.sampling_changing_rate = 0.00002
        self.reverse_input = 1

        # Device Configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = 1

        # Visualization Parameters
        self.visual = 0
        self.visual_path = None

        # Training Date Range
        self.train_min_date = None
        self.train_max_date = None

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.config = Config()

        self.config.num_hidden = [int(x) for x in self.config.num_hidden.split(',')]

        self.predrnn = PredRNNv2(self.config.num_layers, self.config.num_hidden, self.config).to(self.config.device)
        stats = torch.load("submission/model.ckpt-1300", map_location=self.config.device)
        self.predrnn.load_state_dict(stats['net_param'])

        self.resnet = ResNet50().to(self.config.device)
        self.resnet.load_state_dict(torch.load("submission/best_model.pth", map_location=self.config.device))

    def forward(self, pv, hrv):

        zeros_to_concat = torch.zeros(self.config.batch_size, self.config.output_length, self.config.img_width, self.config.img_width)
        hrv_concatenated = torch.cat((hrv, zeros_to_concat), dim=1)

        mask_input = 1
        real_input_flag = np.zeros(
            (self.config.batch_size,
             self.config.total_length - mask_input - 1,
             self.config.img_width // self.config.patch_size,
             self.config.img_width // self.config.patch_size,
             self.config.patch_size ** 2 * self.config.img_channel))
        
        if self.config.reverse_scheduled_sampling == 1:
            real_input_flag[:, :self.config.input_length - 1, :, :] = 1.0
        
        hrv_tensor = torch.FloatTensor(hrv_concatenated).to(self.config.device)
        mask_tensor = torch.FloatTensor(real_input_flag).to(self.config.device)

        g_frames, _ = self.predrnn(hrv_tensor, mask_tensor) # tensor as well?

        output_g_frames = g_frames[:, self.config.input_length - 1:]

        pv_preds = []

        for i in range(self.config.output_length):
            frame = hrv[:, i + self.config.input_length, :, :, :]
            g_frame = output_g_frames[:, i, :, :, :]
            g_frame = torch.tensor(g_frame, dtype=torch.float32).permute(0, 3, 1, 2)
            pv_pred = self.resnet(g_frame).squeeze().cpu().detach().numpy()
            pv_preds.append(pv_pred)

        assert len(pv_preds) == self.config.output_length, f"pv_preds length should be {self.config.output_length}, but got {len(pv_preds)}"    
    
        return pv_preds