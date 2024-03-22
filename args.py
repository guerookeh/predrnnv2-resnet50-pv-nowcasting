import argparse

parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# Training/Testing
parser.add_argument('--is_training', type=int, default=1) # edit this when done training
parser.add_argument('--device', type=str, default='cpu:0') # edit this for gpu

# Model
parser.add_argument('--model_name', type=str, default='predrnn_v2')
parser.add_argument('--pretrained_model', type=str, default='') # edit this correspondingly
parser.add_argument('--num_hidden', type=str, default='64,64,64,64') # maybe edit this
parser.add_argument('--filter_size', type=int, default=5) # maybe edit this
parser.add_argument('--stride', type=int, default=1) # maybe edit this
parser.add_argument('--patch_size', type=int, default=4) # maybe edit this
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1) # (what is this?)

# Data and Paths
parser.add_argument('--save_dir', type=float, default='checkpoints/hrv_predrnnv2') # definitely edit this
parser.add_argument('--hrv_path', type=str, default='data/hrv') # definitely edit this
parser.add_argument('--pv_path', type=str, default='data/pv') # definitely edit this
parser.add_argument('--input_length', type=int, default=12) # 12 frames
parser.add_argument('--total_length', type=int, default=60) # for a total of 60 frames
parser.add_argument('--img_width', type=int, default=128) # 128x128 HRV frames
parser.add_argument('--img_channel', type=int, default=1) # with a single channel (?)

# Currently configured for: 1 day.

# ---- For this section, we need to figure out how many data points we have... ----

# Reverse Scheduled Sampling
parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
parser.add_argument('--r_sampling_step_1', type=float, default=45) # 0.3125 of max_iterations
parser.add_argument('--r_sampling_step_2', type=int, default=90) # 0.625 of max_iterations
parser.add_argument('--r_exp_alpha', type=int, default=9) # 0.0625 of max_iterations
# Scheduled Sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=90) # 0.625 of max_iterations
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# ----------------------------------------------------------------------------------

# Optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=144) # Also need amount of data points here.
parser.add_argument('--display_interval', type=int, default=12) # 0.0833 of max_iterations
parser.add_argument('--test_interval', type=int, default=9) # 0.0625 of max_iterations
parser.add_argument('--snapshot_interval', type=int, default=9) # And here.
parser.add_argument('--num_save_samples', type=int, default=10) # Also here (what's this even for?)
parser.add_argument('--n_gpu', type=int, default=1)

# Visualization of Memory Decoupling
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# Action-Based PredRNN
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')

args = parser.parse_args()
