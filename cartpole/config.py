# CartPole Task Parameter
L_MIN = 0.5   # min length
L_MAX = 5    # max length

# Hyper Parameters
TASK_NUMS = 10
STATE_DIM = 4 # cont
ACTION_DIM = 2 # cat
task_nlayer = 3

Z_DIM = 16
actor_dim = 64
task_dim = 64
value_dim = 64
vae_dim = 64
task_resample_freq = 100
vae_report_freq = 100
actor_report_freq = 100
vae_lr = 0.003
n_sample = 2
gamma = .95
stochastic_encoder = True
vae_thresh = 1e-4+0.2
STEP = 50000
HORIZON = 30 #5,10,20
TEST_SAMPLE_NUMS = 5
use_baseline = True

logdir = 'saved_models'
double_horizon_threshold = 0.9
memo = 'pretrain baseline with straight mse, not bellman backup'