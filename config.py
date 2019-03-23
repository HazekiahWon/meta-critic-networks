# CartPole Task Parameter
L_MIN = 0.5   # min length
L_MAX = 5    # max length

# Hyper Parameters
TASK_NUMS = 10
# STATE_DIM = 4 # cont
# ACTION_DIM = 2 # cat
task_nlayer = 3

Z_DIM = 16
fusion_dim = 16
actor_hdim = 64
dynEmb_hdim = 64
trans_hdim = 64
value_hdim = 64
gauss_dim = 64
task_resample_freq = 100
policy_task_resample_freq = 20
vae_report_freq = 100
actor_report_freq = 40
vae_lr = 0.001
n_sample = 2
# gamma = .95
stochastic_encoder = True
vae_thresh = 5e-4
STEP = 50000
HORIZON = 200 #5,10,20
TEST_SAMPLE_NUMS = 5
use_baseline = True

logdir = 'saved_models'
double_horizon_threshold = 0.9
memo = 'testing the algo in single env'

resume_model_dir = None
reward_scale = 0.1
LOGMIN = -20
LOGMAX = 2