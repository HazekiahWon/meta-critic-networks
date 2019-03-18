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
vae_lr = 0.003
n_sample = 2
gamma = .99
TASK_CONFIG_DIM = 3
EPISODE = 1000
STEP = 500
SAMPLE_NUMS = 30 #5,10,20
TEST_SAMPLE_NUMS = 5