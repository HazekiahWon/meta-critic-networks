from utils import *
from rlkit.torch.distributions import TanhNormal, Normal
from rlkit.torch.core import PyTorchModule
from rlkit.torch.modules import LayerNorm
from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu
nonlinearity = F.leaky_relu

class BaseFCPrototype(PyTorchModule):

    def __init__(self, input_dims, fcs, nonlin=None):
        super(BaseFCPrototype, self).__init__()
        h_layers = list()

        self.pre_layers = nn.ModuleList([nn.Linear(d, fusion_dim) for d in input_dims])
        inf = input_dims[0]
        fc1 = nn.Linear(fusion_dim * len(self.pre_layers), inf)
        for nf in fcs:
            h_layers.append(nn.Linear(inf, nf))
            inf = nf
        if nonlin is None: self.nonlinearity = nonlinearity
        else: self.nonlinearity = nonlin
        self.layers = nn.ModuleList([fc1]+h_layers)

    def forward(self,*inputs):
        fusion_features = list()
        for inp,l in zip(inputs,self.pre_layers):
            fusion_features.append(self.nonlinearity(l(inp)))

        x = torch.cat(fusion_features, dim=-1)
        for l in self.layers:
            x = self.nonlinearity(l(x))

        return x

class FCPrototype(BaseFCPrototype):

    def __init__(self, input_dims, fcs, out_size, out_act, nonlin=None):
        super(FCPrototype, self).__init__(input_dims, fcs, nonlin)
        inf = fcs[-1]
        self.out_layer = nn.Linear(inf, out_size)
        self.out_act = out_act

    def forward(self,*inputs):
        x = super().forward(*inputs)
        out = self.out_layer(x)
        if self.out_act is not None:
            return self.out_act(out)
        else: return out

class GaussianFCPrototype(BaseFCPrototype):

    def __init__(self, input_dims, fcs, out_size, nonlin):
        super(GaussianFCPrototype, self).__init__(input_dims, fcs, nonlin)

        self.mean_layer = nn.Linear(fcs[-1], out_size)
        self.logstd_layer = nn.Linear(fcs[-1], out_size)

    def forward(self,*inputs):
        x = super().forward(*inputs)
        mean = self.mean_layer(x)
        logstd = self.logstd_layer(x)
        logstd = torch.clamp(logstd, LOGMIN, LOGMAX)
        std = torch.exp(logstd)

        unit_normal = Normal(
                ptu.zeros(mean.size()),
                ptu.ones(std.size())
        )
        eps = unit_normal.sample()
        pre_tanh_z = mean.cpu()+std.cpu()*eps
        action = torch.tanh(pre_tanh_z)
        logp = unit_normal.log_prob(eps) #
        logp = logp.sum(dim=1, keepdim=True) # logsum = exp mult
        return action, pre_tanh_z, logp, mean, logstd


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            latent_dim=None,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            # assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return np_ify(outputs)

    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOGMIN, LOGMAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        tanh_normal = TanhNormal(mean, std)
        if reparameterize:
            action, pre_tanh_value = tanh_normal.rsample(
                return_pretanh_value=True
            )
        else:
            action, pre_tanh_value = tanh_normal.sample(
                return_pretanh_value=True
            )
        log_prob = tanh_normal.log_prob(
            action,
            pre_tanh_value=pre_tanh_value
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action.cuda(), log_prob.cuda(), pre_tanh_value.cuda(), mean, log_std

class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

class Actor_with_latent(FCPrototype): # s,z
    def __init__(self, STATE_DIM, ACTION_DIM, nonlin=None):
        super(Actor_with_latent, self).__init__((STATE_DIM, Z_DIM), actor_fcs, ACTION_DIM, lambda x:F.log_softmax(x, dim=-1), nonlin)

class Trans_with_latent(FCPrototype): # s,a,z
    def __init__(self, STATE_DIM, ACTION_DIM, nonlin=None):
        super(Trans_with_latent, self).__init__((STATE_DIM, ACTION_DIM, Z_DIM), trans_fcs, STATE_DIM, None, nonlin)

class VBase_with_latent(FCPrototype): # s,z
    def __init__(self, STATE_DIM, nonlin=None):
        super(VBase_with_latent, self).__init__((STATE_DIM, Z_DIM), value_fcs, 1, None, nonlin)

class VBase(FCPrototype): # s,z
    def __init__(self, STATE_DIM, nonlin=None):
        super(VBase, self).__init__((STATE_DIM,), value_fcs, 1, None, nonlin)

class QBase(FCPrototype): # s,z
    def __init__(self, STATE_DIM, ACTION_DIM, nonlin=None):
        super(QBase, self).__init__((STATE_DIM, ACTION_DIM), value_fcs, 1, None, nonlin)

class Actor(FCPrototype):
    def __init__(self, STATE_DIM, ACTION_DIM, nonlin=None):
        super(Actor, self).__init__((STATE_DIM,), actor_fcs, ACTION_DIM, lambda x:F.log_softmax(x, dim=-1), nonlin)

class GaussianActor(GaussianFCPrototype):
    def __init__(self, STATE_DIM, ACTION_DIM, nonlin=None):
        super(GaussianActor, self).__init__((STATE_DIM,), actor_fcs, ACTION_DIM, nonlin)

class GaussianActor_with_latent(GaussianFCPrototype):
    def __init__(self, STATE_DIM, ACTION_DIM, nonlin=None):
        super(GaussianActor_with_latent, self).__init__((STATE_DIM,Z_DIM), actor_fcs, ACTION_DIM, nonlin)

class DynEmb(FCPrototype): # sas
    def __init__(self, STATE_DIM,ACTION_DIM, nonlin=None):
        super(DynEmb, self).__init__((STATE_DIM,ACTION_DIM,STATE_DIM), dyn_fcs, gauss_dim, None, nonlin)

        self._enc_mu = torch.nn.Linear(gauss_dim, Z_DIM)
        self._enc_log_sigma = torch.nn.Linear(gauss_dim, Z_DIM)


    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False).cuda()  # Reparameterization trick

    def forward(self, s, a, sp):
        out = super(DynEmb, self).forward(s,a,sp) # n, vaedim
        out = self._sample_latent(out) # n,zdim
        out = torch.mean(out, dim=0, keepdim=True)
        return out

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, states, actions):
        z = get_dyn_embedding(states[:-1], actions[:-1], states[1:],
                                     self.encoder)  # 1,z_dim
        nstate = get_predicted_nstate(states, actions,
                                            z.repeat(states.shape[0], 1),
                                            self.decoder, do_grad=True)
        return nstate,z

class TaskConfigNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TaskConfigNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out