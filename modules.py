from utils import *
from rlkit.torch.distributions import TanhNormal, Normal
import rlkit.torch.pytorch_util as ptu
nonlinearity = F.leaky_relu
class BaseFCPrototype(nn.Module):

    def __init__(self, input_dims, fcs):
        super(BaseFCPrototype, self).__init__()
        h_layers = list()

        self.pre_layers = nn.ModuleList([nn.Linear(d, fusion_dim) for d in input_dims])
        inf = input_dims[0]
        fc1 = nn.Linear(fusion_dim * len(self.pre_layers), inf)
        for nf in fcs:
            h_layers.append(nn.Linear(inf, nf))
            inf = nf

        self.layers = nn.ModuleList([fc1]+h_layers)

    def forward(self,*inputs):
        fusion_features = list()
        for inp,l in zip(inputs,self.pre_layers):
            fusion_features.append(nonlinearity(l(inp)))

        x = torch.cat(fusion_features, dim=-1)
        for l in self.layers:
            x = nonlinearity(l(x))

        return x

class FCPrototype(BaseFCPrototype):

    def __init__(self, input_dims, fcs, out_size, out_act):
        super(FCPrototype, self).__init__(input_dims, fcs)
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

    def __init__(self, input_dims, fcs, out_size):
        super(GaussianFCPrototype, self).__init__(input_dims, fcs)

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
        return action, pre_tanh_z, logp

class Actor_with_latent(FCPrototype): # s,z
    def __init__(self, STATE_DIM, ACTION_DIM):
        super(Actor_with_latent, self).__init__((STATE_DIM, Z_DIM), actor_fcs, ACTION_DIM, lambda x:F.log_softmax(x, dim=-1))

class Trans_with_latent(FCPrototype): # s,a,z
    def __init__(self, STATE_DIM, ACTION_DIM):
        super(Trans_with_latent, self).__init__((STATE_DIM, ACTION_DIM, Z_DIM), trans_fcs, STATE_DIM, None)

class VBase_with_latent(FCPrototype): # s,z
    def __init__(self, STATE_DIM):
        super(VBase_with_latent, self).__init__((STATE_DIM, Z_DIM), value_fcs, 1, None)

class VBase(FCPrototype): # s,z
    def __init__(self, STATE_DIM):
        super(VBase, self).__init__((STATE_DIM,), value_fcs, 1, None)

class Actor(FCPrototype):
    def __init__(self, STATE_DIM, ACTION_DIM):
        super(Actor, self).__init__((STATE_DIM,), actor_fcs, ACTION_DIM, lambda x:F.log_softmax(x, dim=-1))

class GaussianActor(GaussianFCPrototype):
    def __init__(self, STATE_DIM, ACTION_DIM):
        super(GaussianActor, self).__init__((STATE_DIM,), actor_fcs, ACTION_DIM)

class DynEmb(FCPrototype): # sas
    def __init__(self, STATE_DIM,ACTION_DIM):
        super(DynEmb, self).__init__((STATE_DIM,ACTION_DIM,STATE_DIM), dyn_fcs, gauss_dim, None)

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

class DynamicsEmb(nn.Module):
    def __init__(self, hidden_size, output_size, emb_dim, z_dim, k, stoch):
        super(DynamicsEmb, self).__init__()
        self.stoch = stoch
        pad = k//2
        self.pre_s = nn.Linear(STATE_DIM, fusion_dim)
        self.pre_a = nn.Linear(ACTION_DIM, fusion_dim)
        self.conv1 = nn.Conv1d(fusion_dim*3, hidden_size, k, padding=pad)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, k, padding=pad)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, k, padding=pad) # 1,hidden_size,t
        self.conv4 = nn.Conv1d(hidden_size, output_size, 1)
        self._enc_mu = torch.nn.Linear(emb_dim, z_dim)
        self._enc_log_sigma = torch.nn.Linear(emb_dim, z_dim)

        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, output_size)

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
        pro_s = nonlinearity(self.pre_s(s))
        pro_a = nonlinearity(self.pre_a(a))
        pro_sp = nonlinearity(self.pre_s(sp))
        #### concat the above in the same dimension
        x = torch.cat((pro_s,pro_a,pro_sp), dim=-1).transpose(-2,-1).unsqueeze(0) # 1,n,sha*3
        #### concat into a new dimension
        out = nonlinearity(self.conv1(x))
        out = out+nonlinearity(self.conv2(out))
        out = out+nonlinearity(self.conv3(out))
        # out = torch.transpose(out, -1, -2) # 1,t,hidden
        # out = self.fc4(out) #
        out = self.conv4(out)
        out = torch.mean(out, dim=-1)

        # out = nonlinearity(self.fc1(x))
        # out = nonlinearity(self.fc2(out))
        # out = nonlinearity(self.fc3(out))
        # out = self.fc4(out)
        # out = torch.mean(out, dim=1)
        if self.stoch:
            out = self._sample_latent(out)
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