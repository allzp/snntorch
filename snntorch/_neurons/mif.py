from .neurons import *
import torch.nn as nn
import torch


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MIF(nn.Module):

    def __init__(
        self,
        R_on=1000,
        R_off=100000,
        v_on=110,
        v_off=5,
        tau=100,
        tau_alpha=100,
        E1=0,
        E2=50,
        C=100 * 10 ** (-6),
        k_th=0.6 * 25,

    ):
        super(MIF, self).__init__()

        self.R_on = R_on
        self.R_off = R_off
        self.v_on = v_on
        self.v_off = v_off
        self.tau = tau
        self.tau_alpha = tau_alpha
        self.E1 = E1
        self.E2 = E2
        self.C = C
        self.k_th = k_th

    def forward(self, _input, x1, x2, G1, G2, a, I, v):
        a = -a / self.tau_alpha + _input
        I = (a - I) / self.tau_alpha + I
        v = (I - G1 * (v - self.E1) - G2 * (v - self.E2)) / self.C + v
        # v = torch.clamp(v, min=-1000, max=1000)
        x1 = 1 / self.tau * ((1 - x1) / (1 + torch.exp((self.v_on - (v - self.E1)) / self.k_th)) - x1 / (
                1 + torch.exp(((v - self.E1) - self.v_off) / self.k_th))) + x1  # v[t] or v[t+1] both fine
        x2 = 1 / self.tau * ((1 - x2) / (1 + torch.exp((self.v_on - (v - self.E2)) / self.k_th)) - x2 / (
                1 + torch.exp(((v - self.E2) - self.v_off) / self.k_th))) + x2  # v[t] or v[t+1] both fine
        G1 = x1 / self.R_on + (1 - x1) / self.R_off
        G2 = x2 / self.R_on + (1 - x2) / self.R_off

        return x1, x2, G1, G2, a, I, v

    def init_mif(self, batch_size, *args):
        """Used to initialize x1, x2, G1, G2, a, I
        *args are the input feature dimensions.
        E.g., ``batch_size=128`` and input feature of size=1x28x28 would require ``init_stein(128, 1, 28, 28)``."""
        x1 = torch.ones((batch_size, *args), device=device, dtype=dtype) * 0.0238
        x2 = torch.ones((batch_size, *args), device=device, dtype=dtype) * 0.0238
        G1 = x1 / self.R_on + (1 - x1) / self.R_off
        G2 = x2 / self.R_on + (1 - x2) / self.R_off
        # v = torch.ones((batch_size, *args), device=device, dtype=dtype) * (self.E1 + self.E2)/2
        v = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        I = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        a = torch.zeros((batch_size, *args), device=device, dtype=dtype)

        return x1, x2, G1, G2, a, I, v
