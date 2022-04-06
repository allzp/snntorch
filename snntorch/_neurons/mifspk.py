from .neurons import *
import torch

class MIFSPK(MIFSpiking):

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
        threshold=30.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism="none",
        state_quant=False,
        output=False,
    ):
        super(MIFSPK, self).__init__(
            R_on,
            R_off,
            v_on,
            v_off,
            tau,
            tau_alpha,
            E1,
            E2,
            C,
            k_th,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        if self.init_hidden:
            self.a, self.I, self.v, self.x1, self.x2, self.G1, self.G2 = self.init_mifspiking()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

    def forward(self, input_, a=False, I=False, v=False, x1=False, x2=False, G1=False, G2=False):
        if hasattr(a, "init_flag") or hasattr(I, "init_flag") or hasattr(v, "init_flag") or hasattr(x1, "init_flag") \
            or hasattr(x2, "init_flag") or hasattr(G1, "init_flag") or hasattr(G2, "init_flag"):  # only triggered on first-pass
            a, I, v, x1, x2, G1, G2 = _SpikeTorchConv(a, I, v, x1, x2, G1, G2, input_=input_)
        elif v is False and hasattr(self.v, "init_flag"):  # init_hidden case
            self.a, self.I, self.v, self.x1, self.x2, self.G1, self.G2 = _SpikeTorchConv(
                self.a, self.I, self.v, self.x1, self.x2, self.G1, self.G2, input_=input_)

        # TO-DO: alternatively, we could do torch.exp(-1 / self.beta.clamp_min(0)),
        # giving actual time constants instead of values in [0, 1] as initial beta
        # beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            self.reset = self.mem_reset(v)
            a, I, v, x1, x2, G1, G2 = self.state_fn(input_, a, I, v, x1, x2, G1, G2)

            if self.state_quant:
                v = self.state_quant(v)

            if self.inhibition:
                spk = self.fire_inhibition(v.size(0), v)  # batch_size
            else:
                spk = self.fire(v)

            return spk, a, I, v, x1, x2, G1, G2

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden:
            self._mifspk_forward_cases(a, I, v, x1, x2, G1, G2)
            self.reset = self.mem_reset(self.v)
            self.v = self.state_fn(input_)

            if self.state_quant:
                self.v = self.state_quant(self.v)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.v.size(0), self.v)
            else:
                self.spk = self.fire(self.v)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.a, self.I, self.v, self.x1, self.x2, self.G1, self.G2
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

    def _base_state_function(self, input_, a, I, v, x1, x2, G1, G2):
        base_fn_a = - a / self.tau_alpha + input_
        base_fn_I = (a - I) / self.tau_alpha + I
        base_fn_v = (I - G1 * (v - self.E1) - G2 * (v - self.E2)) / self.C + v
        base_fn_x1 = 1 / self.tau * ((1 - x1) / (1 + torch.exp((self.v_on - (v - self.E1)) / self.k_th)) - x1 / (
                1 + torch.exp(((v - self.E1) - self.v_off) / self.k_th))) + x1  # v[t] or v[t+1] both fine
        base_fn_x2 = 1 / self.tau * ((1 - x2) / (1 + torch.exp((self.v_on - (v - self.E2)) / self.k_th)) - x2 / (
                1 + torch.exp(((v - self.E2) - self.v_off) / self.k_th))) + x2  # v[t] or v[t+1] both fine
        base_fn_G1 = x1 / self.R_on + (1 - x1) / self.R_off
        base_fn_G2 = x2 / self.R_on + (1 - x2) / self.R_off

        return base_fn_a, base_fn_I, base_fn_v, base_fn_x1, base_fn_x2, base_fn_G1, base_fn_G2

    def _build_state_function(self, input_, a, I, v, x1, x2, G1, G2):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function(input_, a, I, v - self.reset * self.threshold, x1, x2, G1, G2)
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, a, I, v , x1, x2, G1, G2
            ) - self.reset * self._base_state_function(
                input_, a, I, v , x1, x2, G1, G2)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, a, I, v, x1, x2, G1, G2)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn_a = - self.a / self.tau_alpha + input_
        base_fn_I = (self.a - self.I) / self.tau_alpha + self.I
        base_fn_v = (self.I - self.G1 * (self.v - self.E1) - self.G2 * (self.v - self.E2)) / self.C + self.v
        base_fn_x1 = 1 / self.tau * (
                (1 - self.x1) / (1 + torch.exp((self.v_on - (self.v - self.E1)) / self.k_th)) - self.x1 / (
                1 + torch.exp(((self.v - self.E1) - self.v_off) / self.k_th))) + self.x1  # v[t] or v[t+1] both fine
        base_fn_x2 = 1 / self.tau * (
                (1 - self.x2) / (1 + torch.exp((self.v_on - (self.v - self.E2)) / self.k_th)) - self.x2 / (
                1 + torch.exp(((self.v - self.E2) - self.v_off) / self.k_th))) + self.x2  # v[t] or v[t+1] both fine
        base_fn_G1 = self.x1 / self.R_on + (1 - self.x1) / self.R_off
        base_fn_G2 = self.x2 / self.R_on + (1 - self.x2) / self.R_off

        return base_fn_a, base_fn_I, base_fn_v, base_fn_x1, base_fn_x2, base_fn_G1, base_fn_G2

    def _build_state_function_hidden(self, input_):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = tuple(
                map(
                    lambda x, y: x - y,
                    self._base_state_function_hidden(input_),
                    (0, self.reset * self.threshold),
                )
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = tuple(
                map(
                    lambda x, y: x - self.reset * y,
                    self._base_state_function_hidden(input_),
                    self._base_state_function_hidden(input_),
                )
            )
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function_hidden(input_)
        return state_fn

    def _mifspk_forward_cases(self, a, I, v, x1, x2, G1, G2):
        if a is not False or I is not False or v is not False or x1 is not False \
            or x2 is not False or G1 is not False or G2 is not False:
            raise TypeError("When `init_hidden=True`, MIFSPK expects 1 input argument.")

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Leaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Leaky):
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)
