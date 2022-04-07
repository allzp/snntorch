import torch
import torch.nn as nn
from .neurons import *


class STDPMIFSPK(MIFSpiking):
    def __init__(
        self,
        # - stdp
        in_num,
        out_num,
        decay_pre,
        decay_post,
        error_modulator=1.0,
        scaling_pre=1.0,
        scaling_post=1.0,
        # - mif
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
        # - weight matrix
        V_bias=False,
        learn_V=False,  # TODO: combine STDP with gradient
        # - else
        # learn_decay_pre=False,  # TODO: maybe learnable
        # learn_decay_post=False,  # TODO: maybe learnable
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism="none",  # none means no reset
        state_quant=False,
        output=False,
    ):
        super(STDPMIFSPK, self).__init__(
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

        # TODO: input of STDPLeaky should be spike, not weighted spike

        if self.init_hidden:
            (
                self.a,
                self.I,
                self.v,
                self.x1,
                self.x2,
                self.G1,
                self.G2,
                self.trace_pre,
                self.trace_post,
            ) = self.init_stdpmifspk()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

        self.decay_pre = decay_pre
        self.scaling_pre = scaling_pre
        self.decay_post = decay_post
        self.scaling_post = scaling_post
        self.in_num = in_num
        self.out_num = out_num
        self.V_bias = V_bias
        self.error_modulator = error_modulator
        self.V = nn.Linear(self.in_num, self.out_num, bias=self.V_bias)
        self.V.weight.requires_grad = learn_V

    def forward(
        self,
        input_,
        spk=False,
        a=False,
        I=False,
        v=False,
        x1=False,
        x2=False,
        G1=False,
        G2=False,
        trace_pre=False,
        trace_post=False,
    ):
        if (
            hasattr(spk, "init_flag")
            or hasattr(a, "init_flag")
            or hasattr(I, "init_flag")
            or hasattr(v, "init_flag")
            or hasattr(x1, "init_flag")
            or hasattr(x2, "init_flag")
            or hasattr(G1, "init_flag")
            or hasattr(G2, "init_flag")
            or hasattr(trace_pre, "init_flag")
            or hasattr(trace_post, "init_flag")
        ):  # only triggered on first-pass
            # - spk, mem, trace_post have same output size
            spk, a, I, v, x1, x2, G1, G2, trace_post = _SpikeTorchConv(
                spk,
                a,
                I,
                v,
                x1,
                x2,
                G1,
                G2,
                trace_post,
                input_=torch.empty(self.out_num),
            )
            # - trace_pre has input size
            trace_pre = _SpikeTorchConv(trace_pre, input_=input_)

        elif v is False and hasattr(self.v, "init_flag"):  # init_hidden case
            # - internal states, trace_post have same output size
            (
                self.spk,
                self.a,
                self.I,
                self.v,
                self.x1,
                self.x2,
                self.G1,
                self.G2,
                self.trace_post,
            ) = _SpikeTorchConv(
                self.spk,
                self.a,
                self.I,
                self.v,
                self.x1,
                self.x2,
                self.G1,
                self.G2,
                self.trace_post,
                input_=torch.empty(self.out_num),
            )
            # - trace_pre has input size
            self.trace_pre = _SpikeTorchConv(self.trace_pre, input_=input_)

        if not self.init_hidden:
            self.reset = self.mem_reset(v)
            a, I, v, x1, x2, G1, G2 = self.state_fn(input_, a, I, v, x1, x2, G1, G2)

            if self.inhibition:
                spk = self.fire_inhibition(v.size(0), v)  # batch_size
            else:
                spk = self.fire(v)

            # - first add, then decay
            trace_pre = self.decay_pre * (trace_pre + self.scaling_pre * input_)
            trace_post = self.decay_post * (trace_post + self.scaling_post * spk)

            # - weight STDP update
            with torch.no_grad():
                for i in range(self.in_num):
                    for j in range(self.out_num):
                        if input_[i] != 0:
                            self.V.weight[j, i] -= trace_post[j] * self.error_modulator
                        if spk[j] != 0:
                            self.V.weight[j, i] += trace_pre[i] * self.error_modulator

            return spk, a, I, v, x1, x2, G1, G2, trace_pre, trace_post

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden:
            self._stdpmifspk_forward_cases(
                spk, a, I, v, x1, x2, G1, G2, trace_pre, trace_post
            )
            self.reset = self.mem_reset(self.v)
            self.a, self.I, self.v, self.x1, self.x2, self.G1, self.G2 = self.state_fn(
                input_
            )

            if self.inhibition:
                self.spk = self.fire_inhibition(self.v.size(0), self.v)
            else:
                self.spk = self.fire(self.v)

                # - first add, then decay
                self.trace_pre = self.decay_pre * (
                    self.trace_pre + self.scaling_pre * input_
                )
                self.trace_post = self.decay_post * (
                    self.trace_post + self.scaling_post * self.spk
                )

                # - weight STDP update
                with torch.no_grad():
                    for i in range(self.in_num):
                        for j in range(self.out_num):
                            if input_[i] != 0:
                                self.V.weight[j, i] -= (
                                    self.trace_post[j] * self.error_modulator
                                )
                            if self.spk[j] != 0:
                                self.V.weight[j, i] += (
                                    self.trace_pre[i] * self.error_modulator
                                )

            if self.output:  # read-out layer returns output+states
                return (
                    self.spk,
                    self.a,
                    self.I,
                    self.v,
                    self.x1,
                    self.x2,
                    self.G1,
                    self.G2,
                    self.trace_pre,
                    self.trace_post,
                )
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

    def _base_state_function(self, input_, a, I, v, x1, x2, G1, G2):
        base_fn_a = -a / self.tau_alpha + self.V(input_)
        base_fn_I = (a - I) / self.tau_alpha + I
        base_fn_v = (I - G1 * (v - self.E1) - G2 * (v - self.E2)) / self.C + v
        base_fn_x1 = (
            1
            / self.tau
            * (
                (1 - x1) / (1 + torch.exp((self.v_on - (v - self.E1)) / self.k_th))
                - x1 / (1 + torch.exp(((v - self.E1) - self.v_off) / self.k_th))
            )
            + x1
        )  # v[t] or v[t+1] both fine
        base_fn_x2 = (
            1
            / self.tau
            * (
                (1 - x2) / (1 + torch.exp((self.v_on - (v - self.E2)) / self.k_th))
                - x2 / (1 + torch.exp(((v - self.E2) - self.v_off) / self.k_th))
            )
            + x2
        )  # v[t] or v[t+1] both fine
        base_fn_G1 = x1 / self.R_on + (1 - x1) / self.R_off
        base_fn_G2 = x2 / self.R_on + (1 - x2) / self.R_off

        return (
            base_fn_a,
            base_fn_I,
            base_fn_v,
            base_fn_x1,
            base_fn_x2,
            base_fn_G1,
            base_fn_G2,
        )

    def _build_state_function(self, input_, a, I, v, x1, x2, G1, G2):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = self._base_state_function(
                input_, a, I, v - self.reset * self.threshold, x1, x2, G1, G2
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, a, I, v, x1, x2, G1, G2
            ) - self.reset * self._base_state_function(input_, a, I, v, x1, x2, G1, G2)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, a, I, v, x1, x2, G1, G2)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn_a = -self.a / self.tau_alpha + self.V(input_)
        base_fn_I = (self.a - self.I) / self.tau_alpha + self.I
        base_fn_v = (
            self.I - self.G1 * (self.v - self.E1) - self.G2 * (self.v - self.E2)
        ) / self.C + self.v
        base_fn_x1 = (
            1
            / self.tau
            * (
                (1 - self.x1)
                / (1 + torch.exp((self.v_on - (self.v - self.E1)) / self.k_th))
                - self.x1
                / (1 + torch.exp(((self.v - self.E1) - self.v_off) / self.k_th))
            )
            + self.x1
        )  # v[t] or v[t+1] both fine
        base_fn_x2 = (
            1
            / self.tau
            * (
                (1 - self.x2)
                / (1 + torch.exp((self.v_on - (self.v - self.E2)) / self.k_th))
                - self.x2
                / (1 + torch.exp(((self.v - self.E2) - self.v_off) / self.k_th))
            )
            + self.x2
        )  # v[t] or v[t+1] both fine
        base_fn_G1 = self.x1 / self.R_on + (1 - self.x1) / self.R_off
        base_fn_G2 = self.x2 / self.R_on + (1 - self.x2) / self.R_off

        return (
            base_fn_a,
            base_fn_I,
            base_fn_v,
            base_fn_x1,
            base_fn_x2,
            base_fn_G1,
            base_fn_G2,
        )

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

    def _stdpmifspk_forward_cases(
        self, spk, a, I, v, x1, x2, G1, G2, trace_pre, trace_post
    ):
        if (
            spk is not False
            or a is not False
            or I is not False
            or v is not False
            or x1 is not False
            or x2 is not False
            or G1 is not False
            or G2 is not False
            or trace_pre is not False
            or trace_post is not False
        ):
            raise TypeError(
                "When `init_hidden=True`, STDPLeaky expects 1 input argument."
            )

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], STDPMIFSPK):
                cls.instances[layer].a.detach_()
                cls.instances[layer].I.detach_()
                cls.instances[layer].v.detach_()
                cls.instances[layer].x1.detach_()
                cls.instances[layer].x2.detach_()
                cls.instances[layer].G1.detach_()
                cls.instances[layer].G2.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], STDPMIFSPK):
                cls.instances[layer].a = _SpikeTensor(init_flag=False)
                cls.instances[layer].I = _SpikeTensor(init_flag=False)
                cls.instances[layer].v = _SpikeTensor(init_flag=False)
                cls.instances[layer].x1 = _SpikeTensor(init_flag=False)
                cls.instances[layer].x2 = _SpikeTensor(init_flag=False)
                cls.instances[layer].G1 = _SpikeTensor(init_flag=False)
                cls.instances[layer].G2 = _SpikeTensor(init_flag=False)
