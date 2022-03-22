import torch
import torch.nn as nn
from .neurons import *


class STDPLeaky(LIF):

    def __init__(
        self,
        in_num,
        out_num,
        beta,
        decay_pre,
        decay_post,
        scaling_pre=1.0,
        scaling_post=1.0,
        threshold=1.0,
        V_bias=False,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        # learn_decay_pre=False,  # TODO: maybe learnable?
        # learn_decay_post=False,  # TODO: maybe learnable?
        # learn_V=False,  # TODO: if learnable, gradient?
        learn_threshold=False,
        reset_mechanism="subtract",
        output=False,
    ):
        super(STDPLeaky, self).__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            output,
        )

        if self.init_hidden:
            self.spk, self.mem, self.trace_pre, self.trace_post = self.init_stdpleaky()
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
        self.V = nn.Linear(self.in_num, self.out_num, bias=self.V_bias)

    def forward(self, input_, spk=False, mem=False, trace_pre=False, trace_post=False):

        if hasattr(spk, "init_flag") or hasattr(mem, "init_flag") or hasattr(trace_pre, "init_flag") or hasattr(trace_post, "init_flag"):  # only triggered on first-pass
            spk, mem, trace_post = _SpikeTorchConv(spk, mem, trace_post, input_=torch.empty(self.out_num))
            trace_pre = _SpikeTorchConv(trace_pre, input_=input_)

        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.spk, self.mem, self.trace_post = _SpikeTorchConv(self.spk, self.mem, self.trace_post, input_=torch.empty(self.out_num))
            self.trace_pre = _SpikeTorchConv(self.trace_pre, input_=input_)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self.state_fn(input_, mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            trace_pre = self.decay_pre * trace_pre + self.scaling_pre * input_
            trace_post = self.decay_post * trace_post + self.scaling_post * spk

            return spk, mem, trace_pre, trace_post

        # intended for truncated-BPTT where instance variables are hidden states

        if self.init_hidden:
            self._stdpleaky_forward_cases(mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self.state_fn(input_)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

                self.trace_pre = self.decay_pre * self.trace_pre + self.scaling_pre * input_
                self.trace_post = self.decay_post * self.trace_post + self.scaling_post * self.spk

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem, self.trace_pre, self.trace_post
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

        uni_spk = self.spk if self.init_hidden else spk
        uni_trace_pre = self.trace_pre if self.init_hidden else trace_pre
        uni_trace_post = self.trace_post if self.init_hidden else trace_post
        self.V.register_forward_hook(self._stdp_weight_update(input_, uni_spk, uni_trace_pre, uni_trace_post))

    def _stdp_weight_update(self, input_, spk, trace_pre, trace_post):
        for i in range(self.in_num):
            for j in range(self.out_num):
                if input_[i] != 0:
                    self.V.weight[i, j] -= trace_post[j]
                if spk[j] != 0:
                    self.V.weight[i, j] += self.trace_pre[i]

    def _base_state_function(self, input_, mem):
        base_fn = self.beta.clamp(0, 1) * mem + self.V(input_)
        return base_fn

    def _build_state_function(self, input_, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function(input_, mem) - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, mem
            ) - self.reset * self._base_state_function(input_, mem)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, mem)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + self.V(input_)
        return base_fn

    def _build_state_function_hidden(self, input_):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function_hidden(input_) - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function_hidden(
                input_
            ) - self.reset * self._base_state_function_hidden(input_)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function_hidden(input_)
        return state_fn

    def _stdpleaky_forward_cases(self, spk, mem, trace_pre, trace_post):
        if mem is not False or spk is not False or trace_pre is not False or trace_post is not False:
            raise TypeError("When `init_hidden=True`, STDPLeaky expects 1 input argument.")

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RLeaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], STDPLeaky):
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)

