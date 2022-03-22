import torch
import torch.nn as nn
from .neurons import *


class STDPLeaky(LIF):

    def __init__(
        self,
        beta,
        V,
        trace_pre,
        decay_pre,
        scaling_pre,
        trace_post,
        decay_post,
        scaling_post,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        # learn_decay_pre=False,  # TODO: maybe learnable?
        # learn_decay_post=False,  # TODO: maybe learnable?
        learn_V=False,  # TODO: if learnable, gradient?
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
            self.mem = self.init_leaky()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

        self._V_register_buffer(V, learn_V)

        # self.register_buffer("trace_pre", torch.Tensor())
        # self.register_buffer("trace_post", torch.Tensor())
        self.decay_pre = decay_pre
        self.scaling_pre = scaling_pre
        self.decay_post = decay_post
        self.scaling_post = scaling_post

    def forward(self, input_, spk=False, mem=False):
        if hasattr(spk, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            spk, mem, trace_pre, trace_post = _SpikeTorchConv(spk, mem, input_=input_)
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.spk, self.mem, self.trace_pre, self.trace_post = \
                _SpikeTorchConv(self.spk, self.mem, self.trace_pre, self.trace_post, input_=input_)

        # TO-DO: alternatively, we could do torch.exp(-1 / self.beta.clamp_min(0)),
        # giving actual time constants instead of values in [0, 1] as initial beta
        # beta = self.beta.clamp(0, 1)

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

            # TODO: input should be spike
            self.trace_pre = self.decay_pre * self.trace_pre + self.scaling_pre * input_
            self.trace_post = self.decay_post * self.trace_post + self.scaling_post * self.spk

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem, self.trace_pre, self.trace_post
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

        uni_spk = self.spk if self.init_hidden else spk

        if uni_spk:
            self.V -= self.trace_post
        if input_:
            self.V += self.trace_pre

    def _base_state_function(self, input_, spk, mem):
        base_fn = self.beta.clamp(0, 1) * mem + input_ + self.V * spk
        return base_fn

    def _build_state_function(self, input_, spk, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function(input_, spk, mem)
                - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, mem
            ) - self.reset * self._base_state_function(input_, spk, mem)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, spk, mem)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_ + self.V * self.spk
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

    def _stdpleaky_forward_cases(self, spk, mem):
        if mem is not False or spk is not False:
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
            if isinstance(cls.instances[layer], RLeaky):
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)


# class NeuronSTDP(nn.Module):
#
#     def __init__(
#         self,
#         beta,
#         trace_decay,
#         trace_enable=True,
#         trace_scaling=1.0,
#         threshold=1.0,
#     ):
#         super().__init__()
#
#         self.beta = beta
#         self.threshold = threshold
#         self.mem = self.register_buffer("mem", torch.Tensor())
#         self.out_spikes = self.register_buffer("out_spikes", torch.Tensor())
#         self.trace_decay = trace_decay
#         self.trace_scaling = trace_scaling
#         if trace_enable:
#             self.register_buffer("trace", torch.Tensor())
#
#     def forward(self, input_,):
#         heaviside = Heaviside.apply
#         n_batches, time_steps, n_neurons = input_.shape
#         for t in range(time_steps):
#             self.mem = self.beta * self.mem + input_[:, t, :]
#             self.out_spikes[:, t, :] = heaviside(self.mem)
#             self.mem = self.mem - self.out_spikes[:, t, :]
#             self.trace = self.trace_decay * self.trace + self.trace_scaling * self.out_spikes[:, t, :]
#
#
# class STDPLinear():
#     def __init__(
#         self,
#         source: NeuronSTDP,
#         target: NeuronSTDP,
#     ):
#         super().__init__()
#
#         self.source = source
#         self.target = target
#         self.weight = self.register_buffer("weight", torch.Tensor())
#
#     def weight_update(self):
#         n_batches, time_steps, n_neurons = self.source.out_spikes.shape
#         for t in range(time_steps):
#             if self.source.out_spikes[:, t, :]:
#                 self.weight -= self.target.trace[:, t, :]
#             if self.target.out_spikes[:, t, :]:
#                 self.weight += self.source.trace[:, t, :]


