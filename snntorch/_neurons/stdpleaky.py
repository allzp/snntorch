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
        error_modulator=1.0,
        lr_pre=10**(-4),
        lr_post=10**(-4),
        threshold=1.0,
        V_bias=False,
        spike_grad=None,
        init_hidden=True,
        winner=False,
        inh_delay=0,
        inh_V=0,
        inh_tau=0.5,
        learn_beta=False,
        # learn_decay_pre=False,  # TODO: maybe learnable
        # learn_decay_post=False,  # TODO: maybe learnable
        learn_V=False,  # TODO: combine STDP with gradient
        stdp_train=True,  # STDP train flag
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
    ):
        super(STDPLeaky, self).__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            winner,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,

        )

        # TODO: input of STDPLeaky should be spike, not weighted spike

        if self.init_hidden:
            self.spk, self.mem, self.trace_pre, self.trace_post, self.inhibition, self.inh_delay = self.init_stdpleaky()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

        self.decay_pre = decay_pre
        self.lr_pre = lr_pre
        self.decay_post = decay_post
        self.lr_post = lr_post
        self.in_num = in_num
        self.out_num = out_num
        self.V_bias = V_bias
        self.error_modulator = error_modulator
        self.V = nn.Linear(self.in_num, self.out_num, bias=self.V_bias)
        self.V.weight.requires_grad = learn_V
        self.stdp_train = stdp_train
        self.inh_delay = inh_delay,
        self.inh_V = inh_V,
        self.inh_tau = inh_tau,

    def forward(self, input_, spk=False, mem=False, trace_pre=False, trace_post=False, inhibition=False, inh_delay=False):
        if (
            hasattr(spk, "init_flag")
            or hasattr(mem, "init_flag")
            or hasattr(trace_pre, "init_flag")
            or hasattr(trace_post, "init_flag")
            or hasattr(inhibition, "init_flag")
            or hasattr(inh_delay, "init_flag")
        ):  # only triggered on first-pass
            # - spk, mem, trace_post, inhibition, inh_delay have same output size
            spk, mem, trace_post, inhibition, inh_delay = _SpikeTorchConv(
                spk, mem, trace_post, inhibition, inh_delay, input_=torch.empty(self.out_num).to(input_.device)
            )
            # - trace_pre has input size
            trace_pre = _SpikeTorchConv(trace_pre, input_=input_)

        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            # - spk, mem, trace_post, inhibition, inh_delay have same output size
            self.spk, self.mem, self.trace_post, self.inhibition, self.inh_delay = _SpikeTorchConv(
                self.spk, self.mem, self.trace_post, self.inhibition, self.inh_delay, input_=torch.empty(self.out_num)
            )
            # - trace_pre has input size
            self.trace_pre = _SpikeTorchConv(self.trace_pre, input_=input_)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem, inhibition, inh_delay = self.state_fn(input_, spk, mem, inhibition, inh_delay)

            if self.state_quant:
                mem = self.state_quant(mem)

            if self.winner:
                spk = self.fire_winner(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            # - first decay, then add
            trace_pre = self.decay_pre * trace_pre + self.lr_pre * input_
            trace_post = self.decay_post * trace_post + self.lr_post * spk

            # - weight STDP update
            if self.stdp_train:
                with torch.no_grad():
                    input_m = torch.unsqueeze(input_, 1)
                    trace_post_m = torch.unsqueeze(trace_post, 2)
                    # TODO:sum or mean across batch?
                    self.V.weight -= torch.sum(torch.bmm(trace_post_m, input_m), dim=0) * self.error_modulator
                    spk_m = torch.unsqueeze(spk, 2)
                    trace_pre_m = torch.unsqueeze(trace_pre, 1)
                    self.V.weight += torch.sum(torch.bmm(spk_m, trace_pre_m), dim=0) * self.error_modulator

            return spk, mem, trace_pre, trace_post, inhibition, inh_delay

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden:
            # question: why there is mem and self.mem?
            self._stdpleaky_forward_cases(spk, mem, trace_pre, trace_post, inhibition, inh_delay)
            self.reset = self.mem_reset(self.mem)
            self.mem, self.inhibition, self.inh_delay = self.state_fn(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.winner:
                self.spk = self.fire_winner(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

                # - first decay, then add
                self.trace_pre = self.decay_pre * self.trace_pre + self.lr_pre * input_
                self.trace_post = self.decay_post * self.trace_post + self.lr_post * self.spk

                # - weight STDP update
                with torch.no_grad():
                    input_m = torch.unsqueeze(input_, 1)
                    trace_post_m = torch.unsqueeze(self.trace_post, 2)
                    self.V.weight -= torch.sum(torch.bmm(trace_post_m, input_m), dim=0) * self.error_modulator
                    spk_m = torch.unsqueeze(spk, 2)
                    trace_pre_m = torch.unsqueeze(self.trace_pre, 1)
                    self.V.weight += torch.sum(torch.bmm(spk_m, trace_pre_m), dim=0) * self.error_modulator

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem, self.trace_pre, self.trace_post, self.inhibition, self.inh_delay
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

    def _base_state_function(self, input_, spk, mem, inhibition, inh_delay):
        base_fn_inh = self.inh_tau * inhibition + spk
        base_fn_mem = self.beta.clamp(0, 1) * mem + self.V(input_) - (self.inh_V(base_fn_inh) if inh_delay == 0 else 0)
        base_fn_delay = torch.max(inh_delay - 1, torch.tensor([0.]))
        return base_fn_mem, base_fn_inh, base_fn_delay

    def _build_state_function(self, input_, spk, mem, inhibition, inh_delay):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = self._base_state_function(
                input_, spk, mem - self.reset * self.threshold, inhibition, inh_delay
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, spk, mem, inhibition, inh_delay
            ) - self.reset * self._base_state_function(input_, spk, mem, inhibition, inh_delay)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, spk, mem, inhibition, inh_delay)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn_inh = self.inh_tau * self.inhibition + self.spk
        base_fn_mem = self.beta.clamp(0, 1) * self.mem + self.V(input_) - (self.inh_V(base_fn_inh) if self.inh_delay == 0 else 0)
        base_fn_delay = torch.max(self.inh_delay - 1, torch.tensor([0.]))
        return base_fn_mem, base_fn_inh, base_fn_delay

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

    def _stdpleaky_forward_cases(self, spk, mem, trace_pre, trace_post, inhibition, inh_delay):
        if (
            spk is not False
            or mem is not False
            or trace_pre is not False
            or trace_post is not False
            or inhibition is not False
            or inh_delay is not False
        ):
            raise TypeError(
                "When `init_hidden=True`, STDPLeaky expects 1 input argument."
            )

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], STDPLeaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], STDPLeaky):
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)
