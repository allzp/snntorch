from warnings import warn
import torch
import torch.nn as nn


__all__ = [
    "SpikingNeuron",
    "LIF",
    "MIFSpiking",
    "_SpikeTensor",
    "_SpikeTorchConv",
]

dtype = torch.float


class SpikingNeuron(nn.Module):
    """Parent class for spiking neuron models."""

    instances = []
    """Each :mod:`snntorch.SpikingNeuron` neuron (e.g., :mod:`snntorch.Synaptic`) will populate the :mod:`snntorch.SpikingNeuron.instances` list with a new entry.
    The list is used to initialize and clear neuron states when the argument `init_hidden=True`."""

    reset_dict = {
        "subtract": 0,
        "zero": 1,
        "none": 2,
    }

    def __init__(
        self,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
    ):
        super(SpikingNeuron, self).__init__()

        SpikingNeuron.instances.append(self)
        self.init_hidden = init_hidden
        self.inhibition = inhibition
        self.output = output

        self._snn_cases(reset_mechanism, inhibition)
        self._snn_register_buffer(threshold, learn_threshold, reset_mechanism)
        self._reset_mechanism = reset_mechanism

        # TO-DO: Heaviside --> STE; needs a tutorial change too?
        if spike_grad is None:
            self.spike_grad = self.Heaviside.apply
        else:
            self.spike_grad = spike_grad

        self.state_quant = state_quant

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""

        if self.state_quant:
            mem = self.state_quant(mem)

        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)

        return spk

    def fire_inhibition(self, batch_size, mem):
        """Generates spike if mem > threshold, only for the largest membrane. All others neurons will be inhibited for that time step.
        Returns spk."""
        mem_shift = mem - self.threshold
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
        # reset = spk.clone().detach()

        return spk

    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()

        return reset

    def _snn_cases(self, reset_mechanism, inhibition):
        self._reset_cases(reset_mechanism)

        if inhibition:
            warn(
                "Inhibition is an unstable feature that has only been tested for dense (fully-connected) layers. Use with caution!",
                UserWarning,
            )

    def _reset_cases(self, reset_mechanism):
        if (
            reset_mechanism != "subtract"
            and reset_mechanism != "zero"
            and reset_mechanism != "none"
        ):
            raise ValueError(
                "reset_mechanism must be set to either 'subtract', 'zero', or 'none'."
            )

    def _snn_register_buffer(self, threshold, learn_threshold, reset_mechanism):
        """Set variables as learnable parameters else register them in the buffer."""

        self._threshold_buffer(threshold, learn_threshold)

        # reset buffer
        try:
            # if reset_mechanism_val is loaded from .pt, override reset_mechanism
            if torch.is_tensor(self.reset_mechanism_val):
                self.reset_mechanism = list(SpikingNeuron.reset_dict)[
                    self.reset_mechanism_val
                ]
        except AttributeError:
            # reset_mechanism_val has not yet been created, create it
            self._reset_mechanism_buffer(reset_mechanism)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)

    def _reset_mechanism_buffer(self, reset_mechanism):
        """Assign mapping to each reset mechanism state.
        Must be of type tensor to store in register buffer. See reset_dict for mapping."""
        reset_mechanism_val = torch.as_tensor(SpikingNeuron.reset_dict[reset_mechanism])
        self.register_buffer("reset_mechanism_val", reset_mechanism_val)

    def _V_register_buffer(self, V, learn_V):
        if not isinstance(V, torch.Tensor):
            V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @property
    def reset_mechanism(self):
        """If reset_mechanism is modified, reset_mechanism_val is triggered to update.
        0: subtract, 1: zero, 2: none."""
        return self._reset_mechanism

    @reset_mechanism.setter
    def reset_mechanism(self, new_reset_mechanism):
        self._reset_cases(new_reset_mechanism)
        self.reset_mechanism_val = torch.as_tensor(
            SpikingNeuron.reset_dict[new_reset_mechanism]
        )
        self._reset_mechanism = new_reset_mechanism

    @classmethod
    def init(cls):
        """Removes all items from :mod:`snntorch.SpikingNeuron.instances` when called."""
        cls.instances = []

    @staticmethod
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are global variables."""
        for state in args:
            state = torch.zeros_like(state)

    @staticmethod
    class Heaviside(torch.autograd.Function):
        """Default spiking function for neuron.

        **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

        **Backward pass:** Heaviside step function shifted.

        .. math::

            \\frac{∂S}{∂U}=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

        Although the backward pass is clearly not the analytical solution of the forward pass, this assumption holds true on the basis that a reset necessarily occurs after a spike is generated when :math:`U ≥ U_{\\rm thr}`."""

        @staticmethod
        def forward(ctx, input_):
            out = (input_ > 0).float()
            ctx.save_for_backward(out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (out,) = ctx.saved_tensors
            grad = grad_output * out
            return grad


class LIF(SpikingNeuron):
    """Parent class for leaky integrate and fire neuron models."""

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
    ):
        super().__init__(
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self._lif_register_buffer(
            beta,
            learn_beta,
        )
        self._reset_mechanism = reset_mechanism

        # TO-DO: Heaviside --> STE; needs a tutorial change too?
        if spike_grad is None:
            self.spike_grad = self.Heaviside.apply
        else:
            self.spike_grad = spike_grad

    def _lif_register_buffer(
        self,
        beta,
        learn_beta,
    ):
        """Set variables as learnable parameters else register them in the buffer."""
        self._beta_buffer(beta, learn_beta)

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    def _V_register_buffer(self, V, learn_V):
        if not isinstance(V, torch.Tensor):
            V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @staticmethod
    def init_leaky():
        """
        Used to initialize mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        mem = _SpikeTensor(init_flag=False)

        return mem

    @staticmethod
    def init_rleaky():
        """
        Used to initialize spk and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return spk, mem

    @staticmethod
    def init_stdpleaky():
        """
        Used to initialize spk and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)
        trace_pre = _SpikeTensor(init_flag=False)
        trace_post = _SpikeTensor(init_flag=False)

        return spk, mem, trace_pre, trace_post

    @staticmethod
    def init_synaptic():
        """Used to initialize syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """

        syn = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return syn, mem

    @staticmethod
    def init_rsynaptic():
        """
        Used to initialize spk, syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        syn = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return spk, syn, mem

    @staticmethod
    def init_lapicque():
        """
        Used to initialize mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """

        return LIF.init_leaky()

    @staticmethod
    def init_alpha():
        """Used to initialize syn_exc, syn_inh and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        syn_exc = _SpikeTensor(init_flag=False)
        syn_inh = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return syn_exc, syn_inh, mem


class MIFSpiking(SpikingNeuron):

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
        super().__init__(
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self._mifspiking_register_buffer(R_on, R_off, v_on, v_off, tau, tau_alpha, E1, E2, C, k_th)
        self._reset_mechanism = reset_mechanism

        # TO-DO: Heaviside --> STE; needs a tutorial change too?
        if spike_grad is None:
            self.spike_grad = self.Heaviside.apply
        else:
            self.spike_grad = spike_grad

    def _mifspiking_register_buffer(self, R_on, R_off, v_on, v_off, tau, tau_alpha, E1, E2, C, k_th):
        """Set variables as learnable parameters else register them in the buffer."""
        self._mifspiking_buffer(R_on, R_off, v_on, v_off, tau, tau_alpha, E1, E2, C, k_th)

    def _mifspiking_buffer(self, R_on, R_off, v_on, v_off, tau, tau_alpha, E1, E2, C, k_th):
        if not isinstance(R_on, torch.Tensor):
            R_on = torch.as_tensor(R_on)  # TODO: or .tensor() if no copy
        self.register_buffer("R_on", R_on)
        if not isinstance(R_off, torch.Tensor):
            R_off = torch.as_tensor(R_off)
        self.register_buffer("R_off", R_off)
        if not isinstance(v_on, torch.Tensor):
            v_on = torch.as_tensor(v_on)
        self.register_buffer("v_on", v_on)
        if not isinstance(v_off, torch.Tensor):
            v_off = torch.as_tensor(v_off)
        self.register_buffer("v_off", v_off)
        if not isinstance(tau, torch.Tensor):
            tau = torch.as_tensor(tau)
        self.register_buffer("tau", tau)
        if not isinstance(tau_alpha, torch.Tensor):
            tau_alpha = torch.as_tensor(tau_alpha)
        self.register_buffer("tau_alpha", tau_alpha)
        if not isinstance(E1, torch.Tensor):
            E1 = torch.as_tensor(E1)
        self.register_buffer("E1", E1)
        if not isinstance(E2, torch.Tensor):
            E2 = torch.as_tensor(E2)
        self.register_buffer("E2", E2)
        if not isinstance(C, torch.Tensor):
            C = torch.as_tensor(C)
        self.register_buffer("C", C)
        if not isinstance(k_th, torch.Tensor):
            k_th = torch.as_tensor(k_th)
        self.register_buffer("k_th", k_th)

    # TODO: what is V for in SpikingNeuron
    # def _V_register_buffer(self, V, learn_V):
    #     if not isinstance(V, torch.Tensor):
    #         V = torch.as_tensor(V)
    #     if learn_V:
    #         self.V = nn.Parameter(V)
    #     else:
    #         self.register_buffer("V", V)

    @staticmethod
    def init_mifspiking():
        """
        Used to initialize spk and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        a = _SpikeTensor(init_flag=False)
        I = _SpikeTensor(init_flag=False)
        v = _SpikeTensor(init_flag=False)
        x1 = _SpikeTensor(init_flag=False)
        x2 = _SpikeTensor(init_flag=False)
        G1 = _SpikeTensor(init_flag=False)
        G2 = _SpikeTensor(init_flag=False)

        return a, I, v, x1, x2, G1, G2


class _SpikeTensor(torch.Tensor):
    """Inherits from torch.Tensor with additional attributes.
    ``init_flag`` is set at the time of initialization.
    When called in the forward function of any neuron, they are parsed and replaced with a torch.Tensor variable.
    """

    @staticmethod
    def __new__(cls, *args, init_flag=False, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *args,
        init_flag=True,
    ):
        # super().__init__() # optional
        self.init_flag = init_flag


def _SpikeTorchConv(*args, input_):
    """Convert SpikeTensor to torch.Tensor of the same size as ``input_``."""

    states = []
    # if len(input_.size()) == 0:
    #     _batch_size = 1  # assume batch_size=1 if 1D input
    # else:
    #     _batch_size = input_.size(0)
    if (
        len(args) == 1 and type(args) is not tuple
    ):  # if only one hidden state, make it iterable
        args = (args,)
    for arg in args:
        if arg.is_cuda:
            arg = arg.to("cpu")
        arg = torch.Tensor(arg)  # wash away the SpikeTensor class
        arg = torch.zeros_like(input_, requires_grad=True)
        states.append(arg)
    if len(states) == 1:  # otherwise, list isn't unpacked
        return states[0]

    return states
