import snntorch as snn
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

global_error = 2

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.stdp1 = snn.STDPLeaky(
            in_num=1,
            out_num=2,
            beta=0.5,
            decay_pre=0.5,
            decay_post=0.5,
            learn_V=True,)
            # TODO: if set learn_V=True then not working

    def forward(self, x):
        self.stdp1.V.weight.data = torch.Tensor([[0.8], [0]])

        # Record the final layer
        spk_rec = []
        for step in range(num_steps):
            spk1 = self.stdp1(x[step])
            spk_rec.append(spk1)

        return torch.stack(spk_rec, dim=0)


def forward_hook(module, module_inp, module_out):
    print(module_inp)
    print(module.trace_pre)
    print(module_out)
    print(module.trace_post)
    print('weight_before', module.V.weight)
    for i in range(module.in_num):
        for j in range(module.out_num):
            if module_inp[i] != 0:
                module.V.weight[j, i] -= module.trace_post[j] * global_error
            if module_out[j] != 0:
                module.V.weight[j, i] += module.trace_pre[i] * global_error
    print('weight_after', module.V.weight)


net = Net().to(device)
net.stdp1.register_forward_hook(forward_hook)
num_steps = 3
inp = torch.Tensor([[2], [0], [2]])
spk_rec = net(inp)
