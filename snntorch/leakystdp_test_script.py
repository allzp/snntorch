import snntorch as snn
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.stdp1 = snn.STDPLeaky(
            in_num=1,
            out_num=2,
            beta=0.5,
            decay_pre=0.5,
            decay_post=0.5,
            init_hidden=False,
            learn_V=False,)

    def forward(self, x):
        spk1, mem1, trace_pre1, trace_post1 = self.stdp1.init_stdpleaky()

        self.stdp1.V.weight.data = torch.Tensor([[0.8], [0]])

        # Record the final layer
        spk_rec = []
        mem_rec = []
        trace_pre_rec = []
        trace_post_rec = []

        for step in range(num_steps):
            spk1, mem1, trace_pre1, trace_post1 = self.stdp1(x[step], spk1, mem1, trace_pre1, trace_post1)
            # self.stdp1.register_forward_hook(forward_hook)
            spk_rec.append(spk1)
            mem_rec.append(mem1)
            trace_pre_rec.append(trace_pre1)
            trace_post_rec.append(trace_post1)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0), torch.stack(trace_pre_rec, dim=0), torch.stack(trace_post_rec, dim=0)


def forward_hook(module, module_inp, module_out):
    print(module_inp)
    print(module.trace_pre)
    print(module_out)
    print(module.trace_post)
    print('weight_before', module.V.weight)

    # module.V.weight.retain_grad()
    with torch.no_grad():
        for i in range(module.in_num):
            for j in range(module.out_num):
                if module_inp[i] != 0:
                    module.V.weight[j, i] -= module.trace_post[j] #* module.error_modulator
                if module_out[j] != 0:
                    module.V.weight[j, i] += module.trace_pre[i] #* module.error_modulator

    print('weight_after', module.V.weight)


net = Net().to(device)
num_steps = 3
inp = torch.Tensor([[2], [0], [2]])
spk_rec, mem_rec, trace_pre_rec, trace_post_rec = net(inp)
print(spk_rec)
print(mem_rec)
print(trace_pre_rec)
print(trace_post_rec)
