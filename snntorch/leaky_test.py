import snntorch as snn
import torch

leaky1 = snn.Leaky(beta=0.5)

mem1 = leaky1.init_leaky()

inp = torch.Tensor([1.2])

step=0
spk1, mem1 = leaky1(inp, mem1)
print('------', step, '------')
print('in', inp)
print('out_spk', spk1)
print('mem', mem1)

step=1
spk1, mem1 = leaky1(torch.Tensor([0]), mem1)
print('------', step, '------')
print('in', inp)
print('out_spk', spk1)
print('mem', mem1)
