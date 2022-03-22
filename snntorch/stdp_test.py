import snntorch as snn
import torch

# lif1 = snn.Leaky(beta=0.9)
#
# mem1 = lif1.init_leaky()
#
# inp = torch.Tensor([0.1, 0.2])
# spk1, mem1 = lif1(inp, mem1)

stdp1 = snn.STDPLeaky(
        in_num=1,
        out_num=2,
        beta=0.5,
        decay_pre=0.5,
        decay_post=0.5,)

spk1, mem1, pre1, post1 = stdp1.init_stdpleaky()
stdp1.V.weight.data = torch.Tensor([[0.4], [0]])

inp = torch.Tensor([3])
# for step in range(2):
step=0
spk1, mem1, pre1, post1 = stdp1(inp, spk1, mem1, pre1, post1)
print('------', step, '------')
print('in_spk', inp)
print('out_spk', spk1)
print('mem', mem1)
print('pre', pre1)
print('post', post1)
print('weight', stdp1.V.weight)

step=1
spk1, mem1, pre1, post1 = stdp1(torch.Tensor([0]), spk1, mem1, pre1, post1)
print('------', step, '------')
print('in_spk', 0)
print('out_spk', spk1)
print('mem', mem1)
print('pre', pre1)
print('post', post1)
print('weight', stdp1.V.weight)
