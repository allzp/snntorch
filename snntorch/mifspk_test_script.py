import snntorch as snn
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.mif1 = snn.MIFSPK()

    def forward(self, x):
        a, I, v, x1, x2, G1, G2 = self.mif1.init_mifspiking()

        # Record the final layer
        spk_rec = []
        a_rec = []
        I_rec = []
        v_rec = []
        x1_rec = []
        x2_rec = []
        G1_rec = []
        G2_rec = []

        for step in range(num_steps):
            spk, a, I, v, x1, x2, G1, G2 = self.mif1(x[step], a, I, v, x1, x2, G1, G2)
            spk_rec.append(spk)
            a_rec.append(a)
            I_rec.append(I)
            v_rec.append(v)
            x1_rec.append(x1)
            x2_rec.append(x2)
            G1_rec.append(G1)
            G2_rec.append(G2)

        return torch.stack(spk_rec, dim=0), torch.stack(a_rec, dim=0), torch.stack(I_rec, dim=0), \
               torch.stack(v_rec, dim=0), torch.stack(x1_rec, dim=0), torch.stack(x2_rec, dim=0), \
               torch.stack(G1_rec, dim=0), torch.stack(G2_rec, dim=0)



net = Net().to(device)
num_steps = 3
inp = torch.Tensor([[2], [0], [2]])
spk_rec, a_rec, I_rec, v_rec, x1_rec, x2_rec, G1_rec, G2_rec = net(inp)
print(spk_rec)
print(a_rec)
print(I_rec)
print(v_rec)
print(x1_rec)
print(x2_rec)
print(G1_rec)
print(G2_rec)
