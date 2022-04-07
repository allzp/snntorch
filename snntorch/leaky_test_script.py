import snntorch as snn
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.leaky1 = snn.Leaky(beta=0.5, init_hidden=True)

    def forward(self, x):

        # Record the final layer
        spk_rec = []

        for step in range(num_steps):
            spk1 = self.leaky1(x[step])

            spk_rec.append(spk1)

        return torch.stack(spk_rec, dim=0)


net = Net().to(device)
num_steps = 3
inp = torch.Tensor([[2], [0], [2]])
spk_rec = net(inp)
print(spk_rec)
