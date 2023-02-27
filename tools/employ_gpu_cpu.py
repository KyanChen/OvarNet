import torch
import torch.nn as nn
import time
import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d = nn.Sequential(*[nn.Conv2d(3, 512, 5)]+[nn.Conv2d(512, 512, 5) for _ in range(5)])

    def forward(self, x):
        return self.conv2d(x)


net1 = Net()
net1 = torch.nn.DataParallel(net1, device_ids=[0, 1, 2, 3])
torch.backends.cudnn.enabled = True
net2 = Net()
print("Start Running...")
while True:
    device = torch.device("cuda:0")
    imgs = torch.rand((16, 3, 512, 512)).contiguous()
    imgs = imgs.to(device)
    net1.to(device)
    outputs = net1(imgs)

    device = 'cpu'
    imgs = torch.rand((20, 3, 512, 512)).contiguous()
    imgs = imgs.to(device)
    net2.to(device)
    _ = net2(imgs)

# t0 = time.time()
# imgs = torch.rand((1, 3, 1024, 1024))
# for _ in tqdm.tqdm(range(4)):
#     imgs = imgs.to(device)
#     outputs = net(imgs)
# print((time.time() - t0)/4)
#
# t0 = time.time()
# imgs = torch.rand((4, 3, 512, 512))
# for _ in tqdm.tqdm(range(4)):
#     imgs = imgs.to(device)
#     outputs = net(imgs)
# print((time.time() - t0)/4)

