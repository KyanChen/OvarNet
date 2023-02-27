import random
import matplotlib.pyplot as plt
from torch.nn import RNN
import torch

rnn = RNN(input_size=3, hidden_size=32, num_layers=5, batch_first=True)
fc = torch.nn.Linear(32, 1)

x1 = torch.linspace(1, 1000, steps=10000)
x2 = torch.linspace(1, 1000, steps=10000)/2
x3 = torch.linspace(1, 1000, steps=10000)/5
y = [0]
for i in range(1, len(x1)):
    y.append(x1[i-1]*x1[i]/(x3[i-1]*torch.sqrt(x3[i])) - x1[i-1]/x1[i] + x2[i]/x2[i-1])
y = torch.tensor(y)
x1 = x1/max(x1)
x2 = x2/max(x2)
x3 = x3/max(x3)
y = y/max(y)
y = y + 0.1*(torch.rand(len(y))-0.5)

data_set = []
for i in range(1, len(x1)):
    input_tmp = torch.tensor([[x1[i-1], x2[i-1], x3[i-1]], [x1[i], x2[i], x3[i]]])
    data_set.append((input_tmp, y[i]))
batch_size = 10

optimizer = torch.optim.SGD([{'params': rnn.parameters()}, {'params': fc.parameters()}], lr=1e-3, momentum=0.9)
random.shuffle(data_set)
train_set = data_set[:-100]
val_set = data_set[-100:]
plt_loss = []
for i_epoch in range(50):
    random.shuffle(train_set)
    losses = []
    for batch_id in range(len(train_set)//batch_size):
        input = []
        output = []
        for input_tmp, output_tmp in train_set[batch_id*batch_size: (batch_id+1)*batch_size]:
            input.append(input_tmp)
            output.append(output_tmp)
        input = torch.stack(input)
        output = torch.stack(output).view(-1, 1)
        o, h = rnn(input)
        o = fc(o[:, -1, :])
        loss = torch.nn.functional.mse_loss(o, output)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss)
        optimizer.step()
    plt_loss.append(torch.tensor(losses).mean())
plt_loss = torch.tensor(plt_loss).numpy()

plt.plot(plt_loss)
plt.show()

torch.save(rnn, 'rnn.pth')
torch.save(fc, 'fc.pth')
rnn = torch.load('rnn.pth')
fc = torch.load('fc.pth')
rnn = rnn.eval()
for input, output in val_set:
    input = input.unsqueeze(0)
    o, h = rnn(input)
    o = fc(o[:, -1, :])
    print(o, '/', output)