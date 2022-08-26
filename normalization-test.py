import torch
import torch.nn as nn
import sys

import matplotlib.pyplot as plt

n_input, n_hidden, n_out, batch_size, learning_rate, total_data_count = 2, 3, 1, 5, 0.01, 20

torch.manual_seed(2022)
data_x=torch.randn(total_data_count, n_input)
data_y = (torch.rand(size=(total_data_count, 1)) < 0.5).float()
print(data_x)
print(data_y)

model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      #nn.BatchNorm1d(n_hidden),
                      nn.LayerNorm(n_hidden),
                      nn.Linear(n_hidden, n_out),
                      nn.Sigmoid())

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
for epoch in range(1000):
    for i in range(0, total_data_count, batch_size):
        indices = torch.tensor(range(i,i+batch_size))
        data_x_batch=torch.index_select(data_x, 0, indices)
        data_y_batch=torch.index_select(data_y, 0, indices)


        pred_y = model(data_x)
        loss = loss_function(pred_y, data_y)
        print(loss.item())
        losses.append(loss.item())


        loss.backward()
        optimizer.step()
    model.zero_grad()


plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()
