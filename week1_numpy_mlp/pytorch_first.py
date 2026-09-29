import torch
import torch.nn as nn

N = 500
x = torch.randn(N, 10)
W_true = torch.randn(10, 3)
y = x @ W_true + torch.randn(N, 3) * 0.1

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 3),
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step:3d}, Loss: {loss.item():.4f}")

print(f"Done! Final loss: {loss.item():.4f}")
