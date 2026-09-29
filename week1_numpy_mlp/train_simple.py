"""
简单版训练 — 不需要下载 MNIST，用随机数据演示训练过程。
虽然不会学到真正的数字识别，但你能看到 loss 下降的趋势。
"""
import numpy as np

# ── 生成随机数据 ──
N = 500  # 样本数
x = np.random.randn(N, 10)      # 10 维输入
W_true = np.random.randn(10, 3) # 一个"真实"的权重
y = x @ W_true + np.random.randn(N, 3) * 0.1  # 带噪声的标签

# ── 定义网络 ──
class Linear:
    def __init__(self, in_d, out_d):
        self.W = np.random.randn(out_d, in_d) * 0.01
        self.b = np.zeros(out_d)
    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b
    def backward(self, dout):
        self.dW = dout.T @ self.x
        self.db = dout.sum(axis=0)
        return dout @ self.W
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

layer1 = Linear(10, 20)
layer2 = Linear(20, 3)

# ── 训练 ──
lr = 0.01
for step in range(100):
    # 前向
    h = layer1.forward(x)
    h = np.maximum(h, 0)  # ReLU
    pred = layer2.forward(h)

    # 计算 loss（均方误差）
    loss = ((pred - y) ** 2).mean()

    # 反向
    dout = 2 * (pred - y) / N
    dout = layer2.backward(dout)
    dout[dout <= 0] = 0  # ReLU 回传
    layer1.backward(dout)

    # 更新
    layer1.update(lr)
    layer2.update(lr)

    if step % 20 == 0:
        print(f"Step {step:3d}, Loss: {loss:.4f}")

print(f"\n训练完成! 最终 loss: {loss:.4f}")
print(f"loss 下降了吗? {'是的!' if loss < 1 else '不对, 检查代码'}")
