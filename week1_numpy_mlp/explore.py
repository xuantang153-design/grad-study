"""
超参探索脚本 — 扫描不同超参组合，输出对比表格。
用法:
python explore.py
"""
import numpy as np
from datetime import datetime
from net import MLP
from data import load_mnist, get_batches
def train_and_eval(lr: float, hidden1: int, hidden2: int,
epochs: int = 5, batch_size: int = 64) -> dict:
net = MLP([784, hidden1, hidden2, 10])
(x_train, y_train), (x_test, y_test) = load_mnist()
t0 = datetime.now()
for epoch in range(epochs):
for x_batch, y_batch in get_batches(x_train, y_train, batch_size):
net.train_step(x_batch, y_batch, lr)
elapsed = (datetime.now() - t0).total_seconds()
train_acc = net.accuracy(x_train, y_train)
test_acc = net.accuracy(x_test, y_test)
n_params = sum(p.size for l in net.layers if hasattr(l, 'parameters')
for p in l.parameters())
return {
'lr': lr,
'hidden1': hidden1,
'hidden2': hidden2,
'params': n_params,
'train_acc': train_acc,
'test_acc': test_acc,
'time_s': elapsed,
}
def main():
# ── 待扫描的超参 ──
configs = [
# (lr, hidden1, hidden2)
(0.1,  128, 64),
(0.01, 128, 64),
(0.001, 128, 64),
(0.01, 256, 128),
(0.01, 512, 256),
(0.01, 256, 256),
]
print(f"超参扫描 ({len(configs)} 组, 每 5 epochs)...\n")
print(f"{'lr':>6} | {'架构':>14} | {'参数量':>8} | {'Train Acc':>9} "
f"| {'Test Acc':>8} | {'Time':>6}")
print("-" * 64)
results = []
for lr, h1, h2 in configs:
r = train_and_eval(lr, h1, h2)
results.append(r)
arch = f"{h1}->{h2}"
print(f"{r['lr']:>6} | {arch:>14} | {r['params']:>8,} "
f"| {r['train_acc']:>8.4%} | {r['test_acc']:>7.4%} | {r['time_s']:>5.1f}s")
print("\n几点值得观察:")
print("  - lr=0.1 和 lr=0.01 哪个收敛更快?")
print("  - 增加隐藏层宽度提升测试精度还是只提升训练精度? (过拟合?)")
print("  - 参数量翻倍, 测试精度提升了多少? 值得吗?")
if __name__ == '__main__':
main()
