# Week 1: 从零手写神经网络
> 纯 Python + numpy 实现 MLP，在 MNIST 上训练手写数字分类器。
> **目的**：理解反向传播的每一行代码，为后续使用 PyTorch/TF 打地基。
## 文件说明
| 文件 | 说明 |
|------|------|
| `net.py` | 神经网络核心：Linear 层、ReLU、Softmax+交叉熵、MLP、梯度检查工具 |
| `data.py` | MNIST 自动下载 + 加载 + mini-batch 生成 |
| `train.py` | 训练入口，支持命令行参数 |
| `grad_check.py` | 梯度检查脚本 |
| `explore.py` | 超参扫描，对比不同配置效果 |
## 快速开始
```bash
# 1. 训练一个默认网络 (784→256→128→10, lr=0.01, 10 epochs)
python train.py
# 2. 运行梯度检查 (验证反向传播正确性)
python grad_check.py
# 3. 超参探索
python explore.py
# 4. 自定义训练
python train.py --lr 0.1 --hidden1 512 --hidden2 256 --epochs 20 --batch_size 128
```
## 学习路线 (5-7 天)
### Day 1: 读 `net.py`，理解前向传播
1. 打开 `net.py`，先看 `Linear` 类
2. 理解 `forward` 中的 `x @ W.T + b`
3. 看 `ReLU` 和 `SoftmaxCrossEntropy`
4. **练习**：手算一个 batch_size=2, in=4, out=3 的前向传播，验证输出形状
### Day 2: 理解反向传播
1. 看 `Linear.backward` — 对照链式法则
2. 看 `ReLU.backward` — 为什么只有 mask 操作？
3. 看 `SoftmaxCrossEntropy.backward` — 为什么是 probs - 1？
4. **练习**：在纸上手推一次 softmax + cross-entropy 的梯度
### Day 3: 运行梯度检查
```bash
python grad_check.py
```
梯度检查用数值微分 `(f(x+h)-f(x-h))/2h` 验证你的反向传播是否正确。
熟悉 `gradient_check()` 的实现，理解为什么它能捕获 bug。
### Day 4: 训练 & 观察
```bash
python train.py
```
观察训练曲线：
- 训练 loss 是否稳定下降？
- 训练精度和测试精度的差距？
### Day 5: 超参探索
```bash
python explore.py
```
回答输出中的三个问题：
- 不同学习率的影响？
- 网络宽度对过拟合的影响？
- 参数量与精度的关系？
### Day 6-7: 动手修改
尝试以下修改之一：
- **SGD Momentum**：在 `Linear.update` 中加入动量项
- **Dropout**：在 MLP 中加入 Dropout 层
- **Adam 优化器**：实现 Adam 代替 SGD
- **更深的网络**：尝试 4-5 层网络，观察训练是否变困难
## 预期结果
默认配置 (784→256→128→10, lr=0.01, 10 epochs) 应达到:
- 训练精度: ~97%
- 测试精度: ~96%
## 常见问题
**Q: 训练 loss 不降？**
A: 检查学习率是否太大/太小，或权重初始化方式。
**Q: 梯度检查失败？**
A: 大概率 backward 实现有 bug。检查 `dW` 的维度和求和方向。
**Q: 训练太慢？**
A: CPU 跑 MNIST 60000 张确实需要时间。可以把 `--batch_size` 调大，或者只取子集训练。
## 下一步 (Week 2)
有了这个基础，进入 PyTorch 重新实现同样的 MLP，对比：
- PyTorch 版代码量少了多少？
- PyTorch 的自动求导和你手写的 backward 结果一致吗？
- GPU 训练比 CPU 快多少？
准备好了告诉我，我帮你搭第 2 周的脚手架。
