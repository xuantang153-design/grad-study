import numpy as np
from typing import Callable, List, Tuple, Dict


class Linear:
    """????: y = x @ W.T + b"""

    def __init__(self, in_features: int, out_features: int):
        # He??? (???ReLU)
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((out_features,))
        self.x: np.ndarray = None
        self.dW: np.ndarray = None
        self.db: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = dout.T @ self.x
        self.db = dout.sum(axis=0)
        return dout @ self.W

    def update(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def zero_grad(self):
        self.dW = None
        self.db = None

    def parameters(self) -> List[np.ndarray]:
        return [self.W, self.b]

    def gradients(self) -> List[np.ndarray]:
        return [self.dW, self.db]


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class SoftmaxCrossEntropy:
    """Softmax + ????? (???????????)"""

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        batch = logits.shape[0]
        logits_stable = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits_stable)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.log(self.probs[np.arange(batch), labels] + 1e-8).mean()
        return loss

    def backward(self, labels: np.ndarray) -> np.ndarray:
        batch = labels.shape[0]
        dout = self.probs.copy()
        dout[np.arange(batch), labels] -= 1
        dout /= batch
        return dout


class MLP:
    """?????"""

    def __init__(self, layer_dims: List[int]):
        self.layers: List = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                self.layers.append(ReLU())
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update(self, lr: float):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    def train_step(self, x: np.ndarray, labels: np.ndarray, lr: float) -> float:
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, labels)
        dout = self.loss_fn.backward(labels)
        self.backward(dout)
        self.update(lr)
        self.zero_grad()
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return logits.argmax(axis=1)

    def accuracy(self, x: np.ndarray, labels: np.ndarray) -> float:
        preds = self.predict(x)
        return (preds == labels).mean()


# ??????


def gradient_check(
    net: MLP,
    x: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 1e-5,
    tol: float = 1e-6,
) -> List[Dict]:
    """???? vs ???????????"""
    results = []
    logits = net.forward(x)
    loss_fn = net.loss_fn
    _ = loss_fn.forward(logits, labels)
    dout = loss_fn.backward(labels)
    net.backward(dout)

    param_idx = 0
    for layer in net.layers:
        if not hasattr(layer, "parameters"):
            continue
        params = layer.parameters()
        grads = layer.gradients()
        for p, g in zip(params, grads):
            grad_num = np.zeros_like(p)
            it = np.nditer(p, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                old = p[idx]
                p[idx] = old + epsilon
                logits_p = net.forward(x)
                loss_p = loss_fn.forward(logits_p, labels)
                p[idx] = old - epsilon
                logits_n = net.forward(x)
                loss_n = loss_fn.forward(logits_n, labels)
                grad_num[idx] = (loss_p - loss_n) / (2 * epsilon)
                p[idx] = old
                it.iternext()

            diff = np.abs(grad_num - g)
            max_diff = diff.max()
            denominator = np.maximum(np.abs(grad_num), np.abs(g)).max()
            rel_error = max_diff / (denominator + 1e-8)
            results.append(
                {
                    "layer_idx": param_idx,
                    "shape": p.shape,
                    "max_abs_diff": max_diff,
                    "rel_error": rel_error,
                    "passed": rel_error < tol,
                }
            )
            param_idx += 1
    return results
