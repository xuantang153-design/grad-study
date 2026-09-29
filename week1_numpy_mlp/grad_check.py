"""
Gradient check - verify backpropagation implementation.

Principle: Approximate gradient using numerical differentiation:
    (f(x+h) - f(x-h)) / (2*h)
Compare with analytical backpropagation result.
Relative error < 1e-6 indicates correct implementation.
"""
import numpy as np

from net import MLP, gradient_check
from data import load_mnist


def main():
    print("Loading MNIST (using 1 batch for check)...")
    (x_train, y_train), _ = load_mnist()

    # Use small network + mini-batch for fast check
    net = MLP([784, 32, 16, 10])
    batch_x = x_train[:16]
    batch_y = y_train[:16]

    results = gradient_check(net, batch_x, batch_y)

    print(f"\nGradient check results ({len(results)} tensor(s)):")
    print("-" * 60)
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"  [{status}] Layer {r['layer_idx']} shape={str(r['shape']):>12}  max_diff={r['max_abs_diff']:.2e}  rel_error={r['rel_error']:.2e}")

    if all_passed:
        print("\nResult: Backpropagation is correct.")
    else:
        print("\nResult: Error found! Check backward implementation.")


if __name__ == "__main__":
    main()
