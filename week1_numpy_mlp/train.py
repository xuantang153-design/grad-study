import numpy as np
import argparse
from datetime import datetime

from net import MLP, gradient_check
from data import load_mnist, get_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--check_grad", action="store_true", help="Run gradient check before training")
    args = parser.parse_args()

    print("Loading MNIST ...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"  Train: {x_train.shape[0]} images")
    print(f"  Test:  {x_test.shape[0]} images")

    net = MLP([784, args.hidden1, args.hidden2, 10])
    n_params = sum(
        p.size
        for layer in net.layers
        if hasattr(layer, "parameters")
        for p in layer.parameters()
    )
    print(f"\nNetwork: 784 -> {args.hidden1} -> {args.hidden2} -> 10")
    print(f"Params: {n_params:,}")

    # Gradient check
    if args.check_grad:
        print("\nRunning gradient check ...")
        batch_x = x_train[:32]
        batch_y = y_train[:32]
        results = gradient_check(net, batch_x, batch_y)
        all_passed = True
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                all_passed = False
            print(f"  [{status}] Layer {r['layer_idx']} shape={r['shape']}  rel_error={r['rel_error']:.2e}")
        if all_passed:
            print("  All gradient checks passed!\n")
        else:
            print("  Gradient check failed!\n")

    # Training
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    header = f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Time':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = datetime.now()
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in get_batches(x_train, y_train, args.batch_size):
            loss = net.train_step(x_batch, y_batch, args.lr)
            epoch_loss += loss
            n_batches += 1

        train_acc = net.accuracy(x_train, y_train)
        test_acc = net.accuracy(x_test, y_test)
        elapsed = (datetime.now() - t0).total_seconds()

        print(f"{epoch:>6} | {epoch_loss / n_batches:>10.4f} | {train_acc:>8.4%} | {test_acc:>7.4%} | {elapsed:>6.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
