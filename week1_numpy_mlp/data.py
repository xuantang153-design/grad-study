import numpy as np
import struct
import os
from urllib import request
import gzip
from typing import Tuple


MNIST_URLS = {
    "train_images": "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download_mnist(data_dir: str) -> dict:
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for name, url in MNIST_URLS.items():
        fname = url.split("/")[-1]
        out_path = os.path.join(data_dir, fname)
        if not os.path.exists(out_path):
            print(f"Downloading {fname} ...")
            request.urlretrieve(url, out_path)
        paths[name] = out_path
    return paths


def _read_idx(filename: str) -> np.ndarray:
    with gzip.open(filename, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
    return data


def load_mnist(
    data_dir: str = "mnist_data",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    paths = _download_mnist(data_dir)
    x_train = _read_idx(paths["train_images"]).astype(np.float32).reshape(-1, 784) / 255.0
    y_train = _read_idx(paths["train_labels"]).astype(np.int64)
    x_test = _read_idx(paths["test_images"]).astype(np.float32).reshape(-1, 784) / 255.0
    y_test = _read_idx(paths["test_labels"]).astype(np.int64)
    return (x_train, y_train), (x_test, y_test)


def get_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y[idx]
