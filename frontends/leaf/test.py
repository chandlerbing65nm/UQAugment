import torch
from frontends.leaf.frontend import Leaf
import numpy as np


if __name__ == '__main__':
    fe = Leaf()
    x = torch.randn(32, 1, 128000)
    print(x.shape)
    o = fe(x)
    print(o.shape)
    # print(o[0][1])