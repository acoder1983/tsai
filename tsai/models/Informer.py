__all__ = ['Informer']

from ..imports import *
from ..utils import *
from .layers import *
from .utils import *


class Informer(Module):
    def __init__(self, seq_len, d, custom_head):
        self.fc = nn.Linear(seq_len, d)

    def forward(self, x):
        return self.fc(x)

