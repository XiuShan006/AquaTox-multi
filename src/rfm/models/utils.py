from typing import Iterable, Tuple

import torch
from torch_geometric.utils import to_dense_batch
from torchtyping import TensorType


def to_indices(counts: TensorType[int]) -> TensorType[int]:
    indices = torch.arange(len(counts), device=counts.device)
    return torch.repeat_interleave(indices, counts).long()


def to_dense_embeddings(
        embeddings: torch.Tensor,
        counts: Iterable[int],
        fill_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = (
        torch.tensor(counts, device=embeddings.device)
        if not isinstance(counts, torch.Tensor)
        else counts
    )
    indices = torch.arange(len(counts), device=counts.device)
    batch = torch.repeat_interleave(indices, counts).long()
    return to_dense_batch(
        embeddings, batch, fill_value=fill_value
    )


import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
