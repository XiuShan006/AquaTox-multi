import gin
import torch
from torch import nn


@gin.configurable()
class OptimizerBase:

    def __init__(self, cls_name: str, **kwargs):
        self.cls_name = cls_name
        self.optimizer: torch.optim.Optimizer = ...
        self.kwargs = kwargs

    def initialize(
        self,
        model: nn.Module | None = None,
        parameter_groups: list[dict] | None = None,
    ):
        if (model is None) == (parameter_groups is None):
            raise ValueError("Provide exactly one of model or parameter_groups")
        parameters = model.parameters() if model is not None else parameter_groups
        self.optimizer = getattr(torch.optim, self.cls_name)(parameters, **self.kwargs)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
