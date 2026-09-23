import gin
import torchmetrics
import torch
from torchmetrics import Metric
from torchmetrics.regression import PearsonCorrCoef

from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score

gin.external_configurable(BinaryAccuracy, module="tm")
gin.external_configurable(BinaryF1Score, module="tm")

gin.external_configurable(torchmetrics.Accuracy, module="tm")
gin.external_configurable(torchmetrics.AUROC, module="tm")
gin.external_configurable(torchmetrics.Precision, module="tm")
gin.external_configurable(torchmetrics.F1Score, module="tm")
gin.external_configurable(torchmetrics.Recall, module="tm")
gin.external_configurable(torchmetrics.AveragePrecision, module="tm")
gin.external_configurable(torchmetrics.MeanSquaredError, module="tm")
gin.external_configurable(torchmetrics.MeanAbsoluteError, module="tm")
gin.external_configurable(torchmetrics.R2Score, module="tm")
gin.external_configurable(PearsonCorrCoef, module="tm")

class RootMeanSquaredError(torchmetrics.MeanSquaredError):
    def __init__(self):
        super().__init__(squared=False)

gin.external_configurable(RootMeanSquaredError, module="tm")


class RelativeRootMeanSquaredError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("squared_error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        squared_error = torch.sum((preds - target) ** 2)
        target_sum = torch.sum(target)
        n_obs = torch.numel(target)

        self.squared_error_sum += squared_error
        self.target_sum += target_sum
        self.n_obs += n_obs

    def compute(self):
        rmse = torch.sqrt(self.squared_error_sum / self.n_obs)
        mean_target = self.target_sum / self.n_obs
        relative_rmse = rmse / mean_target if mean_target != 0 else torch.tensor(float("inf"))
        return relative_rmse

gin.external_configurable(RelativeRootMeanSquaredError, module="tm")
