import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
preds = [
  dict(
    boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
    scores=torch.tensor([0.536]),
    labels=torch.tensor([0]),
  ),
dict(
        boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
        scores=torch.tensor([0.536]),
        labels=torch.tensor([1]),
    )
]
preds2 = [
    dict(
        boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
        scores=torch.tensor([0.536]),
        labels=torch.tensor([1]),
    )
]
target = [
  dict(
    boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
    labels=torch.tensor([0]),
  ),
dict(
    boxes=torch.tensor([[258.0, 41.0, 562.0, 285.0]]),
    labels=torch.tensor([1]),
  )
]

if __name__ == '__main__':

    metric = MeanAveragePrecision()
    metric.update(preds2, target)
    from pprint import pprint
    pprint(metric.compute())
