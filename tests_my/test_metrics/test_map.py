from torch import BoolTensor, IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

preds = [
    dict(
        boxes=Tensor([
            [258.0, 41.0, 606.0, 285.0],
            [0.0, 41.0, 66.0, 285.0],
            [12.0, 41.0, 66.0, 200.0],
        ]),
        scores=Tensor([0.536, 0.7, 0.6]),
        labels=IntTensor([0, 1, 0]),
    )
]

target = [
    dict(
        boxes=Tensor([[214.0, 41.0, 562.0, 285.0], [12.0, 41.0, 66.0, 210.0]]),
        labels=IntTensor([0, 0]),
    )
]

if __name__ == "__main__":
    # Initialize metric
    metric = MeanAveragePrecision(
        iou_type="bbox",
        # iou_thresholds=[0.5],
        max_detection_thresholds=[100, 500, 1000],
        class_metrics=True,
        compute_on_cpu=True,
        sync_on_compute=False
    )

    # Update metric with predictions and respective ground truth
    metric.update(preds, target)
    metric.update(preds, target)

    # Compute the results
    # import pdb
    # pdb.set_trace()
    result = metric.compute()
    from pprint import pprint
    pprint(result)