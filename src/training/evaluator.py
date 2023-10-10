import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


class Evaluator:
    _METRICS = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "confusion_matrix": confusion_matrix,
    }

    def _get_predictions(
        self,
        test_dataset,
        model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model.eval()

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                y_true.append(labels.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        return y_true, y_pred

    def evaluate(self, test_dataset, model):
        metrics = {}

        y_true, y_pred = self._get_predictions(test_dataset, model)

        for metric in Evaluator._METRICS:
            metrics[metric] = Evaluator._METRICS[metric](y_true, y_pred)

        return metrics
