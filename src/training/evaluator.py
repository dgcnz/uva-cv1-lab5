import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


class Evaluator:
    _METRICS = {
        "accuracy": lambda x: accuracy_score(x[0], x[1]),
        "precision": lambda x: precision_score(x[0], x[1], average="macro"),
        "recall": lambda x: recall_score(x[0], x[1], average="macro"),
        "f1": lambda x: f1_score(x[0], x[1], average="macro"),
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

                outputs, _ = model(inputs)
                _, preds = torch.max(outputs, 1)

                y_true.append(labels.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        return y_true, y_pred

    def evaluate_and_log(self, test_dataset, model):
        metrics = {}

        y_true, y_pred = self._get_predictions(test_dataset, model)

        for metric in Evaluator._METRICS:
            metrics[metric] = Evaluator._METRICS[metric]((y_true, y_pred))

        wandb.log(metrics)
        y_pred = [i[0] for i in y_pred]
        y_true = [i[0] for i in y_true]

        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=test_dataset.classes,
                )
            }
        )
