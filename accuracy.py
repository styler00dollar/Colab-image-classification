import torch


def calculate_accuracy(y_pred, y):
    with torch.no_grad():
        batch_size = y.shape[0]

        _, top_pred = y_pred.topk(k=1)

        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
    return acc_1


from sklearn.metrics import accuracy_score
import numpy as np


def calc_accuracy_gridmix(preds: torch.Tensor, trues: torch.Tensor) -> float:
    lam = trues[-1, :][0].data.cpu().numpy()
    true_label = [trues[0, :].long(), trues[1, :].long()]
    trues = true_label[0] if lam > 0.5 else true_label[1]
    trues = trues.data.cpu().numpy().astype(np.uint8)
    preds = torch.softmax(preds, dim=1).float()
    preds = np.argmax(preds.data.cpu().numpy(), axis=1).astype(np.uint8)
    metric = accuracy_score(trues, preds)
    return float(metric)
