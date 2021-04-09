import torch
def calculate_accuracy(y_pred, y):
    with torch.no_grad():
        batch_size = y.shape[0]

        _, top_pred = y_pred.topk(k=1)

        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].view(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
    return acc_1
