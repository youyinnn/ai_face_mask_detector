import torch
import sklearn
import sklearn.metrics


def test_model(model, data_loader):
    scores = [0] * len(data_loader)
    for i, (x,y) in enumerate(data_loader):
        y_pred = model(x)

        scores[i] = sklearn.metrics.accuracy_score(y,y_pred)
    print(sum(scores)/len(scores))
