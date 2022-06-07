import torch.cuda as cuda
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
# the evaluation function, use it by:
#   1. provide your data loader for the test data
#   2. the prediction function
#   3. the function to translate the original model output to the label, say: [0.13, 32, 0, 0, -0.33] -> 1


def evaluate(test_loader, model, output_label_match_fn,
             pre_average='macro', rec_average='macro', f1_average='macro'):
    all_outputs = np.empty([0])
    all_targets = np.empty([0])

    for i, (items, targets) in enumerate(test_loader):
        #items = Variable(items)
        #classes = Variable(classes)

        if cuda.is_available():
            items = items.cuda()
            targets = targets.cuda()

        outputs = model(items).argmax(dim=1)

        all_outputs = np.append(all_outputs, outputs.cpu().detach().numpy())
        all_targets = np.append(all_targets, targets.argmax(dim=1).cpu().detach().numpy())

    acc = accuracy_score(all_outputs, all_targets)
    print("acc", acc)
    pre = precision_score(all_outputs, all_targets, average=pre_average)
    rec = recall_score(all_outputs, all_targets, average=rec_average)
    f1 = f1_score(all_outputs, all_targets, average=f1_average)
    conf_m = confusion_matrix(all_outputs, all_targets)
    results = {
        'acc': acc,
        'pre': pre,
        'rec': rec,
        'f1': f1,
        'conf_m': conf_m,
    }
    return results
