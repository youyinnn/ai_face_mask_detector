import torch.cuda as cuda
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
import numpy as np


# the evaluation function, use it by:
#   1. provide your data loader for the test data
#   2. the prediction function
#   3. the function to translate the original model output to the label, say: [0.13, 32, 0, 0, -0.33] -> 1


def evaluate(test_loader, model, output_label_match_fn,
             pre_average='micro', rec_average='micro', f1_average='micro'):
    all_outputs = np.empty([0])
    all_targets = np.empty([0])
    model.eval()
    for i, (items, targets) in enumerate(test_loader):
        # items = Variable(items)
        # classes = Variable(classes)

        if cuda.is_available():
            items = items.cuda()
            targets = targets.cuda()

        outputs = model(items).argmax(dim=1)

        all_outputs = np.append(all_outputs, outputs.cpu().detach().numpy())
        all_targets = np.append(all_targets, targets.argmax(dim=1).cpu().detach().numpy())
    # print(metrics.classification_report(all_targets, all_outputs, digits=4))
    acc = accuracy_score(all_targets, all_outputs, )
    # print("acc", acc)
    pre = precision_score(all_targets, all_outputs, average=pre_average)
    rec = recall_score(all_targets, all_outputs, average=rec_average)
    f1 = f1_score(all_targets, all_outputs, average=f1_average)
    conf_m = confusion_matrix(all_targets, all_outputs, )
    classification_report = metrics.classification_report(all_targets, all_outputs, target_names=
    ['cloth_mask', 'no_face_mask', 'surgical_mask', 'n95_mask', 'mask_worn_incorrectly'],
                                                          digits=4, output_dict=True)
    results = {
        'report': classification_report,
        'acc': acc,
        'pre': pre,
        'rec': rec,
        'f1': f1,
        'conf_m': conf_m,
    }
    #print("THE REPORT:")
    #print(classification_report)
    return results
