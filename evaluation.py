import torch.cuda as cuda
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# the evaluation function, use it by:
#   1. provide your data loader for the test data
#   2. the prediction function
#   3. the function to translate the original model output to the label, say: [0.13, 32, 0, 0, -0.33] -> 1


def evaluate(test_loader, predict_fn, output_label_match_fn,
             pre_average='macro', rec_average='macro', f1_average='macro'):
    all_outputs = []
    all_classes = []

    for i, (items, classes) in enumerate(test_loader):
        items = Variable(items)
        classes = Variable(classes)

        if cuda.is_available():
            items = items.cuda()
            classes = classes.cuda()

        outputs = predict_fn(items)

        all_outputs.extend(outputs.cpu().detach().numpy())
        all_classes.extend(classes.cpu())

    acc = accuracy_score([output_label_match_fn(output)
                         for output in all_outputs], all_classes)
    pre = precision_score([output_label_match_fn(output)
                          for output in all_outputs], all_classes, average=pre_average)
    rec = recall_score([output_label_match_fn(output)
                       for output in all_outputs], all_classes, average=rec_average)
    f1 = f1_score([output_label_match_fn(output)
                  for output in all_outputs], all_classes, average=f1_average)
    conf_m = confusion_matrix([output_label_match_fn(output)
                              for output in all_outputs], all_classes)

    return acc, pre, rec, f1, conf_m
