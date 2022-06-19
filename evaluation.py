import torch
import torch.cuda as cuda
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
import numpy as np


# the evaluation function, use it by:
#   1. provide your data loader for the test data
#   2. the prediction function
#   3. the function to translate the original model output to the label, say: [0.13, 32, 0, 0, -0.33] -> 1

def downgrade_target(targets, device):
  down_grade_target = [list(torch.sum(torch.reshape(t, (5, 4)), dim=1).cpu().detach().numpy()) for t in targets]
  new_targets = torch.tensor(np.array(down_grade_target), device=device)
  return new_targets

def downgrade_argmax_target_to_gen(t):
  if t in [0, 2]:
    return 0
  if t in [1, 3]:
    return 1
  if t in [4, 6]:
    return 2
  if t in [5, 7]:
    return 3
  if t in [8, 10]:
    return 4
  if t in [9, 11]:
    return 5
  if t in [12, 14]:
    return 6
  if t in [13, 15]:
    return 7
  if t in [16, 18]:
    return 8
  if t in [17, 19]:
    return 9

def downgrade_argmax_target_to_race(t):
  if t in [0, 1]:
    return 0
  if t in [2, 3]:
    return 1
  if t in [4, 5]:
    return 2
  if t in [6, 7]:
    return 3
  if t in [8, 9]:
    return 4
  if t in [10, 11]:
    return 5
  if t in [12, 13]:
    return 6
  if t in [14, 15]:
    return 7
  if t in [16, 17]:
    return 8
  if t in [18, 19]:
    return 9

def evaluate(test_loader, model,
             pre_average='micro', rec_average='micro', f1_average='micro', device='cpu'):
    all_outputs = np.empty([0])
    all_targets = np.empty([0])
    all_new_outputs_gen = np.empty([0])
    all_new_targets_gen = np.empty([0])
    all_new_outputs_race = np.empty([0])
    all_new_targets_race = np.empty([0])
    model.eval()
    for i, (items, targets) in enumerate(test_loader):
        # items = Variable(items)
        # classes = Variable(classes)

        if cuda.is_available():
            items = items.cuda()
            targets = targets.cuda()

        new_targets_20 = targets.argmax(dim=1).cpu().detach().numpy()
        targets = downgrade_target(targets, device)
        outputs = list(model(items).argmax(dim=1).cpu().detach().numpy())

        new_targets_gen = [downgrade_argmax_target_to_gen(t) for t in new_targets_20]
        new_outputs_gen = []

        for i in range(len(new_targets_gen)):
          new_outputs_gen.append(outputs[i] * 2 + (new_targets_gen[i] % 2))

        new_targets_race = [downgrade_argmax_target_to_race(t) for t in new_targets_20]
        new_outputs_race = []

        for i in range(len(new_targets_race)):
          new_outputs_race.append(outputs[i] * 2 + (new_targets_race[i] % 2))

        # print(list(new_targets_20))
        # print(list(new_targets_gen))
        # print(outputs)
        # print(new_outputs_gen)

        all_outputs = np.append(all_outputs, outputs)
        all_targets = np.append(all_targets, targets.argmax(dim=1).cpu().detach().numpy())
        all_new_outputs_gen = np.append(all_new_outputs_gen, new_outputs_gen)
        all_new_targets_gen = np.append(all_new_targets_gen, new_targets_gen)
        all_new_outputs_race = np.append(all_new_outputs_race, new_outputs_race)
        all_new_targets_race = np.append(all_new_targets_race, new_targets_race)

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

    classification_report_gen = metrics.classification_report(all_new_targets_gen, all_new_outputs_gen, target_names=
    ['cloth_mask_m', 'cloth_mask_fm', 'no_face_mask_m', 'no_face_mask_fm', 'surgical_mask_m', 'surgical_mask_fm', 'n95_mask_m', 'n95_mask_fm', 'mask_worn_incorrectly_m', 'mask_worn_incorrectly_fm'],
                                                          digits=4, output_dict=True)
    conf_m_gen = confusion_matrix(all_new_targets_gen, all_new_outputs_gen, )
    
    classification_report_race = metrics.classification_report(all_new_targets_race, all_new_outputs_race, target_names=
    ['cloth_mask_caas', 'cloth_mask_afar', 'no_face_mask_caas', 'no_face_mask_afar', 'surgical_mask_caas', 'surgical_mask_afar', 'n95_mask_caas', 'n95_mask_afar', 'mask_worn_incorrectly_caas', 'mask_worn_incorrectly_afar'],
                                                          digits=4, output_dict=True)
    conf_m_race = confusion_matrix(all_new_targets_race, all_new_outputs_race, )
    

    results = {
        'report': classification_report,
        'acc': acc,
        'pre': pre,
        'rec': rec,
        'f1': f1,
        'conf_m': conf_m,
        'report_gen': classification_report_gen,
        'conf_m_gen': conf_m_gen,
        'report_race': classification_report_race,
        'conf_m_race': conf_m_race,
    }
    #print("THE REPORT:")
    #print(classification_report)
    return results
