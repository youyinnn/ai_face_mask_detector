import torch
import torch.cuda as cuda
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
import numpy as np

import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from data_process.DatasetHelper import mask_label_list, mask_gen_label_list, mask_race_label_list

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
    mask_label_list,
                                                          digits=4, output_dict=True)

    classification_report_gen = metrics.classification_report(all_new_targets_gen, all_new_outputs_gen, target_names=
    mask_gen_label_list,
                                                          digits=4, output_dict=True)
    conf_m_gen = confusion_matrix(all_new_targets_gen, all_new_outputs_gen, )
    
    classification_report_race = metrics.classification_report(all_new_targets_race, all_new_outputs_race, target_names=
    mask_race_label_list,
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

five_label_display_name = ['cloth', 'no_face', 'surgical', 'n95', 'incorrect']
gen_label_display_name = ['cloth_m', 'cloth_fm', 'no_face_m', 'no_face_fm', 'surgical_m', 'surgical_fm', 'n95_m', 'n95_fm', 'incorrect_m', 'incorrect_fm']
race_label_display_name = ['cloth_caas', 'cloth_afar', 'no_face_caas', 'no_face_afar', 'surgical_caas', 'surgical_afar', 'n95_caas', 'n95_afar', 'incorrect_caas', 'incorrect_afar']

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6), dpi=150)
    """
    - cm :  calculate the value of the confusion matrix
    - classes : class for every row/column
    - normalize : True:show percentage, False:show counts
    """
    if normalize:   #for calculating the percentage
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("show percentage：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #interpolation = 'nearest': when the display resolution is different with our image,
    #our script will output the image without adding other values between pixels.
    #https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
    plt.title(title, fontsize=15, pad=10) # adding our title 
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-30)#deciding what labels will be shown on x axis
    plt.yticks(tick_marks, classes)# deciding what labels will be shown on y axis
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)  #y axis label
    plt.xlabel('Predicted label', fontsize=12)  #x axis label
    plt.show()

def read_socres(file_path, conf_m_title):
    print(file_path)
    if os.path.exists(file_path):
      with open(file_path, 'rb') as f:
          a = np.load(f, allow_pickle=True)
          report = a['report']
          # print(a.item().keys())
          
          df_arr = {'precision': [], 'recall': [], 'f1-score': []}
          for mask_label in mask_label_list:
              del report.item()[mask_label]['support']
              for kk in report.item()[mask_label].keys():
                df_arr[kk].append(report.item()[mask_label][kk])
          df = pd.DataFrame(data=df_arr, index=five_label_display_name)
          
          print(df)
          
          print('Overall acc: ', a['acc'])
          plot_confusion_matrix(a['conf_m'], classes=five_label_display_name, title=conf_m_title)
          
def read_socres_gen(file_path, conf_m_title):
    print(file_path)
    if os.path.exists(file_path):
      with open(file_path, 'rb') as f:
          a = np.load(f, allow_pickle=True)
          report = a['report_gen']
          # print(a.item().keys())
          
          df_arr = {'precision': [], 'recall': [], 'f1-score': []}
          for gen_label in mask_gen_label_list:
              del report.item()[gen_label]['support']
              for kk in report.item()[gen_label].keys():
                df_arr[kk].append(report.item()[gen_label][kk])
                
          bias = []
          for i in range(len(df_arr['f1-score'])):
            if i % 2 != 0:
              bias.append(df_arr['f1-score'][i] - df_arr['f1-score'][i - 1])
            else:
              bias.append(' ')
              
          df_arr['bias(fm - m)'] = bias
                
          df = pd.DataFrame(data=df_arr, index=gen_label_display_name)
          
          print(df)
          print('Overall acc: ', a['acc'])
          plot_confusion_matrix(a['conf_m_gen'], classes=gen_label_display_name, title=conf_m_title)


def read_socres_race(file_path, conf_m_title):
    print(file_path)
    if os.path.exists(file_path):
      with open(file_path, 'rb') as f:
          a = np.load(f, allow_pickle=True)
          report = a['report_race']
          # print(a.item().keys())
          
          df_arr = {'precision': [], 'recall': [], 'f1-score': []}
          for race_label in mask_race_label_list:
              del report.item()[race_label]['support']
              for kk in report.item()[race_label].keys():
                df_arr[kk].append(report.item()[race_label][kk])
          
          bias = []
          for i in range(len(df_arr['f1-score'])):
            if i % 2 != 0:
              bias.append(df_arr['f1-score'][i] - df_arr['f1-score'][i - 1])
            else:
              bias.append(' ')
              
          df_arr['bias(afar - caas)'] = bias
          df = pd.DataFrame(data=df_arr, index=race_label_display_name)
          
          print(df)
          print('Overall acc: ', a['acc'])
          plot_confusion_matrix(a['conf_m_race'], classes=race_label_display_name, title=conf_m_title)

