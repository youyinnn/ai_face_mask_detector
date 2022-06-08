import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch

# draw a confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm :  calculate the value of the confusion matrix
    - classes : class for every row/column
    - normalize : True:show percentage, False:show counts
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("show percentage：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('show specific number：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#test data, can be edited according to the result in fact
#here we use a as sample tensor
a  = np.array([[3.0212e-09, 5.1222e-08, 4.1752e-05, 9.9849e-01, 1.4634e-03],
        [1.2625e-06, 2.0127e-05, 2.4095e-02, 9.2246e-01, 5.3423e-02],
        [5.0156e-11, 3.1125e-10, 1.0847e-05, 9.9998e-01, 1.2953e-05],
        [1.1720e-06, 2.7668e-07, 7.3066e-01, 2.4161e-01, 2.7734e-02],
        [4.5282e-09, 8.6848e-06, 3.2957e-04, 9.5855e-01, 4.1109e-02],
        [3.9356e-08, 8.0953e-06, 3.1157e-03, 8.8281e-01, 1.1406e-01]])


cnf_matrix = np.array([[187,2,1,5,5],
                      [12,176,2,7,3],
                      [0,0,198,2,0],
                      [6,5,4,173,12],
                      [1,3,1,9,186]])
maskTypes = ['Cloth_mask', 'No_face_mask', 'Surgical_mask', 'K95_mask', 'mask_worn_incorrectly']

plot_confusion_matrix(cnf_matrix, classes=maskTypes, normalize=True, title='Normalized confusion matrix')

"""
precision=TP/(TP+FP)
accuracy=(TP+TN)/(TP+TN+FP+FN)
recall=TP/(TP+FN)
f-measure=2*(Precision*Recall)/(Precision+Recall)
"""
cloth_mask=cnf_matrix[0]
No_face_mask=cnf_matrix[1]
Surgical_mask=cnf_matrix[2]
K95_mask=cnf_matrix[3]
mask_worn_incorrectly=cnf_matrix[4]

#for convinience , I use the specific number here. It will be changed later
#For cloth_mask
accuracy=( np.diag(cnf_matrix).sum() )/cnf_matrix.sum()
ColSum=cnf_matrix.sum(axis=0)
RowSum=cnf_matrix.sum(axis=0)
precision1=cloth_mask[0]/ColSum[0]
recall1=cloth_mask[0]/RowSum[0]
fm1=2*(precision1*recall1)/(precision1+recall1)
print("accuracy: ",accuracy)
print("\n\n\n\n\n\n")
print("precision 1: ",precision1)
print("recall 1: ",recall1)
print("fm 1: ",fm1)
#For No_face_mask
precision2=No_face_mask[1]/ColSum[1]
recall2=No_face_mask[1]/RowSum[1]
fm2=2*(precision2*recall2)/(precision2+recall2)
print("precision 2: ",precision2)
print("recall 2: ",recall2)
print("fm 2: ",fm2)
#For Surgical_mask
precision3=Surgical_mask[2]/ColSum[2]
recall3=Surgical_mask[2]/RowSum[2]
fm3=2*(precision3*recall3)/(precision3+recall3)
print("precision 3: ",precision3)
print("recall 3: ",recall3)
print("fm 3: ",fm3)
#For K95_Mask
precision4=K95_mask[3]/ColSum[3]
recall4=K95_mask[3]/RowSum[3]
fm4=2*(precision4*recall4)/(precision4+recall4)
print("precision 4: ",precision4)
print("recall 4: ",recall4)
print("fm 4: ",fm4)
#For mask_worn_incorrectly
precision5=mask_worn_incorrectly[4]/ColSum[4]
recall5=mask_worn_incorrectly[4]/RowSum[4]
fm5=2*(precision5*recall5)/(precision5+recall5)
print("precision 5: ",precision5)
print("recall 5: ",recall5)
print("fm 5: ",fm5)

name_list = ['Cloth_mask', 'No_mask', 'Surgical', 'K95', 'incorrectly']
precision = [precision1, precision2, precision3, precision4, precision5]
recall = [recall1, recall2, recall3, recall4, recall5]
fm = [fm1, fm2, fm3, fm4, fm5]
x = list(range(len(precision)))
total_width, n = 0.4, 2
width = total_width / n

plt.bar(x, precision, width=width, label='precision', fc='y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, recall, width=width, label='recall', tick_label=name_list, fc='r')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, fm, width=width, label='fmeasure', tick_label=name_list, fc='b')

plt.title("Our accuracy is %f" % accuracy)
plt.legend()
plt.show()
