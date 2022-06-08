import torch
import numpy as np

a = np.array([[3.0212e-09, 5.1222e-08, 4.1752e-05, 9.9849e-01, 1.4634e-03],
        [1.2625e-06, 2.0127e-05, 2.4095e-02, 9.2246e-01, 5.3423e-02],
        [5.0156e-11, 3.1125e-10, 1.0847e-05, 9.9998e-01, 1.2953e-05],
        [1.1720e-06, 2.7668e-07, 7.3066e-01, 2.4161e-01, 2.7734e-02],
        [4.5282e-09, 8.6848e-06, 3.2957e-04, 9.5855e-01, 4.1109e-02],
        [3.9356e-08, 8.0953e-06, 3.1157e-03, 8.8281e-01, 1.1406e-01]])
cnf_matrix = np.array([[0,0,0,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0]])
a = torch.from_numpy(a)
print(a.size())
b = a.numpy()
b = b.tolist()
count=0
for i in b:
    print("%.15f" %i[3])
    print("%.15f %d" % (max(i),i.index(max(i))))
    cnf_matrix[count][i.index(max(i))]+=1
    count+=1
print(cnf_matrix)
cloth_mask=cnf_matrix[0]
No_face_mask=cnf_matrix[1]
Surgical_mask=cnf_matrix[2]
K95_mask=cnf_matrix[3]
mask_worn_incorrectly=cnf_matrix[4]


"""
print(cnf_matrix.sum(axis=0))
print(np.diag(cnf_matrix).sum())
print(Surgical_mask.sum())
print(cnf_matrix.sum())
"""