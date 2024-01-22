import numpy as np
import torch
import os

path1 = '/data1/xujiahao/Project/HMM/output3.txt'
path2 = '/data1/xujiahao/Project/Few-NERD-main/data/supervised/test_label.txt'

list1 = []
list2 = []
str = ''
for s in open(path1):
    x = s.split(' ')
    if x[-1] == '\n':
        x = x[:-1]
    list1.append(x)
    continue
for s in open(path2):
    x = s.split(' ')
    if x[-1] == '\n' or x[-1] == '':
        x = x[:-1]
    list2.append(x)
    continue
# print(list2)

TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(list1)):
    # if i==30000:
    #     break
    s1 = list1[i]
    s2 = list2[i]
    for j in range(len(s2)):
        x = s1[j]
        y = s2[j]
        if x == y:
            if x == 'O':
                TN += 1
            else:
                TP += 1
        else:
            if x == 'O':
                FP += 1
            else:
                FN += 1

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_dsc = 2 * TP / (2 * TP + FP + FN)

print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
print(f'Precision: {precision}, Recall: {recall}, F1-Measure: {f1_dsc}')