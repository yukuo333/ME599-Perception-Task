# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:58:10 2021

@author: LBY
"""

from glob import glob
import csv
import numpy as np
import random
from random import choice

# yolov5
file_1 = np.loadtxt('prediction_contains_invalid.csv', skiprows=1, dtype=str, delimiter=',')
# deepnet 1
file_2 = np.loadtxt('prediction_label.csv', skiprows=1, dtype=str, delimiter=',')
# deepnet 2
file_3 = np.loadtxt('test_label.csv', skiprows=1, dtype=str, delimiter=',')

weight_file_1 = 1/3
weight_file_2 = 1/3
weight_file_3 = 1/3

thre = 0.5

file_to_submit = 'predict_combined.csv'

decided = 0
undecided = 0

valid_label_list = ['0', '1', '2']
with open(file_to_submit, 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    writer.writerow(['guid/image', 'label'])
    for i in range(len(file_1)):
        temp_name = file_1[i][0]
        temp_dict = {'0':0, '1':0, '2':0}
        temp_label_1 = file_1[i][1]
        temp_label_2 = file_2[i][1]
        temp_label_3 = file_3[i][1]
        if temp_label_1 in valid_label_list:
            temp_dict[temp_label_1] += weight_file_1
            temp_dict[temp_label_2] += weight_file_2
            temp_dict[temp_label_3] += weight_file_3
            method_1_valid = True
        else:
            temp_dict[temp_label_2] += 0.5
            temp_dict[temp_label_3] += 0.5
            method_1_valid = False
        #print(temp_name, temp_label_1, temp_label_2, temp_label_3)
        #print(temp_dict)
        flag = 0
        for key, val in temp_dict.items():
            if temp_dict[key] > thre:
                flag = 1
                break
        if flag == 1:
            label_selected = key
            decided += 1
        else:
            if method_1_valid:
                label_selected = random.randint(0, 2)
                undecided += 1
            else:
                choose_list = [temp_label_2, temp_label_3]
                label_selected = choice(choose_list)
                undecided += 1
            
        print(temp_name, temp_label_1, temp_label_2, temp_label_3, 'flag:', flag, 'label:', label_selected)
        writer.writerow([temp_name, label_selected])
    



