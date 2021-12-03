# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:33:00 2021

@author: LBY
"""

import numpy as np
from glob import glob
import csv
import random


file_to_submit = "prediction_1.csv"
head = "exp12/"

with open(file_to_submit, 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    writer.writerow(['guid/image', 'label'])

test_set_names = np.loadtxt('labels_names_test.csv', skiprows=1, dtype=str, delimiter=',')
test_set = glob(head + 'labels/*.txt')



ii = 0
jj = 0
iii = 0
with open(file_to_submit, 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    writer.writerow(['guid/image', 'label'])
    for name in test_set_names:
        print(name[0], )
        temp_name = name[0].replace('/', '__')
        file_name = head + "labels/"+ temp_name + ".txt"
        status = 1
        try:
            file_handle = open(file_name, mode='r')
            content = file_handle.readlines()
            file_handle.close()
            print(len(content))
            if len(content) == 1:
                is_single = 1
                temp_label = content[0][0]
                writer.writerow([name[0], temp_label])
            else:
                is_single = 0
                conf_list = []
                conf_len = len(content)
                for i in range(conf_len):
                    temp_str = content[i]
                    temp_list = temp_str.split(' ')
                    temp_conf = float(temp_list[-1].strip())
                    conf_list.append(temp_conf)
                conf_list_np = np.array(conf_list).reshape(conf_len, 1)
                conf_list_np_max_idx = conf_list_np.argmax()
                temp_label = content[conf_list_np_max_idx][0] 
                print(temp_label)
                writer.writerow([name[0], temp_label])
        except:
            status = 0
            temp_label = 10  # unknown, write 10
            writer.writerow([name[0], temp_label])
        if status == 1:
            if is_single == 1:
                ii += 1
            else:
                iii += 1
        else:
            jj += 1
        