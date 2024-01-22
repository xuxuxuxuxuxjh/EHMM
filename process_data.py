import numpy as np
import torch
import os

input_path = '/data1/xujiahao/Project/Few-NERD-main/data/supervised/test.txt'
output1_path = '/data1/xujiahao/Project/Few-NERD-main/data/supervised/test_text.txt'
output2_path = '/data1/xujiahao/Project/Few-NERD-main/data/supervised/test_label.txt'
file1 = open(output1_path, 'r+')
file2 = open(output2_path, 'r+')
s1 = ''
s2 = ''
for line in open("/data1/xujiahao/Project/Few-NERD-main/data/supervised/test.txt"):
    if line == '\n':
        s1 += '\n'
        s2 += '\n'
        continue
    items = line.split()
    word, tag = items[0],items[1].rstrip()
    s1 += word + ' '
    s2 += tag + ' '
file1.write(s1)
file2.write(s2)

# 'In the early 1930s the band moved to the Grill Room of the Taft Hotel in New York ; the band was renamed `` George Hall and His Hotel Taft Orchestra `` .'