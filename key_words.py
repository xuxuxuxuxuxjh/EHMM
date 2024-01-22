import numpy as np
import torch

# 初始化词典
tag2id, id2tag  = {},{} # tag 表示实体标签，如 B-LOC E-LOC 等
word2id, id2word = {},{} # word 表示中文字符

# 根据数据建立词典
for line in open("/data1/xujiahao/Project/Few-NERD-main/data/supervised/train.txt"):
    if line == "\n":
        continue
    items = line.split()
    word, tag = items[0],items[1].rstrip()
    # print(word, tag)
    
    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag

M = len(word2id)
N = len(tag2id)

count = np.zeros((N,M))

print(M)
print(N)

# for i in range(N):
#     print(id2tag[i])

input_path = '/data1/xujiahao/Project/Few-NERD-main/data/supervised/train.txt'
output_path = '/data1/xujiahao/Project/Few-NERD-main/data/supervised/key_words.txt'
file = open(output_path, 'r+')
list1 = []
list2 = []
for i, line in enumerate(open(input_path)):
    print(i)
    if line == '\n':
        for s in list2:
            if s == 'O':
                continue
            for x in list1:
                if x == 'The' or x=='and' or x=='of' or x=='``' or x==',' or x=='.' or x=='the' or x=='in' or x=='to' or x=='(' or x==')' or x=='by' or x=='for' or x=='is' or x=='a' or x=='as' or x=='\'s' or x=='was' or x=='are' or x=='on' or x==':' or x=='at' or x=='\'\'' or x=='with' or x=='from' or x=='In' or x==';' or x=='that' or x=='an' or x=='he' or x=='his' or x=='who' or x=='had' or x=='were'or x=='which' or x=='also' or x=='be' or x=='It' or x=='He':
                    continue
                count[tag2id[s]][word2id[x]] += 1
        list1 = []
        list2 = []
        continue
    items = line.split()
    word, tag = items[0],items[1].rstrip()
    list1.append(word)
    list2.append(tag)

for i in range(N):
    print(i)
    file.read()
    ma = 0
    gs = 0
    ma = count[i].max()   
    if ma == 0:
        file.write('\n')
        continue
    s = ''
    for j in range(M):
        if ma == count[i][j]:
            s += id2word[j] + ' '
            gs += 1
            count[i][j]=0
    ma = count[i].max()   
    if gs < 5 and ma > 0:
        for j in range(M):
            if ma == count[i][j]:
                s += id2word[j] + ' '
                gs += 1
                count[i][j]=0
    ma = count[i].max()   
    if gs < 5 and ma > 0:
        for j in range(M):
            if ma == count[i][j]:
                s += id2word[j] + ' '
                gs += 1
                count[i][j]=0
    ma = count[i].max()   
    if gs < 5 and ma > 0:
        for j in range(M):
            if ma == count[i][j]:
                s += id2word[j] + ' '
                gs += 1
                count[i][j]=0
    ma = count[i].max()   
    if gs < 5 and ma > 0:
        for j in range(M):
            if ma == count[i][j]:
                s += id2word[j] + ' '
                gs += 1
                count[i][j]=0
    s += '\n'
    file.write(s)
