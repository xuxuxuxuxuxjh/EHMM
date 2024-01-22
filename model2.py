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

pi = np.zeros(N)
A = np.zeros((N,N))
B = np.zeros((N,M))

print(M)
print(N)

pre_tag = ""
for line in open("/data1/xujiahao/Project/Few-NERD-main/data/supervised/train.txt"):
    if line == "\n": # 遇到空行跳过本次统计
        pre_tag = "" # 同时将 pre_tag 设为空
        continue
        
    items = line.split()
    wordId, tagId = word2id[items[0]], tag2id[items[1].rstrip()]
    
    if pre_tag == "":
        pi[tagId] += 1
        B[tagId][wordId] += 1
    else:
        B[tagId][wordId] += 1
        A[tag2id[pre_tag]][tagId] += 1
    
    pre_tag = items[1].strip() # 为下个时刻记录本时刻的 pre_tag

pi = pi/sum(pi)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])

def log(v):
    if v==0:
        return np.log(v+0.00001)
    return np.log(v)

#维特比算法
def viterbi(str, pi, A, B, key_words, raw_label):
    x = []
    for word in str.split(" "):
        if word not in word2id:
            word = '1'
        x.append(word2id[word])
    T = len(x)

    val = np.zeros(N)
    sum = np.ones(N)
    for word in str.split(" "):
        if word not in word2id:
            word = '1'
        for i, lst in enumerate(key_words):
            for j, y in enumerate(lst):
                if word == y:
                    # print(id2tag[i], word)
                    sum[i] += 0.3 * (len(lst) - j)
    for i in range(N):
        if val[i] > 0:
            continue;
        S = 0
        for j in range(N):
            if id2tag[i].split('-')[0] == id2tag[j].split('-')[0]:
                S += sum[i]
        for j in range(N):
            if id2tag[i].split('-')[0] == id2tag[j].split('-')[0]:
                val[j] = sum[j] / S       
    
    # if sum(val) > 0:
    #     val = val / sum(val)
    
    dp = np.zeros((T,N))  # T 是序列长度，N 是状态总数
    ptr = np.zeros((T,N),dtype=int) # 存放下标
    
    for j in range(N):
        dp[0][j] = log(pi[j]) + log(B[j][x[0]])  # t = 1 时刻的得分单独计算
    
    # 从第二个时刻开始由上至下、由左至右地更新DP数组
    for i in range(1, T):
        for j in range(N):
            dp[i][j] = -999999999
            if raw_label[i] == 'O' and j > 0:
                continue
            if raw_label[i].split('-')[0] == id2tag[j].split('-')[0]:
                # print(raw_label[i], id2tag[j])
                for k in range(N):
                    # print(log(A[k][j]), log(B[j][x[i]]), val[j])
                    score = dp[i-1][k] + log(A[k][j]) + log(B[j][x[i]]) + log(val[j])
                    if score > dp[i][j]:   # 如果得分高于先前，更新DP数组
                        dp[i][j] = score
                        ptr[i][j] = k  # 记录路径
            # else:
            #     for k in range(N):
            #         score = dp[i-1][k] + log(A[k][j]) + log(B[j][x[i]])
            #         if score > dp[i][j]:   # 如果得分高于先前，更新DP数组
            #             dp[i][j] = score
            #             ptr[i][j] = k  # 记录路径                
    
    best_seq = [0]*T
    best_seq[T-1] = np.argmax(dp[T-1]) # 取最后时刻的DP数组中最大值的下标
    
    for i in range(T-2, -1, -1):# 从 T-1 遍历到 0 时刻
        best_seq[i] = ptr[i+1][best_seq[i+1]]
    
    Ans = ''
    for i in range(len(best_seq)):
        Ans += id2tag[best_seq[i]] + ' '
    Ans += '\n'
    return Ans

key_path = '/data1/xujiahao/Project/HMM/key_words.txt'
id_path = '/data1/xujiahao/Project/HMM/list.txt'
label_list = []
for line in open(id_path):
    if line == '\n':
        continue
    label_list.append(line)
key_words = []
for i, line in enumerate(open(key_path)):
    if i > len(label_list):
        continue
    x = line.split(' ')
    if x[-1] == '\n':
        x = x[:-1]
    key_words.append(x)

raw_label = []
for line in open('/data1/xujiahao/Project/HMM/output2.txt'):
    x = line.split(' ')
    x = x[:-1]
    raw_label.append(x)

#测试
test_path = '/data1/xujiahao/Project/Few-NERD-main/data/supervised/test_text.txt'
output_path = '/data1/xujiahao/Project/HMM/output3.txt'

file = open(output_path, 'r+')
Ans = ''
for i, line in enumerate(open(test_path)):
    if line == '\n':
        continue
    Ans = ''
    if line[-1] == '\n':
        Ans += viterbi(line[0:-1], pi, A, B, key_words, raw_label[i])
    else:
        Ans += viterbi(line, pi, A, B, key_words,raw_label[i])
    print(i)
    file.read()
    file.write(Ans)
