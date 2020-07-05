import numpy as np

'''
softmax函数
'''

def softmax(input):
    # 先求行维度的最大值
    max_v = np.max(input, axis=1)
    # max_v扩大维度
    max_v = max_v[:, np.newaxis]
    # input按行减去其最大值，避免出现数值上溢
    input -= max_v
    # 求exp
    exp_input = np.exp(input)
    # 按行对exp_input求和
    sum_exp_input = np.sum(exp_input, axis=1)
    return exp_input / sum_exp_input

'''
softmax交叉熵损失函数 - 非one-hot版本
输入：pred - [N,C], target - [N,1]
'''

def Softmax_cross_entropy(pred, target):
    # 将pred通过softmax函数
    pred_softmax = softmax(pred)
    N = pred_softmax.shape[0]
    # 求target对应的pred_softmax值
    res = pred_softmax[range(N), target]
    # 求交叉熵
    loss = - np.sum(np.log(res + 1e-7)) / N
    return loss

'''
softmax交叉熵损失函数 - one-hot版本
输入：pred - [N,C], target - [N,C]
'''

def Softmax_cross_entropy_onehot(pred, target):
    # 将pred通过softmax函数
    pred_softmax = softmax(pred)
    # 求target对应的pred_softmax值
    res = pred_softmax
    # 求交叉熵
    N = res.shape[0]
    # 先计算log，避免非label的res_log下溢
    res_log = np.log(res + 1e-7)
    loss = - np.sum(res_log * target) / N
    return loss

pred = np.array([[5,2.4,7.9],[2.1,4.5,6.1],[1,2.4,8.9]])
# 验证softmax
# print(softmax(pred))
# 非one-hot target
target_non_onehot = np.array([0, 1, 1])
# 验证Softmax_cross_entropy
# print(Softmax_cross_entropy(pred, target_non_onehot))
target_onehot = np.array([[1,0,0],[0,1,0],[0,1,0]])
# 验证Softmax_cross_entropy_onehot
print(Softmax_cross_entropy_onehot(pred, target_onehot))
