import torch
import numpy as np
from sklearn import metrics


def aupr(label, prob):
    precision, recall, thresholds = metrics.precision_recall_curve(label, prob)
    area = metrics.auc(recall, precision)
    return area


def auroc(label, prob):
    area = metrics.roc_auc_score(label, prob, multi_class='ovo')
    return area


def compute_oscr(pred_k, pred_u, labels):
    #  pred_k：已知类别经过Softmax得到的置信度 N*C，N为样本数，C为已知类的类别数
    #  pred_u：未知类别经过Softmax得到的置信度 M*C，M为样本数，C为已知类的类别数
    #  labels：已知类别标签 N*1，N为样本数
    #  x1，x2 最大置信度 N*1，M*1
    if type(pred_k) is not np.ndarray:
        pred_k, pred_u, labels = pred_k.numpy(), pred_u.numpy(), labels.view(-1).numpy()
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    #  pred_k中最大置信度对应的类别
    pred = np.argmax(pred_k, axis=1)
    #  pred_k中预测正确的样本
    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    # 注意，不同点在于这里
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)
    #  Cutoffs are of prediction values
    CCR = [0 for _ in range(n + 1)]
    FPR = [0 for _ in range(n + 1)]
    idx = predict.argsort()
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]
    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()
        #  True  Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        #  False Positive Rate
        FPR[k] = float(FP) / float(len(x2))
    CCR[n] = 0.0
    FPR[n] = 0.0
    #  Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)
    OSCR = 0
    #  Compute AUROC Using Trapezoidal Rule
    for j in range(n):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0
        OSCR = OSCR + h * w

    return OSCR


if __name__ == '__main__':
    prob_k = torch.randn((8, 4))
    print(prob_k)
    prob_u = torch.randn((5, 4))
    print(prob_u)
    # label = torch.randint(4, (1, 8))
    label = prob_k.max(1)[1]
    print(label)
    roc = compute_oscr(prob_k, prob_u, label)
    # pr = aupr(label, prob)
    # roc = auroc(label, torch.softmax(prob, dim=1))
    print('end')
