import numpy as np
import math

def InfoGain(data:np.array, split:(int, float), log_base=2):

    '''
        split: a tuple of (feature_idx, threshold)
    '''

    N = data.shape[0]
    split_idx, split_threshold = split[0], split[1]

    H_Y = 0.
    Y, N_Y = np.unique(data[:, -1], return_counts=True)
    for i in N_Y:
        H_Y -= i / N * math.log(i / N, log_base)
    
    data_left = data[np.where(data[:, split_idx] >= split_threshold)]
    data_right = data[np.where(data[:, split_idx] < split_threshold)]
    N_left, N_right = data_left.shape[0], data_right.shape[0]
    assert N_left + N_right == N

    H_Y_left, H_Y_right = 0., 0.
    Y_left, N_Y_left = np.unique(data_left[:, -1], return_counts=True)
    Y_right, N_Y_right = np.unique(data_right[:, -1], return_counts=True)
    for i in N_Y_left:
        H_Y_left -= i / N_left * math.log(i / N_left, log_base)
    for i in N_Y_right:
        H_Y_right -= i / N_right * math.log(i / N_right, log_base)
    H_Y_cond_S = N_left / N * H_Y_left + N_right / N * H_Y_right

    return H_Y - H_Y_cond_S

def Gain(data:np.array, split:(int, float), log_base=2):
    '''
        split: a tuple of (feature_idx, threshold)
    '''

    Info_Gain = InfoGain(data, split)

    N = data.shape[0]
    split_dim, split_threshold = split[0], split[1]
    N_left, N_right = len(np.where(data[:, split_dim] >= split_threshold)), len(np.where(data[:, split_dim] < split_threshold))
    H_S = - N_left / N * math.log(N_left / N, log_base) - N_right / N * math.log(N_right / N, log_base)

    if H_S != 0.:
        Gain_Ratio = Info_Gain / H_S
        return Gain_Ratio, H_S
    else:
        return Info_Gain, H_S

def DetermineCandidateSplits(data):
    '''
        return:
            C: a list of split tuples
    '''

    C = []

    for i in range(data.shape[1] - 1):
        X_i = data[:, i]
        thresholds_i = np.sort(np.unique(X_i))
        for j in range(len(thresholds_i)):
             C.append((i, thresholds_i[j]))
    return C

def FindBestSplit(data, C):

    Best_Gain_Ratio = 0.
    Best_Split = None
    for split in C:
        Gain_, H_S_ = Gain(data, split)
        if H_S_ != 0.:
            if Gain_ > Best_Gain_Ratio:
                Best_Gain_Ratio = Gain_
                Best_Split = split
        
    return Best_Gain_Ratio, Best_Split

class Node():

    def __init__(self, is_leaf=False, feature_idx=None, threshold=None, left_child=None, right_child=None, data=None) -> None:
        self.is_leaf = is_leaf
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.data = data

    def classifiy(self, x):

        if self.is_leaf:
            labels, counts = np.unique(self.data[:,-1], return_counts=True)
            return labels[np.argmax(counts)]
        else:
            if x[self.feature_idx] >= self.threshold:
                return self.left_child.classifiy(x)
            else:
                return self.right_child.classifiy(x)
            
    def len(self):
    
        if self.is_leaf:
            return 1
        else:
            return 1 + self.left_child.len() + self.right_child.len()
    
    def print(self, depth=0):
        if not self.is_leaf:
            print('|   '*depth + '|---x_{}>={}'.format(self.feature_idx, self.threshold))
            self.left_child.print(depth+1)
            print('|   '*depth + '|---x_{}<={}'.format(self.feature_idx, self.threshold))
            self.right_child.print(depth+1)
        else:
            labels, counts = np.unique(self.data[:,-1], return_counts=True)
            print('|   '*depth + '|---label: {}'.format(labels[np.argmax(counts)]))

            
def MakeSubtree(data):

    C = DetermineCandidateSplits(data)
    Best_Gain_Ratio, Best_Split = FindBestSplit(data, C)

    if Best_Gain_Ratio == 0. or Best_Split is None:
        return Node(is_leaf=True, data=data)
    else:
        split_idx, split_threshold = Best_Split[0], Best_Split[1]
        data_left = data[np.where(data[:, split_idx] >= split_threshold)]
        data_right = data[np.where(data[:, split_idx] < split_threshold)]
        left_child = MakeSubtree(data_left)
        right_child = MakeSubtree(data_right)
        return Node(is_leaf=False, feature_idx=split_idx, threshold=split_threshold, left_child=left_child, right_child=right_child)

# def classifiy(tree:Node, x):

#     if tree.is_leaf:
#         labels, counts = np.unique(tree.data[:,-1], return_counts=True)
#         return labels[np.argmax(counts)]
#     else:
#         if x[tree.feature_idx] >= tree.threshold:
#             return classifiy(tree.left_child, x)
#         else:
#             return classifiy(tree.right_child, x)