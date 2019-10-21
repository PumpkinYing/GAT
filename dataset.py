import numpy as np 
import torch

def load_data(path="./data/data/") :
    print("Loading dataset...")

    # matrix = np.loadtxt(path+"matrix.txt", delimiter=' ')
    feature = np.loadtxt(path+"feature.txt", delimiter=' ')
    out = np.loadtxt(path+"out.txt", delimiter=' ')

    feature = torch.FloatTensor(normalize(feature))
    matrix = torch.FloatTensor(np.ones((feature.shape[0], feature.shape[0])))
    # matrix = torch.FloatTensor(matrix+np.eye(matrix.shape[0]))
    # matrix = torch.FloatTensor(normalize(matrix))
    out = torch.FloatTensor(out)

    idx_train = range(500)
    idx_val = range(500,1000)
    idx_test = range(1100,1600)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return matrix, feature, out, idx_train, idx_val, idx_test

def normalize(mx) :
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
