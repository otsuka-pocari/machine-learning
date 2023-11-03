from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """RBFカーネルPCAの実装

    パラメータ
    ------------
    X: {Numpy ndarray}, shape = [n_examples, n_features]

    gamma: float
        RBFカーネルのチューニングパラメータ

    n_components: int
        返される主成分の個数

    戻り値
    ------------
    X_pc: {Numpy ndarray}, shape = [n_examples, k_features]
        射影されたデータセット

    """

    # M×M次元のデータセットでペアごとのユークリッド距離の2乗を計算

