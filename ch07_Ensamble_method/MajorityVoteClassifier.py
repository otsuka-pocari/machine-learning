from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """多数決アンサンブル分類器

    パラメータ
    ----------
    classifiers : array-like, shape =[n_classifiers]
        アンサンブルの様々な分類器

    vote : str, {'classlabel', 'probability'} (default: 'classlabel')
        'classlabel'の場合、クラスラベルの予測はクラスラベルのargmaxに基づく
        'probability'の場合、クラスラベルの予測はクラスの所属確率の
        argmaxに基づく(分類器が調整済みであることが奨励される)

    weights : array-like, shape = [n_classifiers] (optional, default=None)
        'int'または'float'型の値のリストが提供された場合、分類器は重要度で重み付けされる
        'weights=None'の場合は均一な重みを使用

    """

    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key,
                                  value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(sself, X, y):
        """ 分類機を学習させる
        

        """
