import numpy as np

class AdalineGD(object):
    """ADAptive LInear NEuron分類器

    パラメータ
    ------------
    eta : float
        学習率(0.0より大きく1.0以下の値)
    n_iter : int
        訓練データの訓練回数
    random_state : int
        重みを初期化するための乱数シード
    
    属性
    -----------
    w_ : 1次元配列
        適合後の重み
    cost_ : リスト
        各エポックでの誤差平方和のコスト関数

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """訓練データに適応させる

        パラメータ
        ----------
        X : {配列のようなデータ構造}, shape = [n_examples, n_features]
            訓練データ
            n_examplesは訓練データの個数、n_featureは特徴量の個数
        y : 配列のようなデータ構造, shape = [examples]
            目的関数

        戻り値
        -------
        self :object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,  size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):   # 訓練回数分まで訓練データを反復
            net_input = self.net_input(X)
            # activationメソッドは単なる恒等関数であるため、
            # このコードではなんの効果もないことに注意。かわりに、
            # 直接`output = self.net_input(X)`と供述することもできた。
            # activationメソッドの目的は、より概念的なものである。
            # つまり、(後ほど説明する)ロジスティック回帰の場合は、
            # ロジスティック回帰の分類器を実装するためにシグモイド関数に変更することもできる
            output = self.activation(net_input)
            # 誤差 y^(i) - Φ(z^(i))の計算
            errors = (y - output)
            # w_1, ... , w_m の更新
            # Δ w_j = ηΣ_i(y^(i) - Φ(z^(i))) x_j^(i) (j = 1, ... , m)
            self.w_[1:] += self.eta * X.T.dot(errors)
            # w_0の更新 Δ w_0 = ηΣ_i(y(i) - Φ(z^(i)))
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算 J(w) = 1/2 Σ_i(y^(i) - Φ(z(i)))^2
            cost = (errors**2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)
        return self 
    
    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return X

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        # 0.0以上が真→1 偽→-1
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
