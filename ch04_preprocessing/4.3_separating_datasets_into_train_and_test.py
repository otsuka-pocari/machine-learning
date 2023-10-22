import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# wineデータセットを読み込む
df_wine = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
        header=None)
# 列名を設定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity od ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'proline']
# クラスラベルを表示
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())

# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割(全体の30%をテストデータにする)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
