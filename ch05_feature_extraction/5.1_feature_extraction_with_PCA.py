import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print('5.1.2-主成分を抽出する')
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data', header=None)

# 2列目以降のデータをXに１列目のデータをyに格納
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.3, stratify=y, random_state=0)
# 平均と標準偏差を用いて標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)                  # 共分散行列を作成
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)  # 固有値と固有ベクトルを計算
print('\nEigenvalues \n %s' % eigen_vals)
print('------')

print('5.1.3-全分散と説明分散')
# figに保存
fig = plt.figure(1)
print('fig1')
# 固有値を合計
tot = sum(eigen_vals)
# 分散説明率を計算
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積和を取得
cum_var_exp = np.cumsum(var_exp)
# 分散説明率の棒グラフを作成
plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
# 分散説明率の累積和の階段グラフを作成
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Proncipal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
print('------')
