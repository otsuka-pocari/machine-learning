import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from SBS import SBS

# 前提コード
df_wine = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
        header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity od ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print('4.5.3-L1正則化による疎な解')
# L1正則化ロジスティック回帰のインスタンスを生成
LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')
# L1正則化ロジスティック回帰のインスタンスを生成:逆正則化パラメータC=1.0はデフォルト値であり、
# 値を大きくしたり小さくしたりすると、正則化の効果を強めたり弱めたりできる
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
# 訓練データに適合
lr.fit(X_train_std, y_train)
# 訓練データに対する正解率の表示
print('Training accuracy:', lr.score(X_train_std, y_train))
# テストデータに対する正解率の表示
print('Test accuracy:', lr.score(X_test_std, y_test))
print(lr.intercept_)
# 重みの表示
print(lr.coef_)
# 描画の準備
fig = plt.figure(figsize=(16, 9), dpi=40)
ax = plt.subplot(111)
# 各係数の色のリスト
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
# 空のリストを生成(重み係数、逆正則化パラメータ)
weights, params = [], []
# 逆生息パラメータの値ごとに処理
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

# 重み係数をNumPy配列に変換
weights = np.array(weights)
# 各重み係数をプロット
for column, color in zip(range(weights.shape[1]), colors):
    # 横軸を逆正則化パラメータ、縦軸を重み係数とした折れ線グラフ
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1],
             color=color)

# y=0に黒い波線を引く
plt.axhline(0, color='black', linestyle='--', linewidth=3)
# 横軸の範囲の設定
plt.xlim([10**(-5), 10**5])
# 軸のラベルの設定
plt.ylabel('weight coefficient')
plt.xlabel('C')
# 横軸を対数スケールに設定
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.tight_layout()
plt.show()

fig = plt.figure(2)
# k最近傍法分類器のインスタンスを生成(近傍点数=5)
knn = KNeighborsClassifier(n_neighbors=5)
# 逐次後退選択のインスタンスを生成
sbs = SBS(knn, k_features=1)
# 逐次後退選択を実行
sbs.fit(X_train_std, y_train)
# 特徴量の個数のリスト(13, 13, ..., 1)
k_feat = [len(k) for k in sbs.subsets_]
# 横軸を特徴量の個数、縦軸をスコアとした折れ線グラフのプロット
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
