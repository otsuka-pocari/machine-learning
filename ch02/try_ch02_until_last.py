import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ADALINE import AdalineGD
from plot_decision_regions import plot_decision_regions

s = os.path.join('https://archive.ics.uci.edu', 'ml',
                 'machine-learning-databases', 'iris', 'iris.data')
df = pd.read_csv(s, header=None, encoding='utf-8')
# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1, Iris-versicolorを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
# 1-100行目の1、3列目の抽出
X = df.iloc[0:100, [0,2]].values

print('P39----')
# fig1に保存
# 描画領域を1行2列に分割
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# 勾配降下法によるADALINEの学習(学習率 eta=0.01)
fig = plt.figure(1)
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X,y)
# エポック数とコストの関係を表す折れ線グラフのプロット(縦軸のコストは常用対数)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
# 軸のラベルの設定
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
# タイトルの設定
ax[0].set_title('Adaline - Learning rate 0.01')
# 勾配降下法によるADALINEの学習(学習率 eta=0.0001)
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# エポック数とコストの関係を表す折れ線グラフのプロット
ax[1].plot(range(1,len(ada2.cost_)+1), ada2.cost_, marker='o')
# 軸のラベルの設定
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
# タイトルの設定
ax[1].set_title('Adaline - Learning rate 0.0001')
print('------\n')

print('P41-42 ---')
# データのコピー
X_std = np.copy(X)
# 各列の標準化
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
# fig2に保存
fig = plt.figure(2)
# 勾配降下法によるADALINEの学習(標準語の学習率eta=0.01)
ada_gd = AdalineGD(n_iter=15, eta=0.01)
# モデルの適合
ada_gd.fit(X_std, y)
# 境界領域のプロット
plot_decision_regions(X_std, y, classifier=ada_gd)
# タイトルの設定
plt.title('Adaline - gradient Descent')
# 軸のラベルの設定
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
# 凡例の設定(左上に配置)
plt.legend(loc='upper left')
# fig3に保存
fig = plt.figure(3)
# エポック数とコストの関係を表す折れ線グラフのプロット
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
# 図の表示
plt.tight_layout()
plt.show()
