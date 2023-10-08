import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from plot_decision_regions import plot_decision_regions

print('P26---')
v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
print(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
print('------\n')

print('P28---')
s = os.path.join('https://archive.ics.uci.edu', 'ml',
                 'machine-learning-databases', 'iris', 'iris.data')
print('URL:',s)
df = pd.read_csv(s, header=None, encoding='utf-8')
print(df.tail())
print('------\n')

print('P29-P30')
# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1, Iris-versicolorを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
# 1-100行目の1、3列目の抽出
X = df.iloc[0:100, [0,2]].values
# fig1に保存
fig = plt.figure(1)
# 品種setosaのプロット(赤の◯)
plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='setosa')
# 品種versicolorのプロット(青の☓)
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='versicolor')
# 軸のラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定(左上に配置)
plt.legend(loc='upper left')
# パーセプトロンのオブジェクトの生成(インスタンス化)
ppn = Perceptron(eta=0.1, n_iter=10)
# 訓練データへのモデルの適応
ppn.fit(X, y)
# fig2に保存
fig = plt.figure(2)
# エポックと誤分類の関係を表す折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Number of update')
print('-------\n')
# fig3に保存
fig = plt.figure(3)
# 決定領域のプロット
plot_decision_regions(X, y, classifier=ppn)
# 軸ラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定(左上に配置)
plt.legend(loc='upper left')
# 図の表示
plt.show()

