import matplotlib.pyplot as plt
import numpy as np
from LogisticRegressionGD import LogisticRegressionGD
from plot_decision_regions import plot_decision_regions
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

print('3.3.1-ロジスティック回帰の直感的知識と条件付き確率---')
# シグモイド何晏数を定義
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
# figに保存
fig = plt.figure(1)
print('fig1')
# 0.1間隔で-7以上7っ未満のデータを生成
z = np.arange(-7, 7, 0.1)
# 生成したデータでシグモイド関数を実行
phi_z = sigmoid(z)
# 元のデータとシグモイド関数の出力をプロット
plt.plot(z, phi_z)
# 垂直線を追加(z=0)
plt.axvline(0.0, color='k')
# y軸の上限/下限を設定
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
# y軸の目盛りを追加
plt.yticks([0.0, 0.5, 1.0])
# Axesクラスのオブジェクトの取得
ax = plt.gca()
# y軸の目盛りに合わせて水平グリッド線を追加
ax.yaxis.grid(True)
print('---\n')

print('3.3.2-ロジスティック関数の重みの学習---')
# y=1のコストを計算する関数
def cost_1(z):
    return - np.log(sigmoid(z))

# y=0のコストを計算する関数
def cost_0(z):
    return - np.log(1 - sigmoid(z))
# figに保存
fig = plt.figure(2)
print('fig2')
# 0.1間隔で-10以上10未満のデータを生成
z = np.arange(-10, 10, 0.1)
# シグモイド関数を実行
phi_z = sigmoid(z)
# y=1のコストを計算する関数を実行
c1 = [cost_1(x) for x in z]
# 結果をプロット
plt.plot(phi_z, c1, label='J(w) if y=1')
# y=0のコストを計算する関数を実行
c0 = [cost_0(x) for x in z]
# 結果をプロット
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')
# x軸とy軸の上限/下限を設定
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
# 軸のラベルを設定
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
# 凡例を設定
plt.legend(loc='upper center')
print('---\n')

print('3.3.3-ADALINE実装をロジスティック回帰のアルゴリズムに変換する')
# 3.2からのコードの引用
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train== 1)]
# figに保存
fig = plt.figure(3)
print('fig3')
# ロジスティック回帰のインスタンスを生成
lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
# モデルを訓練データに適合させる
lrgd.fit(X_train_01_subset, y_train_01_subset)
# 決定領域をプロット
plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
print('---\n')

print('3.3.4-scikit-learnを使ったロジスティック回帰モデルの訓練')
# 3.2からのコードの引用
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
# figに保存
fig = plt.figure(4)
print('fig4')
# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
# 訓練データをモデルに適合させる
lr.fit(X_train_std, y_train)
# 決定領域をプロット
plot_decision_regions(X_combined_std, y_combined, classifier=lr,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal wigth [standardized]')
plt.legend(loc='upper left')

print(lr.predict_proba(X_test_std[:3,:]))
print(lr.predict(X_test_std[:3, :]))
print(lr.predict(X_test_std[0, :].reshape(1, -1)))
print('---\n')

print('3.3.5-正則化による過学習への対処')
# figに保存
fig = plt.figure(5)
print('fig5')
# 空のリストを生成(重み係数、正則化パラメータ)
weights, params = [], []
# 10個の逆正則化パラメータに対応するロジスティック回帰モデルをそれぞれ処理
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1,
                            solver='lbfgs', multi_class='ovr')
    lr.fit(X_train_std, y_train)
    # 重み係数を格納
    weights.append(lr.coef_[1])
    # 逆正則化パラメータを格納
    params.append(10.**c)

# 重み係数をNumpy配列に変換
weights = np.array(weights)
# 横軸に逆正則化パラメータ、縦軸に重み係数をプロット
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
# 横軸を対数スケールに設定
plt.xscale('log')
plt.show()
