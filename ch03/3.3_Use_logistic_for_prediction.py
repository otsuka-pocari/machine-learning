import matplotlib.pyplot as plt
import numpy as np

print('3.3.1-ロジスティック回帰の直感的知識と条件付き確率---')
# シグモイド何晏数を定義
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
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
# グラフを表示
plt.tight_layout()
plt.show()
print('---\n')
