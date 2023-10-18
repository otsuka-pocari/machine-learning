import matplotlib.pyplot as plt
import numpy as np

print('3.6.1-情報利益の最大化:できるだけ高い効果を得る')
# ジニ不純度の関数を定義
def gini(p):
    return (p)*(1-(p)) + (1-p)*(1-(1-p))

# エントロピーの関数を定義
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1-p))

# 分類誤差の関数を定義
def error(p):
    return 1 - np.max([p, 1-p])

# 確率を表す配列を生成(0から0.99まで0.01刻み)
x = np.arange(0.0, 1.0, 0.01)
# 配列の値を元にエントロピー、分類誤差を計算
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
# 図の作成を開始
fig = plt.figure()
ax = plt.subplot(111)
