import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plot_decision_regions import plot_decision_regions
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import matplotlib.image as mpimg

# 前提コード
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

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
fig = plt.figure(1)
ax = plt.subplot(111)
print('fig1')
# エントロピー(2種)、分類誤差のそれぞれをループ処理
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy(scaled)',
                           'Gini impurity', 'Misclassification error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

# 凡例の設定(中央の上に配置)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

# 2本の水平の波線を書く
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
# 横軸の上限/下限の設定
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('imurity index')
print('---')

print('3.6.2-決定木の構築')
# ジニ不純度を指標とする決定機のインスタンを生成
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
# figに保存
fig = plt.figure(2)
print('fig2')
# 決定木のモデルを訓練データに適合させる
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree_model,
                      test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')

# figに保存
fig = plt.figure(3)
print('fig3')
tree.plot_tree(tree_model)

# figの保存
fig = plt.figure(4)
print('fig4')
dot_data = export_graphviz(tree_model, filled=True, rounded=True,
                           class_names=['Setosa','Versicolor','Vinginica'],
                           feature_names=['petal length','petal width'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')
img = mpimg.imread('tree.png')
imgplot = plt.imshow(img)
plt.show()
