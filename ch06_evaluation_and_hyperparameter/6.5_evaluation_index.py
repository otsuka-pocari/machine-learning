import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
import ssl

# SSLの証明書の期限切れに対する処置
ssl._create_default_https_context = ssl._create_unverified_context

# 前提コード
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'breast-cancer-wisconsin/wdbc.data', header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    stratify=y, random_state=1)

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
# 本コード
print('6.5.1-混合行列を解釈する')
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
# テストと予測のデータから混合行列を生成
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

# 図のサイズを指定
fig, ax = plt.subplots(figsize=(2.5, 2.5))
# matshow関数で行列からヒートマップを描画
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')   # 件数を表示

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()
print('------')

print('6.5.2-分類モデルの適合率と再現率を最適化する')
# 適合率、再現率、F1スコアを出力
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

# カスタムの性能指標を出力
c_gamma_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'svc__C':c_gamma_range, 'svc__kernel':['linear']},
              {'svc__C':c_gamma_range, 'svc__gamma':c_gamma_range,
               'svc__kernel':['rbf']}]

scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10, n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
print('------')

print('6.5.3-ROC曲線をプロットする')
# スケーリング、主成分分析、ロジスティック回帰を指定して、Pipelineクラスをインスタンス化
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2),
                        LogisticRegression(penalty='l2', random_state=1,
                                           solver='lbfgs', C=100))
# 2つの特徴量を抽出
X_train2 = X_train[:, [4, 14]]
# 層化k分割交差検証イテレータを表すStratifiedFoldクラスをインスタンス化
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
# 0から1までの間で100個の要素を生成
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    # predict_probaメソッドで確率を予測、fitメソッドでモデルを適合させる
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])
    # roc_curve関数でROC曲線の性能を計算してプロット
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)  # FPR(X軸)とTPR(Y軸)を線形補間
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)                 # 曲線下面積(AUC)を計算
    plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

# 当て推量をプロット
plt.plot([0, 1], [0, 1],
         linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing')
# FPR, TPR, ROC AUCそれぞれの平均を計算してプロット
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
# 完全に予測が正解したときのROC曲線をプロット
plt.plot([0, 0, 1], [0, 1, 1],
         linestyle=':', color='black', label='Perfect perfomance')
# グラフの各項目を指定
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
print('------')

print('6.5.4-他クラス分類のための性能指標')
pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1,
                         greater_is_better=True,
                         average='micro')
print('------')
