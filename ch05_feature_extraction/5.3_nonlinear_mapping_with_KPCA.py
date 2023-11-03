from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# figに保存
fig = plt.figure(1)
# 2つの半月形データを作成してプロット
X, y = make_moon(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()
