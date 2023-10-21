import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print('4.2.2-pandasを使ったカテゴリーデータのエンコーディング')
# サンプルデータを生成
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2'],])
# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)
print('------')

print('4.2.3-順序特徴量のマッピング')
print('sizeの順序を教える')
# Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
print(df)
print('------')
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))
print('------')
print('4.2.4-クラスラベルのエンコーディング')
# クラスラベルと整数を対応させるディクショナリを生成
class_mapping = {label: idx for idx, label in 
                 enumerate(np.unique(df['classlabel']))}
print(class_mapping)
print('------')
# クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
print('------')
# 整数とクラスラベルを対応させるディクショナリを生成
inv_class_mapping = {v: k for k, v in class_mapping.items()}
# 整数からクラスラベルに変換
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)
print('------')
# ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print('------')
# クラスラベルを文字列に戻す
print(class_le.inverse_transform(y))
