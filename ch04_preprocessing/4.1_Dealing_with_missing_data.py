import pandas as pd
from io import StringIO

print('4.1.1-表形式のデータで欠陥値を特定する')
# サンプルデータを作成
csv_data = '''A,B,C,D
            1.0,2.0,3.0,4.0
            5.0,6.0,,8.0
            10.0,11.0,12.0'''
# サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))
print(df)
print('------')
# 各特徴量の欠陥をカウント
print('欠陥一覧')
print(df.isnull().sum())
print('------')
print('4.1.2-欠陥値を持つ訓練データ/特徴量を取り除く')
# 欠陥地を含む行を排除
print(df.dropna())
print('------')
# 欠陥地を含む列を排除
print(df.dropna(axis=1))
print('------')
# すべての列がNaNである行だけ排除(すべての値がNaNである行はないため、配列全体が返される)
print(df.dropna(how='all'))
print('------')
# 非NaN値が4つ未満の行を削除
print(df.dropna(thresh=4))
print('------')
# 特定の列(この場合は'C')にNaNが服割れている行だけを削除
print(df.dropna(subset=['C']))
