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
