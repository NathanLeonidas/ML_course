import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



path = '../kaggledb/train.csv'
df = pd.read_csv(path)

keys = df.keys()
nondiscrete_keyvals = []

for key in keys:
    #print(key)
    if (df[key].dtype == 'int64') and (key!='Id') and (key!='SalePrice'): nondiscrete_keyvals.append(key)
    #print(df[key].head())
    #print('-'*20 +'\n\n')

Y = np.array(df['SalePrice'])
X = np.array(df[nondiscrete_keyvals])


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)




