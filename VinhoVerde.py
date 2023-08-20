import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split  # scikit_learn


df = pd.read_csv('winequality-red.csv', sep=';')
print(df['quality'].value_counts().sort_index())  # quantidade de notas
df['vinho_bom'] = df['quality'].map(lambda x: 0 if x < 7 else 1)  # 0 se o valor da linha for menor do que 7
print(df.head())

Xtrain, Xtest, ytrain, ytest = train_test_split(df.iloc[:, :-2], df['vinho_bom'], train_size=0.5)
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)


"""
O problema do gread_search é que ele faz combinações exaltivas (todas) dos parâmetros que queremos colocar no modelo.
Exemplo. Para tunar 7 parâmetros com 10 valores cada, temos que fazer 10**7 combinações, ou seja, teríamos que treinar
10000000 de modelos
"""

# primeira alternativa: Random search - busca aleatório








