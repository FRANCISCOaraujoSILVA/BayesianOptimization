import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split  # scikit_learn
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize   # skop is scikit-optimize 0.9.0
import skopt.plots

"""
O problema do gread_search é que ele faz combinações exaltivas (todas) dos parâmetros que queremos colocar no modelo.
Exemplo. Para tunar 7 parâmetros com 10 valores cada, temos que fazer 10**7 combinações, ou seja, teríamos que treinar
10000000 de modelos
-----------------------------------------------------------------------------------------------------------------------
Scikit-Optimize, ou skopt, é uma biblioteca simples e eficiente para minimizar funções de caixa preta (muito) caras e 
barulhentas.
Ele implementa vários métodos para otimização baseada em modelo sequencial. O Skopt pretende ser acessível e fácil de 
usar em muitos contextos.
-----------------------------------------------------------------------------------------------------------------------
No GridSearch damos a mesma importância para todos os parãmetros, sendo que na verdade isso faz o nosso modelo gastar
muito tempo. Na realidade, alguns parâmetros têm mais influencia que outros.
-----------------------------------------------------------------------------------------------------------------------
No Bayesian Optimization damos mais importância aos parâmetros que mais importam e exploramos mais ao redor desses
parâmetros.

-----------------------------------------------------------------------------------------------------------------------
# exploration exploitation tradeoff. Para saber mais.
"""

df = pd.read_csv('winequality-red.csv', sep=';')
df['vinho_bom'] = df['quality'].map(lambda x: 0 if x < 7 else 1)  # 0 se o valor da linha for menor do que 7


Xtrain, Xtest, ytrain, ytest = train_test_split(df.iloc[:, :-2], df['vinho_bom'], train_size=0.5)


def trainModel(params):
    learning_rate, num_leaves, min_child_samples, subample, colsample_bytree = params

    mdl = LGBMClassifier(learning_rate=learning_rate,
                         num_leaves=num_leaves,
                         min_child_samples=min_child_samples,
                         subsample=subample,
                         colsample_bytree=colsample_bytree,
                         random_state=0,
                         subsample_freq=1,
                         n_estimators=100)

    mdl.fit(Xtrain, ytrain)
    p = mdl.predict_proba(Xtest)[:, 1]
    return -roc_auc_score(ytest, p)


# Valor mínimo e máximo do parâmetro que queremos avaliar
space = [
         (1e-3, 1e-1, 'log-uniform'),   # learning rate
         (2, 200),                      # num_leaves
         (1, 100),                      # min_child_samples
         (0.05, 1.0),                   # subsample
         (0.1, 1)                       # colsample_bytree
         ]


BayesianOptimizer = gp_minimize(trainModel, space, random_state=1, verbose=1, n_calls=60, n_random_starts=10)
# skopt.plots.plot_convergence(BayesianOptimizer)
print(BayesianOptimizer.x)


