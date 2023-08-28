from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import numpy as np

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')


# Cria o modelo
def create_model(neurons, optimizer, activation, kernel_initializer, loss):
    model = Sequential()
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    model.add(Dropout(0.2))
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy'])
    return model


# Espaço de busca dos hiperparâmetros
space = [
    Integer(8, 16, name='neurons'),
    Categorical(['adam', 'sgd'], name='optimizer'),
    Categorical(['relu', 'tanh'], name='activation'),
    Categorical(['random_uniform', 'normal'], name="kernel_initializer"),
    Categorical(['binary_crossentropy', 'hinge'], name="loss")]


# Função objetivo para otimizar os hiperparâmetros
def objective(params):
    neurons, optimizer, activation, kernel_initializer, loss = params

    # Dividindo os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(previsores, classe, test_size=0.25, random_state=42)

    model = create_model(neurons, optimizer, activation, kernel_initializer, loss)
    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)
    _, accuracy = model.evaluate(X_test, y_test)

    return -accuracy  # Minimizar a negação da acurácia


# Executar a otimização de hiperparâmetros com gp_minimize
result = gp_minimize(objective, space, n_calls=20, random_state=42, verbose=1)

# Imprimir os melhores hiperparâmetros encontrados
best_params = result.x
print("Melhores hiperparâmetros encontrados:", best_params)

# Melhores hiperparâmetros encontrados: [16, 'adam', 'relu', 'random_uniform', 'binary_crossentropy']

# Classificar um valor:

neurons = best_params[0]
optimizer = best_params[1]
activation = best_params[2]
kernel_initializer = best_params[3]
loss = best_params[4]

model = Sequential()  # criação do model sequencial
model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
model.add(Dropout(0.2))
model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy'])
model.fit(previsores, classe, batch_size=10, epochs=100)

novo = np.array([[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1095, 0.9053, 8589, 153.4,
                  0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656,
                  0.7119, 0.2654, 0.4601, 0.1189]])

previsao = model.predict(novo)
print(previsao)

previsao = (previsao > 0.95)  # vai ser True se a previsão foir maior que 0.95
print(previsao)
