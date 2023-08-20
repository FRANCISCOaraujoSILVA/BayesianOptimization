from skopt import gp_minimize
from skopt.plots import plot_convergence


# Função a ser minimizada
def objective_function(x):
    return x[0]**2 + x[1]**2


# Espaço de busca para os parâmetros
space = [(-5.0, 5.0), (-5.0, 5.0)]  # Intervalos para os dois parâmetros

result = gp_minimize(objective_function, space, n_calls=20, random_state=10, verbose=1)
print(' ')
print(result.x)

# Função plot_convergence para visualizar a convergência
# plot_convergence(result)
