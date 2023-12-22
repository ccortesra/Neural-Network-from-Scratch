import numpy as np


sigmoid = lambda a: 1 / (1 + np.exp(-a))
step = lambda x: np.piecewise(x, [x<=0.0, x>0.0],[0,1])
relu = lambda x: np.piecewise(x, [x<=0.0, x>0.0],[0, lambda x:x])
tanh = lambda a: (np.exp(a))/(np.exp(-a)+np.exp(a))-(np.exp(-a))/(np.exp(-a)+np.exp(a))


def mse(a,b):
  return np.sum((a-b)**2)

def d_mse(a,b):
  return 2*(a-b)

class Perceptron:
  def __init__(self, weights=False, entries= False, act_f = False):
    self.weights = weights
    self.activaction_function = act_f
    self.entries = entries

  def calculate_output(self):
    return np.dot(self.weights, self.entries)

class Layer:
  def __init__(self, units, prev_units, activation_function):
    self.weights = np.ones((units, prev_units))
    self.biases = np.ones(units)
    self.act_func = activation_function
  

  def calculate(self, prev_output):

    # Prev Output
    # Es un arreglo de m elementos con las salidad de las m neuronas de la capa anterior
    # (X_0, X_1, ... X_{m-1})

    # Weights
    # Matriz con todos los pesos de todas las neuronas y todas las entradas, siendo n el número de neuronas y m el número de entradas, su dimensión es de n x m 
    # (X_00,     X_01,     ... X_0{m-1})
    # (X_10,     X_11,     ... X_1{m-1})
    # ...
    # (X_{n-1}0, X_{n-1}1, ... X_{n-1}{m-1})
    result_matrix = np.dot(self.weights, prev_output)
    result = result_matrix + self.biases
    if self.act_func:
      result = self.act_func(result_matrix)
    
    return result


class NeuralNetwork:
  def __init__(self, input_dim):
    self.layers = []
    self.input_dim = input_dim

  def add_layer(self, number_of_units, activaction_function=False):

    try:
      # Numero de neuronas de la capa anterior(para construir la matriz)
      prev_units = len(self.layers[len(self.layers)-1].neurons)
      # Añadir una nueva capa
      self.layers.append(Layer(number_of_units, prev_units, activaction_function))
    
    except: # No hay layers en la neurona por lo tanto pasamos la dimension de entrada para construir la matriz de pesos 
      self.layers.append(Layer(number_of_units, self.input_dim, activaction_function))


  
  def compute(self, x, y):
    input  = x.reshape(-1, 1)
    for i in range(len(self.layers)):
      input = self.layers[i].calculate(x)
    output = input
    error = mse(output,y)
    print('MSE: ', error)
    return input


def main():
  nn = NeuralNetwork(1)
  nn.add_layer(1)
  input = np.array([1.5])
  y_target = np.array([0.8]) 
  print(nn.compute(input, y_target))

main()