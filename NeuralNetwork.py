import numpy as np


sigmoid = lambda a: 1 / (1 + np.exp(-a))
step = lambda x: np.piecewise(x, [x<=0.0, x>0.0],[0,1])
relu = lambda x: np.piecewise(x, [x<=0.0, x>0.0],[0, lambda x:x])
d_relu = lambda x: np.piecewise(x, [x<=0.0, x>0.0],[0, 1])
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
    self.z = None
    self.prev_a = None


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
    self.z = result 
    self.prev_a = prev_output
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


  
  def epoch(self, x, y):
    # Entrada X:
    input  = x.reshape(-1, 1)
    # Vamos a pasar por cada capa haciendo los cálculos correspondientes
    for i in range(len(self.layers)):
      input = self.layers[i].calculate(x)
    # Output son todos esos valores computados
    output = input
    # Se cálcula el error con el Minimum Squareed Error en este caso(puede ser cualquier función)
    error = mse(output, y)
    print('MSE: ', error)
    # Se retorna la salida
    return input



  def backpropagation(self, predicted, y, lr = 0.01):
    i = len(self.layers)-1
    # d (C0) / d(a(L))
    d_C_a = d_mse(predicted,y)
    print('d_C_a', d_C_a)
    # Vamos de adelante hacia atrás
    while i >= 0:
      current_layer = self.layers[i]
      
      # z(L) son los resultados de los weights y biases al momento de hacer una epoch, por lo tanto podriamos necesitas
      # estos datos
      # d (a(L)) / d(z(L))
      if current_layer.act_func:
        d_a_z = d_relu(current_layer.z)
      else: 
        d_a_z = 1
      # d (z(L)) / d(w(L))
      d_z_w = current_layer.prev_a
      print('d_z_w', d_z_w)
      # d (C0) / d(W) =  d (C0) / d(a(L)) * d (a(L)) / d(z(L)) * d (z(L)) / d(w(L))
      weight_nudge = d_C_a*d_a_z*d_z_w
      # d (C0) / d(W) =  d (C0) / d(a(L)) * d (a(L)) / d(z(L)) * d (z(L)) / d(b(L)) (last term is 1)
      bias_nudge = d_C_a*d_a_z


      # Update the weights and biases
      current_layer.weights -= weight_nudge*lr
      current_layer.biases -= bias_nudge*lr

      i-=1

def main():
  nn = NeuralNetwork(1)
  nn.add_layer(1)
  input = np.array([1.5])
  y_target = np.array([0.8]) 

  for _ in range(100):
    output = nn.epoch(input, y_target)
    print('Output: ', output)
    nn.backpropagation(output, y_target)
  

main()