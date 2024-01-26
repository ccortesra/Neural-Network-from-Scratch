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
    self.weights = np.array([[1,2],[3,4]], dtype='float64')
    self.biases = np.array([[1],[1]], dtype='float64')
    self.act_func = activation_function
    self.z = None
    self.prev_a = None
    self.old_weights = None
    self.old_biases = None
    self.weight_nudge = None
    self.bias_nudge = None



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
      result = self.act_func(result)
    return result


class NeuralNetwork:
  def __init__(self, input_dim):
    self.layers = []
    self.input_dim = input_dim

  def add_layer(self, number_of_units, activaction_function=False):

    try:
      # Numero de neuronas de la capa anterior(para construir la matriz)
      prev_units = len(self.layers[len(self.layers)-1].biases)
      # Añadir una nueva capa
      self.layers.append(Layer(number_of_units, prev_units, activaction_function))
    
    except: # No hay layers en la neurona por lo tanto pasamos la dimension de entrada para construir la matriz de pesos 
      self.layers.append(Layer(number_of_units, self.input_dim, activaction_function))


  
  def epoch(self, x, y):
    # Entrada X:
    input  = x.reshape(-1, 1)
    # Vamos a pasar por cada capa haciendo los cálculos correspondientes
    for i in range(len(self.layers)):
      input = self.layers[i].calculate(input)
    # Output son todos esos valores computados
    output = input
    # Se cálcula el error con el Minimum Squareed Error en este caso(puede ser cualquier función)
    error = mse(output, y)
    print('MSE: ', error)
    # Se retorna la salida
    return input

  def derivative_a_z(self, current_layer):
    if current_layer.act_func:
      d_a_z = d_relu(current_layer.z)
    else: 
      d_a_z = 1

    d_a_z = d_a_z.reshape(-1,1)
    d_a_z = np.tile(d_a_z, (1,len(current_layer.prev_a)))
    return d_a_z
  
  def derivative_z_w(self, current_layer):
    d_z_w_row = np.ravel(current_layer.prev_a)
    num_rows = len(current_layer.z)
    d_z_w = np.tile(d_z_w_row, (num_rows, 1))
    return d_z_w
  

  def backpropagation(self, predicted, y, lr = 0.001):
    i = len(self.layers)-1
    
    # Derivada del error con respecto a las salidas (a(l)):
    # Simplemente metemos las salidas y los y esperados y los computamos
    # d (C) / d(a_i_(L)) 
    # CASO BASE: ULTIMA CAPA
    d_C_a = d_mse(predicted,y.reshape(-1,1)).reshape(-1,1)
    print('d_C_a', d_C_a)
    # Vamos de adelante hacia atrás

    while i >= 0:
      current_layer = self.layers[i]
      
      # d (C) / d(a_i_(L))
      # CASO GENÉRICO
      if i < len(self.layers)-1:
        d_C_a = delta_a.reshape(-1,1)
      # z(L) son los resultados de los weights y biases al momento de hacer una epoch, por lo tanto podriamos necesitas
      # estos datos
      # d (a(L)) / d(z(L))
      d_a_z = self.derivative_a_z(current_layer)
      # d (z(L)) / d(w(L))
      d_z_w = self.derivative_z_w(current_layer)
      print('d_z_w', d_z_w)

      # Delta del Costo / Delta del Z      
      delta_Z = d_C_a*d_a_z
      # Delta del Costo / Delta del Peso
      delta_weight = delta_Z*d_z_w
      # Cambio de el bias y del peso
      bias_nudge = delta_Z.copy()[:,0] * lr
      weight_nudge = delta_weight*lr

      # Delta del costo / Delta de la entrada anterior (Peso) (SERVIRÁ PARA BACKPROP)
      delta_a = delta_Z*current_layer.weights
      delta_a = np.sum(delta_a, axis=0)

      current_layer.old_weights = current_layer.weights
      current_layer.old_biases = current_layer.biases

      current_layer.weights -= weight_nudge
      current_layer.biases -= bias_nudge.reshape(-1,1)

      i-=1

def main():
  nn = NeuralNetwork(input_dim=2)
  nn.add_layer(2,relu)
  nn.add_layer(2,relu)
  nn.add_layer(2, relu)
  input = np.array([1,2])
  y_target = np.array([1,100])

  for _ in range(10000):
    output = nn.epoch(input, y_target)
    print('Output: ', output)
    nn.backpropagation(output, y_target)
  

main()