First Version of the Neural Network from Scratch made in Python

This Neural Network is based on a OOP architecture, we have three classes.

Perceptron: Basic unit of a neural network, has weights for each entry, its activation functionand its entries.
This class remained unused as we used an approch using matrix multiplication per layer.

Layer: Using a matrix approach, we represented the weights of each neuron of the layer as rows
of our Layer, and the columns would be the entries from the previous layer, therefore
the i,j element of the "weights" matrix, would represent the weight of the ith neuron
that links to the jth entry.

Layer has a biases vector, each element matches with one neuron of the layer.

NeuralNetwork:

Basically a DS to store all the layers, actually an array where we append all of our Layer
objects. As they are contigous this make the approach pretty favourable.


*LIMITATIONS*
Limitations of the version are mainly related with the possible functions parameters, for example
error function or activaction fucntions, this code provides 5 activation functions, but I only
included de derivative of relu, this is failry easy to include so its up to you.

Likewise, I only included MSE as error function (important remark, this MSE does not average up),
and its derivative
