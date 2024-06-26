import numpy as np

def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

class NeuralNetwork:
    def __init__(self):
      # Weights
      self.w1 = np.random.normal()
      self.w2 = np.random.normal()
      self.w3 = np.random.normal()
      self.w4 = np.random.normal()
      self.w5 = np.random.normal()
      self.w6 = np.random.normal()

      # Biases
      self.b1 = np.random.normal()
      self.b2 = np.random.normal()
      self.b3 = np.random.normal()
    
    def feed_forward(self, input):
        out_h1 = sigmoid(self.w1 * input[0] + self.w2 * input[1] + self.b1)
        out_h2 = sigmoid(self.w3 * input[0] + self.w4 * input[1] + self.b2)
        out_o1 = sigmoid(self.w5 * out_h1 + self.w6 * out_h2 + self.b3)
        return out_o1
    
    def train(self, data, all_y_trues):
       learn_rate = 0.1
       epochs = 1000 

       for epoch in range(epochs):
          for x, y_true in zip(data, all_y_trues):
            # --- Do a feedforward (we'll need these values later)
            sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
            h1 = sigmoid(sum_h1)

            sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
            h2 = sigmoid(sum_h2)

            sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
            o1 = sigmoid(sum_o1)
            y_pred = o1

            # --- Calculate partial derivatives.
            # --- Naming: d_L_d_w1 represents "partial L / partial w1"
            d_L_d_ypred = -2 * (y_true - y_pred)

            # Neuron o1
            d_ypred_d_w5 = h1 * self.deriv_sigmoid(sum_o1)
            d_ypred_d_w6 = h2 * self.deriv_sigmoid(sum_o1)
            d_ypred_d_b3 = self.deriv_sigmoid(sum_o1)

            d_ypred_d_h1 = self.w5 * self.deriv_sigmoid(sum_o1)
            d_ypred_d_h2 = self.w6 * self.deriv_sigmoid(sum_o1)

            # Neuron h1
            d_h1_d_w1 = x[0] * self.deriv_sigmoid(sum_h1)
            d_h1_d_w2 = x[1] * self.deriv_sigmoid(sum_h1)
            d_h1_d_b1 = self.deriv_sigmoid(sum_h1)

            # Neuron h2
            d_h2_d_w3 = x[0] * self.deriv_sigmoid(sum_h2)
            d_h2_d_w4 = x[1] * self.deriv_sigmoid(sum_h2)
            d_h2_d_b2 = self.deriv_sigmoid(sum_h2)

            # --- Update weights and biases
            # Neuron h1
            self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
            self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
            self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

            # Neuron h2
            self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
            self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
            self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

            # Neuron o1
            self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
            self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
            self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

          # --- Calculate total loss at the end of each epoch
          if epoch % 10 == 0:
            y_preds = np.apply_along_axis(self.feed_forward, 1, data)
            loss = self.mse_loss(all_y_trues, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))
    
    @staticmethod
    def mse_loss(y_true, y_pred):
       return ((y_true - y_pred) ** 2).mean()

    @staticmethod
    def deriv_sigmoid(x):
      fx = sigmoid(x)
      return fx * (1 - fx)

