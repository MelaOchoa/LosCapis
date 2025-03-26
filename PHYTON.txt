#FUNCIÓN LINEAL
import numpy as np

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01, epochs=20):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
   
    def activation(self, x):
        if self.activation_function == "lineal":
            return x
   
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)
   
    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                total_error += abs(error)
            print(f"Epoch {epoch+1}/{self.epochs}, Error: {total_error}")

# Prueba con el caso OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

perceptron_and = Perceptron(input_size=2, activation_function="lineal", learning_rate=0.1, epochs=15)
perceptron_and.train(X_or, y_or)

print("Predicciones OR:")
for x in X_or:
    print(f"Entrada: {x}, Salida: {perceptron_and.predict(x)}")

#FUNCIÓN ESCALÓN
import numpy as np

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01, epochs=20):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
   
    def activation(self, x):
        if self.activation_function == "escalon":
            return 1 if x >= 0 else 0
   
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)
   
    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                total_error += abs(error)
            print(f"Epoch {epoch+1}/{self.epochs}, Error: {total_error}")

# Prueba con el caso OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

perceptron_and = Perceptron(input_size=2, activation_function="escalon", learning_rate=0.1, epochs=15)
perceptron_and.train(X_or, y_or)

print("Predicciones OR:")
for x in X_or:
    print(f"Entrada: {x}, Salida: {perceptron_and.predict(x)}")


#FUNCIÓN SIGMOIDAL
import numpy as np

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01, epochs=20):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
   
    def activation(self, x):
        if self.activation_function == "sigmoide":
            return 1 / (1 + np.exp(-x))
   
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)
   
    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                total_error += abs(error)
            print(f"Epoch {epoch+1}/{self.epochs}, Error: {total_error}")

# Prueba con el caso OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

perceptron_and = Perceptron(input_size=2, activation_function="sigmoide", learning_rate=0.1, epochs=15)
perceptron_and.train(X_or, y_or)

print("Predicciones OR:")
for x in X_or:
    print(f"Entrada: {x}, Salida: {perceptron_and.predict(x)}")


#FUNCIÓN RELU
import numpy as np

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01, epochs=20):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
   
    def activation(self, x):
        if self.activation_function == "relu":
            return max(0, x)
   
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)
   
    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                total_error += abs(error)
            print(f"Epoch {epoch+1}/{self.epochs}, Error: {total_error}")

# Prueba con el caso OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

perceptron_and = Perceptron(input_size=2, activation_function="relu", learning_rate=0.1, epochs=15)
perceptron_and.train(X_or, y_or)

print("Predicciones OR:")
for x in X_or:
    print(f"Entrada: {x}, Salida: {perceptron_and.predict(x)}")


#FUNCIÓN SOFTMAX
import numpy as np

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01, epochs=20):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
   
    def activation(self, x):
        if self.activation_function == "softmax":
            return np.exp(x) / np.sum(np.exp(x))

   def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)
   
    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                total_error += abs(error)
            print(f"Epoch {epoch+1}/{self.epochs}, Error: {total_error}")

# Prueba con el caso OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

perceptron_and = Perceptron(input_size=2, activation_function="softmax", learning_rate=0.1, epochs=15)
perceptron_and.train(X_or, y_or)

print("Predicciones OR:")
for x in X_or:
    print(f"Entrada: {x}, Salida: {perceptron_and.predict(x)}")


#FUNCIÓN TANGENTE HIPERBÓLICA
import numpy as np

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01, epochs=20):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
   
    def activation(self, x):
        if self.activation_function == "tanh":
            return np.tanh(x)
   
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)
   
    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                total_error += abs(error)
            print(f"Epoch {epoch+1}/{self.epochs}, Error: {total_error}")

# Prueba con el caso OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

perceptron_and = Perceptron(input_size=2, activation_function="tanh", learning_rate=0.1, epochs=15)
perceptron_and.train(X_or, y_or)

print("Predicciones OR:")
for x in X_or:
    print(f"Entrada: {x}, Salida: {perceptron_and.predict(x)}")

