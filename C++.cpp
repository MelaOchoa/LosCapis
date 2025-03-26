//FUNCIÓN LINEAL
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

class Perceptron {
private:
    vector<double> weights;
    double bias;
    double learningRate;
    int epochs;
    string activationFunction;

    double activation(double x) {
        if (activationFunction == "lineal") {
            return x;
        }
        return x; // Por defecto, lineal
    }

public:
    Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20) {
        srand(time(0));
        weights.resize(inputSize);
        for (double &w : weights) {
            w = ((double)rand() / RAND_MAX) * 2 - 1; // Valores aleatorios entre -1 y 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1;
        this->learningRate = learningRate;
        this->epochs = epochs;
        this->activationFunction = activationFunction;
    }

    double predict(const vector<double>& X) {
        double linearOutput = bias;
        for (size_t i = 0; i < X.size(); i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    void train(const vector<vector<double>>& X, const vector<double>& y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (size_t i = 0; i < X.size(); i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;
                for (size_t j = 0; j < weights.size(); j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += abs(error);
            }
            cout << "Epoch " << epoch + 1 << "/" << epochs << ", Error: " << totalError << endl;
        }
    }
};

int main() {
    vector<vector<double>> X_or = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> y_or = {0, 1, 1, 1};

    Perceptron perceptron(2, "lineal", 0.1, 15);
    perceptron.train(X_or, y_or);

    cout << "Predicciones OR:" << endl;
    for (const auto& x : X_or) {
        cout << "Entrada: [" << x[0] << ", " << x[1] << "], Salida: " << perceptron.predict(x) << endl;
    }

    return 0;
}

/* FUNCIÓN ESCALÓN *\
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

class Perceptron {
private:
    vector<double> weights;
    double bias;
    double learningRate;
    int epochs;
    string activationFunction;

    // Función de activación Escalón
    double activation(double x) {
        if (activationFunction == "escalon") {
            return (x >= 0) ? 1 : 0;
        }
        return 0; // Por defecto, devuelve 0 si no es una función reconocida
    }

public:
    Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20) {
        srand(time(0));
        weights.resize(inputSize);
        for (double &w : weights) {
            w = ((double)rand() / RAND_MAX) * 2 - 1; // Pesos aleatorios entre -1 y 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1;
        this->learningRate = learningRate;
        this->epochs = epochs;
        this->activationFunction = activationFunction;
    }

    // Función de predicción
    double predict(const vector<double>& X) {
        double linearOutput = bias;
        for (size_t i = 0; i < X.size(); i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    // Función de entrenamiento
    void train(const vector<vector<double>>& X, const vector<double>& y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (size_t i = 0; i < X.size(); i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;
                for (size_t j = 0; j < weights.size(); j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += abs(error);
            }
            cout << "Epoch " << epoch + 1 << "/" << epochs << ", Error: " << totalError << endl;
        }
    }
};

int main() {
    // Conjunto de datos OR
    vector<vector<double>> X_or = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> y_or = {0, 1, 1, 1};

    // Creación del perceptrón con activación escalón
    Perceptron perceptron(2, "escalon", 0.1, 15);
    perceptron.train(X_or, y_or);

    // Predicciones
    cout << "Predicciones OR:" << endl;
    for (const auto& x : X_or) {
        cout << "Entrada: [" << x[0] << ", " << x[1] << "], Salida: " << perceptron.predict(x) << endl;
    }

    return 0;
}

/* FUNCIÓN SIGMOIDAL *\
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;
class Perceptron {
private:
    vector<double> weights;
    double bias;
    double learningRate;
    int epochs;
    string activationFunction;

    // Función de activación Sigmoide
    double activation(double x) {
        if (activationFunction == "sigmoide") {
            return 1.0 / (1.0 + exp(-x));
        }
        return 0.0; // Por defecto, si no reconoce la activación
    }
public:
    Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20) {
        srand(time(0));
        weights.resize(inputSize);
        for (double &w : weights) {
            w = ((double)rand() / RAND_MAX) * 2 - 1; // Pesos aleatorios entre -1 y 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1;
        this->learningRate = learningRate;
        this->epochs = epochs;
        this->activationFunction = activationFunction;
    }

    // Función de predicción
    double predict(const vector<double>& X) {
        double linearOutput = bias;
        for (size_t i = 0; i < X.size(); i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    // Función de entrenamiento
    void train(const vector<vector<double>>& X, const vector<double>& y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (size_t i = 0; i < X.size(); i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;
                for (size_t j = 0; j < weights.size(); j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += abs(error);
            }
            cout << "Epoch " << epoch + 1 << "/" << epochs << ", Error: " << totalError << endl;
        }
    }
};
int main() {
    // Conjunto de datos OR
    vector<vector<double>> X_or = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> y_or = {0, 1, 1, 1};

    // Creación del perceptrón con activación sigmoide
    Perceptron perceptron(2, "sigmoide", 0.1, 15);
    perceptron.train(X_or, y_or);

    // Predicciones
    cout << "Predicciones OR:" << endl;
    for (const auto& x : X_or) {
        cout << "Entrada: [" << x[0] << ", " << x[1] << "], Salida: " << perceptron.predict(x) << endl;
    }
    return 0;
}
/* FUNCIÓN RELU *\
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

class Perceptron {
private:
    vector<double> weights;
    double bias;
    double learningRate;
    int epochs;
    string activationFunction;

    // Función de activación ReLU
    double activation(double x) {
        if (activationFunction == "relu") {
            return max(0.0, x);
        }
        return 0.0; // Si la activación no está definida
    }

public:
    Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20) {
        srand(time(0));
        weights.resize(inputSize);
        for (double &w : weights) {
            w = ((double)rand() / RAND_MAX) * 2 - 1; // Pesos aleatorios entre -1 y 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1;
        this->learningRate = learningRate;
        this->epochs = epochs;
        this->activationFunction = activationFunction;
    }

    // Función de predicción
    double predict(const vector<double>& X) {
        double linearOutput = bias;
        for (size_t i = 0; i < X.size(); i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    // Función de entrenamiento
    void train(const vector<vector<double>>& X, const vector<double>& y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (size_t i = 0; i < X.size(); i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;
                for (size_t j = 0; j < weights.size(); j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += abs(error);
            }
            cout << "Epoch " << epoch + 1 << "/" << epochs << ", Error: " << totalError << endl;
        }
    }
};

int main() {
    // Conjunto de datos OR
    vector<vector<double>> X_or = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> y_or = {0, 1, 1, 1};

    // Creación del perceptrón con activación ReLU
    Perceptron perceptron(2, "relu", 0.1, 15);
    perceptron.train(X_or, y_or);

    // Predicciones
    cout << "Predicciones OR:" << endl;
    for (const auto& x : X_or) {
        cout << "Entrada: [" << x[0] << ", " << x[1] << "], Salida: " << perceptron.predict(x) << endl;
    }

    return 0;
}

/* FUNCIÓN SOFTMAX *\
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <numeric>
using namespace std;

class Perceptron {
private:
    vector<double> weights;
    double bias;
    double learningRate;
    int epochs;
    string activationFunction;

    // Función de activación Softmax
    vector<double> activation(const vector<double>& x) {
        vector<double> expValues(x.size());
        double sumExp = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            expValues[i] = exp(x[i]);
            sumExp += expValues[i];
        }
        for (size_t i = 0; i < x.size(); i++) {
            expValues[i] /= sumExp; // Normalización de Softmax
        }
        return expValues;
    }
public:
    Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20) {
        srand(time(0));
        weights.resize(inputSize);
        for (double &w : weights) {
            w = ((double)rand() / RAND_MAX) * 2 - 1; // Pesos aleatorios entre -1 y 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1;
        this->learningRate = learningRate;
        this->epochs = epochs;
        this->activationFunction = activationFunction;
    }
    // Función de predicción con Softmax
    vector<double> predict(const vector<double>& X) {
        vector<double> linearOutput(1, bias);
        for (size_t i = 0; i < X.size(); i++) {
            linearOutput[0] += X[i] * weights[i];
        }
        return activation(linearOutput);
    }
    // Función de entrenamiento
    void train(const vector<vector<double>>& X, const vector<int>& y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (size_t i = 0; i < X.size(); i++) {
                vector<double> prediction = predict(X[i]);
                double error = y[i] - prediction[0]; // Softmax genera un valor en [0,1]
                for (size_t j = 0; j < weights.size(); j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += abs(error);
            }
            cout << "Epoch " << epoch + 1 << "/" << epochs << ", Error: " << totalError << endl;
        }
    }
};
int main() {
    // Conjunto de datos OR
    vector<vector<double>> X_or = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<int> y_or = {0, 1, 1, 1};

    // Creación del perceptrón con activación Softmax
    Perceptron perceptron(2, "softmax", 0.1, 15);
    perceptron.train(X_or, y_or);

    // Predicciones
    cout << "Predicciones OR:" << endl;
    for (const auto& x : X_or) {
        vector<double> output = perceptron.predict(x);
        cout << "Entrada: [" << x[0] << ", " << x[1] << "], Salida: " << output[0] << endl;
    }
    return 0;
}
/* FUNCIÓN TANGENTE HIPERBÓLICA *\
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class Perceptron {
private:
    vector<double> weights;
    double bias;
    double learning_rate;
    int epochs;
    string activation_function;

public:
    // Constructor
    Perceptron(int input_size, string activation_function, double learning_rate = 0.01, int epochs = 20) {
        this->learning_rate = learning_rate;
        this->epochs = epochs;
        this->activation_function = activation_function;

        // Inicialización aleatoria de pesos y bias
        srand(time(0));
        for (int i = 0; i < input_size; i++) {
            weights.push_back(((double)rand() / RAND_MAX) * 2 - 1); // Valores entre -1 y 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // Función de activación tanh
    double activation(double x) {
        if (activation_function == "tanh") {
            return tanh(x);
        }
        return x; // Por defecto, función identidad
    }

    // Predicción
    double predict(vector<double> X) {
        double linear_output = bias;
        for (size_t i = 0; i < X.size(); i++) {
            linear_output += X[i] * weights[i];
        }
        return activation(linear_output);
    }

    // Entrenamiento
    void train(vector<vector<double>> X, vector<double> y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_error = 0;
            for (size_t i = 0; i < X.size(); i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;

                // Actualización de pesos y bias
                for (size_t j = 0; j < weights.size(); j++) {
                    weights[j] += learning_rate * error * X[i][j];
                }
                bias += learning_rate * error;
                total_error += abs(error);
            }
            cout << "Epoch " << epoch + 1 << "/" << epochs << ", Error: " << total_error << endl;
        }
    }
};

int main() {
    // Datos de entrada OR
    vector<vector<double>> X_or = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> y_or = {0, 1, 1, 1};

    // Crear y entrenar el perceptrón
    Perceptron perceptron(2, "tanh", 0.1, 15);
    perceptron.train(X_or, y_or);

    // Realizar predicciones
    cout << "Predicciones OR:" << endl;
    for (const auto& x : X_or) {
        cout << "Entrada: [" << x[0] << ", " << x[1] << "], Salida: " << perceptron.predict(x) << endl;
    }

    return 0;
}
