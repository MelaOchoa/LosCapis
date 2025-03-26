// FUNCIÓN LINEAL
import java.util.Random;

class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private String activationFunction;

    public Perceptron(int inputSize, String activationFunction, double learningRate, int epochs) {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextDouble() * 2 - 1; // Valores entre -1 y 1
        }
        bias = rand.nextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double activation(double x) {
        if (activationFunction.equals("lineal")) {
            return x;
        }
        return x; // Función lineal por defecto
    }

    public double predict(double[] X) {
        double linearOutput = bias;
        for (int i = 0; i < X.length; i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    public void train(double[][] X, double[] y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < X.length; i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;

                // Actualizar pesos y sesgo
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.abs(error);
            }
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + ", Error: " + totalError);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        double[][] X_or = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };

        double[] y_or = {0, 1, 1, 1};

        Perceptron perceptron = new Perceptron(2, "lineal", 0.1, 15);
        perceptron.train(X_or, y_or);

        System.out.println("Predicciones OR:");
        for (double[] x : X_or) {
            System.out.println("Entrada: [" + x[0] + ", " + x[1] + "], Salida: " + perceptron.predict(x));
        }
    }
}

// FUNCIÓN ESCALÓN
import java.util.Random;

class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private String activationFunction;

    public Perceptron(int inputSize, String activationFunction, double learningRate, int epochs) {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextDouble() * 2 - 1; // Inicializa pesos entre -1 y 1
        }
        bias = rand.nextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private int activation(double x) {
        if (activationFunction.equals("escalon")) {
            return (x >= 0) ? 1 : 0;
        }
        return 0; // Valor por defecto si no se reconoce la función
    }

    public int predict(double[] X) {
        double linearOutput = bias;
        for (int i = 0; i < X.length; i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    public void train(double[][] X, int[] y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            int totalError = 0;
            for (int i = 0; i < X.length; i++) {
                int prediction = predict(X[i]);
                int error = y[i] - prediction;

                // Actualizar pesos y sesgo
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.abs(error);
            }
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + ", Error: " + totalError);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        double[][] X_or = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };

        int[] y_or = {0, 1, 1, 1};

        Perceptron perceptron = new Perceptron(2, "escalon", 0.1, 15);
        perceptron.train(X_or, y_or);

        System.out.println("Predicciones OR:");
        for (double[] x : X_or) {
            System.out.println("Entrada: [" + x[0] + ", " + x[1] + "], Salida: " + perceptron.predict(x));
        }
    }
}

// FUNCIÓN SIGMOIDAL
import java.util.Random;

class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private String activationFunction;

    public Perceptron(int inputSize, String activationFunction, double learningRate, int epochs) {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextDouble() * 2 - 1; // Inicializa pesos entre -1 y 1
        }
        bias = rand.nextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double activation(double x) {
        if (activationFunction.equals("sigmoide")) {
            return 1 / (1 + Math.exp(-x));
        }
        return 0; // Valor por defecto si no se reconoce la función
    }

    public double predict(double[] X) {
        double linearOutput = bias;
        for (int i = 0; i < X.length; i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    public void train(double[][] X, double[] y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < X.length; i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;

                // Actualizar pesos y sesgo
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.abs(error);
            }
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + ", Error: " + totalError);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        double[][] X_or = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };

        double[] y_or = {0, 1, 1, 1};

        Perceptron perceptron = new Perceptron(2, "sigmoide", 0.1, 15);
        perceptron.train(X_or, y_or);

        System.out.println("Predicciones OR:");
        for (double[] x : X_or) {
            System.out.println("Entrada: [" + x[0] + ", " + x[1] + "], Salida: " + perceptron.predict(x));
        }
    }
}
// FUNCIÓN RELU
import java.util.Random;

class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private String activationFunction;

    public Perceptron(int inputSize, String activationFunction, double learningRate, int epochs) {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextDouble() * 2 - 1; // Inicializa pesos entre -1 y 1
        }
        bias = rand.nextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double activation(double x) {
        if (activationFunction.equals("relu")) {
            return Math.max(0, x); // ReLU: max(0, x)
        }
        return x; // Valor por defecto si no se reconoce la función
    }

    public double predict(double[] X) {
        double linearOutput = bias;
        for (int i = 0; i < X.length; i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    public void train(double[][] X, double[] y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < X.length; i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;

                // Actualizar pesos y sesgo
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.abs(error);
            }
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + ", Error: " + totalError);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        double[][] X_or = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };

        double[] y_or = {0, 1, 1, 1};

        Perceptron perceptron = new Perceptron(2, "relu", 0.1, 15);
        perceptron.train(X_or, y_or);

        System.out.println("Predicciones OR:");
        for (double[] x : X_or) {
            System.out.println("Entrada: [" + x[0] + ", " + x[1] + "], Salida: " + perceptron.predict(x));
        }
    }
}

// FUNCIÓN SOFTMAX
import java.util.Random;

class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private String activationFunction;

    public Perceptron(int inputSize, String activationFunction, double learningRate, int epochs) {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextDouble() * 2 - 1; // Inicializa pesos entre -1 y 1
        }
        bias = rand.nextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double[] softmax(double[] x) {
        double sumExp = 0;
        double[] expValues = new double[x.length];

        // Calculamos la exponencial de cada valor y su suma
        for (int i = 0; i < x.length; i++) {
            expValues[i] = Math.exp(x[i]);
            sumExp += expValues[i];
        }

        // Normalizamos dividiendo cada valor por la suma total
        for (int i = 0; i < x.length; i++) {
            expValues[i] /= sumExp;
        }

        return expValues;
    }

    public double predict(double[] X) {
        double linearOutput = bias;
        for (int i = 0; i < X.length; i++) {
            linearOutput += X[i] * weights[i];
        }

        // Softmax solo tiene sentido para múltiples clases, aquí simplificamos
        double[] softmaxOutput = softmax(new double[]{linearOutput, 1 - linearOutput});
        return softmaxOutput[1]; // Retornamos la probabilidad de la clase positiva
    }

    public void train(double[][] X, double[] y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < X.length; i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;

                // Actualizar pesos y sesgo
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.abs(error);
            }
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + ", Error: " + totalError);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        double[][] X_or = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };

        double[] y_or = {0, 1, 1, 1};

        Perceptron perceptron = new Perceptron(2, "softmax", 0.1, 15);
        perceptron.train(X_or, y_or);

        System.out.println("Predicciones OR:");
        for (double[] x : X_or) {
            System.out.println("Entrada: [" + x[0] + ", " + x[1] + "], Salida: " + perceptron.predict(x));
        }
    }
}

// FUNCIÓN TANGENTE HIPERBÓLICA
import java.util.Random;

class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private String activationFunction;

    public Perceptron(int inputSize, String activationFunction, double learningRate, int epochs) {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextDouble() * 2 - 1; // Valores entre -1 y 1
        }
        bias = rand.nextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double activation(double x) {
        if (activationFunction.equals("tanh")) {
            return Math.tanh(x);
        }
        return x; // Función lineal por defecto
    }

    public double predict(double[] X) {
        double linearOutput = bias;
        for (int i = 0; i < X.length; i++) {
            linearOutput += X[i] * weights[i];
        }
        return activation(linearOutput);
    }

    public void train(double[][] X, double[] y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < X.length; i++) {
                double prediction = predict(X[i]);
                double error = y[i] - prediction;

                // Actualizar pesos y sesgo
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.abs(error);
            }
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + ", Error: " + totalError);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        double[][] X_or = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };

        double[] y_or = {0, 1, 1, 1};

        Perceptron perceptron = new Perceptron(2, "tanh", 0.1, 15);
        perceptron.train(X_or, y_or);

        System.out.println("Predicciones OR:");
        for (double[] x : X_or) {
            System.out.println("Entrada: [" + x[0] + ", " + x[1] + "], Salida: " + perceptron.predict(x));
        }
    }
}
