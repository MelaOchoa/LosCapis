/// FUNCIÓN LINEAL
using System;

class Perceptron
{
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private string activationFunction;

    public Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20)
    {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = rand.NextDouble() * 2 - 1; // Valores entre -1 y 1
        }
        bias = rand.NextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double Activation(double x)
    {
        if (activationFunction == "lineal")
        {
            return x;
        }
        return x; // Se pueden agregar más funciones de activación aquí
    }

    public double Predict(double[] X)
    {
        double linearOutput = bias;
        for (int i = 0; i < X.Length; i++)
        {
            linearOutput += X[i] * weights[i];
        }
        return Activation(linearOutput);
    }

    public void Train(double[][] X, double[] y)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double prediction = Predict(X[i]);
                double error = y[i] - prediction;
                for (int j = 0; j < weights.Length; j++)
                {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.Abs(error);
            }
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError}");
        }
    }
}

class Program
{
    static void Main()
    {
        double[][] X_or = new double[][]
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        double[] y_or = { 0, 1, 1, 1 };

        Perceptron perceptron = new Perceptron(inputSize: 2, activationFunction: "lineal", learningRate: 0.1, epochs: 15);
        perceptron.Train(X_or, y_or);

        Console.WriteLine("Predicciones OR:");
        foreach (var x in X_or)
        {
            Console.WriteLine($"Entrada: [{x[0]}, {x[1]}], Salida: {perceptron.Predict(x)}");
        }
    }
}

/// FUNCIÓN ESCALÓN
using System;

class Perceptron
{
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private string activationFunction;

    public Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20)
    {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = rand.NextDouble() * 2 - 1; // Valores entre -1 y 1
        }
        bias = rand.NextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double Activation(double x)
    {
        if (activationFunction == "escalon")
        {
            return x >= 0 ? 1 : 0;
        }
        return x; // Se pueden agregar más funciones de activación aquí
    }

    public double Predict(double[] X)
    {
        double linearOutput = bias;
        for (int i = 0; i < X.Length; i++)
        {
            linearOutput += X[i] * weights[i];
        }
        return Activation(linearOutput);
    }

    public void Train(double[][] X, double[] y)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double prediction = Predict(X[i]);
                double error = y[i] - prediction;
                for (int j = 0; j < weights.Length; j++)
                {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.Abs(error);
            }
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError}");
        }
    }
}

class Program
{
    static void Main()
    {
        double[][] X_or = new double[][]
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        double[] y_or = { 0, 1, 1, 1 };

        Perceptron perceptron = new Perceptron(inputSize: 2, activationFunction: "escalon", learningRate: 0.1, epochs: 15);
        perceptron.Train(X_or, y_or);

        Console.WriteLine("Predicciones OR:");
        foreach (var x in X_or)
        {
            Console.WriteLine($"Entrada: [{x[0]}, {x[1]}], Salida: {perceptron.Predict(x)}");
        }
    }
}
/// FUNCIÓN SIGMOIDAL
using System;
class Perceptron
{
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private string activationFunction;

    public Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20)
    {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = rand.NextDouble() * 2 - 1; // Valores entre -1 y 1
        }
        bias = rand.NextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }
    private double Activation(double x)
    {
        if (activationFunction == "sigmoide")
        {
            return 1 / (1 + Math.Exp(-x));
        }
        return x; // Se pueden agregar más funciones de activación aquí
    }
    public double Predict(double[] X)
    {
        double linearOutput = bias;
        for (int i = 0; i < X.Length; i++)
        {
            linearOutput += X[i] * weights[i];
        }
        return Activation(linearOutput);
    }
    public void Train(double[][] X, double[] y)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double prediction = Predict(X[i]);
                double error = y[i] - prediction;
                for (int j = 0; j < weights.Length; j++)
                {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.Abs(error);
            }
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError}");
        }
    }
}
class Program
{
    static void Main()
    {
        double[][] X_or = new double[][]
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        double[] y_or = { 0, 1, 1, 1 };
        Perceptron perceptron = new Perceptron(inputSize: 2, activationFunction: "sigmoide", learningRate: 0.1, epochs: 15);
        perceptron.Train(X_or, y_or);

        Console.WriteLine("Predicciones OR:");
        foreach (var x in X_or)
        {
            Console.WriteLine($"Entrada: [{x[0]}, {x[1]}], Salida: {perceptron.Predict(x)}");
        }
    }
}
/// FUNCIÓN RELU
using System;

class Perceptron
{
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private string activationFunction;

    public Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20)
    {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = rand.NextDouble() * 2 - 1; 
        }
        bias = rand.NextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double Activation(double x)
    {
        if (activationFunction == "relu")
        {
            return Math.Max(0, x);
        }
        return x;
    }

    public double Predict(double[] X)
    {
        double linearOutput = 0;
        for (int i = 0; i < X.Length; i++)
        {
            linearOutput += X[i] * weights[i];
        }
        linearOutput += bias;
        return Activation(linearOutput);
    }

    public void Train(double[][] X, double[] y)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double prediction = Predict(X[i]);
                double error = y[i] - prediction;
                for (int j = 0; j < weights.Length; j++)
                {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.Abs(error);
            }
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError}");
        }
    }

    public static void Main()
    {
        double[][] X_or = new double[][]
        {
            new double[] {0, 0},
            new double[] {0, 1},
            new double[] {1, 0},
            new double[] {1, 1}
        };
        double[] y_or = new double[] {0, 1, 1, 1};

        Perceptron perceptron_or = new Perceptron(inputSize: 2, activationFunction: "relu", learningRate: 0.1, epochs: 15);
        perceptron_or.Train(X_or, y_or);

        Console.WriteLine("Predicciones OR:");
        foreach (var x in X_or)
        {
            Console.WriteLine($"Entrada: [{x[0]}, {x[1]}], Salida: {perceptron_or.Predict(x)}");
        }
    }
}

/// FUNCIÓN SOFTMAX
using System;
using System.Linq;

class Perceptron
{
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private string activationFunction;

    public Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20)
    {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
            weights[i] = rand.NextDouble() * 2 - 1;
        bias = rand.NextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double[] Activation(double[] x)
    {
        if (activationFunction == "softmax")
        {
            double sumExp = x.Sum(Math.Exp);
            return x.Select(v => Math.Exp(v) / sumExp).ToArray();
        }
        return x;
    }

    public double[] Predict(double[] X)
    {
        double linearOutput = X.Zip(weights, (xi, wi) => xi * wi).Sum() + bias;
        return Activation(new double[] { linearOutput });
    }

    public void Train(double[][] X, double[] y)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double[] prediction = Predict(X[i]);
                double error = y[i] - prediction[0];
                for (int j = 0; j < weights.Length; j++)
                    weights[j] += learningRate * error * X[i][j];
                bias += learningRate * error;
                totalError += Math.Abs(error);
            }
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError}");
        }
    }
}

class Program
{
    static void Main()
    {
        double[][] X_or = new double[][]
        {
            new double[] {0, 0},
            new double[] {0, 1},
            new double[] {1, 0},
            new double[] {1, 1}
        };
        double[] y_or = new double[] { 0, 1, 1, 1 };

        Perceptron perceptron_or = new Perceptron(2, "softmax", 0.1, 15);
        perceptron_or.Train(X_or, y_or);

        Console.WriteLine("Predicciones OR:");
        foreach (var x in X_or)
        {
            Console.WriteLine($"Entrada: [{string.Join(", ", x)}], Salida: {string.Join(", ", perceptron_or.Predict(x))}");
        }
    }
}

/// FUNCIÓN TANGENTE HIPERBÓLICA
using System;
using System.Linq;

class Perceptron
{
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private string activationFunction;

    public Perceptron(int inputSize, string activationFunction, double learningRate = 0.01, int epochs = 20)
    {
        Random rand = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = rand.NextDouble() * 2 - 1; // Valores entre -1 y 1
        }
        bias = rand.NextDouble() * 2 - 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activationFunction = activationFunction;
    }

    private double Activation(double x)
    {
        if (activationFunction == "tanh")
        {
            return Math.Tanh(x);
        }
        return x; // Por defecto lineal
    }

    public double Predict(double[] X)
    {
        double linearOutput = X.Zip(weights, (x, w) => x * w).Sum() + bias;
        return Activation(linearOutput);
    }

    public void Train(double[][] X, double[] y)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double prediction = Predict(X[i]);
                double error = y[i] - prediction;

                for (int j = 0; j < weights.Length; j++)
                {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
                totalError += Math.Abs(error);
            }
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError}");
        }
    }
}

class Program
{
    static void Main()
    {
        double[][] X_or = new double[][]
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };

        double[] y_or = { 0, 1, 1, 1 };

        Perceptron perceptronAnd = new Perceptron(inputSize: 2, activationFunction: "tanh", learningRate: 0.1, epochs: 15);
        perceptronAnd.Train(X_or, y_or);

        Console.WriteLine("Predicciones OR:");
        foreach (var x in X_or)
        {
            Console.WriteLine($"Entrada: [{string.Join(", ", x)}], Salida: {perceptronAnd.Predict(x)}");
        }
    }
}

