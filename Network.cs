using System.Diagnostics;

public class Network
    {

        private Layer[] layers;

        // Spins up a new network
        // length of config array = # of layers in network. Value at each index = # of nodes in respective layer
        public Network(params int[] config)
        {
            if (config.Length <= 1)
            {
                throw new InvalidLayerCountException();
            }

            layers = new Layer[config.Length];

            // create layers
            for (int i = 0; i < config.Length; i++)
            {
                layers[i] = new Layer(config[i]);
            }

            // connect layers
            for (int i = 0; i < config.Length; i++)
            {
                if (i == 0)
                    layers[i].connect(null, layers[i+1]);
                else if (i == config.Length - 1)
                    layers[i].connect(layers[i-1], null);
                else
                    layers[i].connect(layers[i-1], layers[i+1]);
            }
        }


        // Takes a list of inputs, runs them through the network, and returns the list of outputs
        public double[] process(double[] inputValues)
        {

            layers[0].input(inputValues);

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].propagate();
            }

            return layers[layers.Length - 1].output();
        }


        public void printDebugStuff()
        {
            for (int i = 0; i < layers.Length; i++)
            {
                Console.BackgroundColor = ConsoleColor.Green;
                Console.WriteLine("Layer " + (i + 1) + ": ");
                Console.WriteLine();
                Console.BackgroundColor = ConsoleColor.Black;


                Console.ForegroundColor = ConsoleColor.Blue;
                Console.WriteLine("Weights: ");
                Console.ForegroundColor = ConsoleColor.White;
                for (int j = 0; j < layers[i].weights.GetLength(0); j++)
                {
                    for (int k = 0; k < layers[i].weights.GetLength(1); k++)
                    {
                        Console.Write(layers[i].weights[j,k] + ", ");
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Blue;
                Console.WriteLine("Biases: ");
                Console.ForegroundColor = ConsoleColor.White;
                for (int j = 0; j < layers[i].biases.Length; j++)
                {
                    Console.Write(layers[i].biases[j] + ", ");
                }
                Console.WriteLine();
                Console.WriteLine();

            }
        }

    }


    public class InvalidLayerCountException : Exception
    {
        public InvalidLayerCountException() {}
    }

    public class InvalidInputLengthException : Exception
    {
        public InvalidInputLengthException() {}
    }