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

            layers[0].input(inputValues,true);

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].propagate();
            }

            return layers[layers.Length - 1].output();
        }


        public void printDebugStuff()
        {
            Console.WriteLine();

            for (int i = 0; i < layers.Length; i++)
            {
                //print layer number
                Console.BackgroundColor = ConsoleColor.Green;
                Console.WriteLine("Layer " + (i + 1) + ": ");
                Console.WriteLine();
                Console.BackgroundColor = ConsoleColor.Black;

                //print activation values
                Console.ForegroundColor = ConsoleColor.Blue;
                Console.WriteLine("Activations: ");
                Console.ForegroundColor = ConsoleColor.White;
                double[] activationValues = layers[i].output();
                for (int j = 0; j < activationValues.Length; j++)
                {
                    Console.Write(activationValues[j] + ", ");
                }
                Console.WriteLine();
                Console.WriteLine();

                //print weight values
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

                //print bias values
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


        // runs a set of training data through the network, and calculates the average cost over the entire set
        public double cost(TrainingDataPoint[] data)
        {
            double cost = 0;

            foreach (TrainingDataPoint point in data)
            {
                process(point.inputs);
                cost += layers[layers.Length - 1].layerCost(point);
            }

            return cost / data.Length;
        }


        // calculates cost of network, makes small tweaks to weights and biases, determining how the cost changes each time, and takes a single step in the direction that will reduce cost the most
        public double train(TrainingDataPoint[] data, double learnRate)
        {
            const double h = 0.00001;
            double originalCost = cost(data);

            foreach(Layer layer in layers)
            {
                for (int i = 0; i < layer.weights.GetLength(0); i++)
                {
                        for (int j = 0; j < layer.weights.GetLength(1); j++)
                        {
                            layer.weights[i,j] += h;
                            double deltaCost = cost(data) - originalCost;
                            layer.weights[i,j] -= h;
                            layer.weightCostGradient[i,j] = deltaCost/h;
                        }
                }

                for (int i = 0; i < layer.biases.Length; i++)
                {
                    layer.biases[i] += h;
                    double deltaCost = cost(data) - originalCost;
                    layer.biases[i] -= h;
                    layer.biasCostGradient[i] = deltaCost/h;
                }
            }

            applyAllGradients(learnRate);

            return cost(data);
        }


        // applies all weight and bias gradients to every layer in the network
        public void applyAllGradients(double learnRate)
        {
            foreach (Layer layer in layers)
            {
                layer.applyGradients(learnRate);
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