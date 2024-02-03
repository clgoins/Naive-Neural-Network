using System.Diagnostics;

    public enum activationFunctions
    {
        relu, leaky, sigmoid, tanh
    }

public class Network
    {



        private Layer[] layers;
        private Connection[] connections;

        // Spins up a new network
        // length of config array = # of layers in network. Value at each index = # of nodes in respective layer
        public Network(params int[] config)
        {

            if (config.Length <= 1)
            {
                throw new InvalidLayerCountException();
            }

            layers = new Layer[config.Length];
            connections = new Connection[config.Length - 1];

            // create layers
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(config[i]);
            }

            // create connections
            for (int i = 0; i < connections.Length; i++)
            {
                connections[i] = new Connection(layers[i], layers[i + 1]);
            }
        }


        // Takes a list of inputs, runs them through the network, and returns the list of outputs
        public double[] processRaw(double[] inputValues)
        {

            layers[0].input(inputValues,true);

            for (int i = 0; i < connections.Length; i++)
            {
                connections[i].propagate();
            }

            return layers[layers.Length - 1].output();
        }


        // Process a list of inputs, returns a single int representing the index of the output node with the greatest value
        public int process(double[] inputValues)
        {
            layers[0].input(inputValues,true);

            for (int i = 0; i < connections.Length; i++)
            {
                connections[i].propagate();
            }

            int maxIndex = 0;
            double[] outputs = layers[layers.Length - 1].output();

            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] > maxIndex)
                    maxIndex = i;
            }

            return maxIndex;
        }


        // chooses the activation function to use for the network
        public void setActivationFunction(activationFunctions activationFunction)
        {
            foreach(Layer layer in layers)
            {
                layer.setActivationFunction(activationFunction);
            }
        }


        // chooses an activation function specific to the output layer
        public void setOutputLayerActivationFunction(activationFunctions activationFunction)
        {
            layers[layers.Length - 1].setActivationFunction(activationFunction);
        }


        // runs a set of training data through the network, and calculates the average cost over the entire set
        public double cost(TrainingDataPoint[] data)
        {
            double cost = 0;

            foreach (TrainingDataPoint point in data)
            {
                processRaw(point.inputs);
                cost += layers[layers.Length - 1].layerCost(point);
            }

            return cost / data.Length;
        }


        // calculates cost of network, makes small tweaks to weights and biases, determining how the cost changes each time, and takes a single step in the direction that will reduce cost the most
        public double train(TrainingDataPoint[] data, double learnRate)
        {

            double cost = 0;

            foreach (TrainingDataPoint point in data)
            {
                processRaw(point.inputs);
                cost += layers[layers.Length - 1].layerCost(point);
                calculateAllGradients(point);
            }

            applyAllGradients(learnRate / data.Length);
            clearAllGradients();

            return cost / data.Length;
        }

        private void calculateAllGradients(TrainingDataPoint dataPoint)
        {
            connections[connections.Length - 1].calculateOutputLayerPartials(dataPoint);

            for (int i = connections.Length - 1; i >= 0; i--)
            {
                connections[i].calculateGradients();
            }
        }


        // applies all weight and bias gradients to every layer in the network
        private void applyAllGradients(double learnRate)
        {
            foreach (Connection connection in connections)
            {
                connection.applyGradients(learnRate);
            }
        }


        // Resets all weight and bias gradients to 0 for the next learning iteration
        private void clearAllGradients()
        {
            foreach (Connection connection in connections)
            {
                connection.clearGradients();
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