public class Connection
{
    private Layer leftLayer;
    private Layer rightLayer;

    private double[,] weights;
    private double[] biases;
    private double[,] weightCostGradients;
    private double[] biasCostGradients;

    Random rand = new Random();

    public Connection(Layer leftLayer, Layer rightLayer)
    {
        this.leftLayer = leftLayer;
        this.rightLayer = rightLayer;

        weights = new double[leftLayer.nodeCount,rightLayer.nodeCount];
        biases = new double[rightLayer.nodeCount];
        weightCostGradients = new double[leftLayer.nodeCount,rightLayer.nodeCount];
        biasCostGradients = new double[rightLayer.nodeCount];

        initializeWeightsAndBiases();
    }


    // sets all weights and biases to random values
    private void initializeWeightsAndBiases()
    {

        for (int i = 0; i < rightLayer.nodeCount; i++)
        {
            biases[i] = rand.NextDouble();

            for (int j = 0; j < leftLayer.nodeCount; j++)
            {
                weights[j,i] = rand.NextDouble();
            }
        }

    }


    // applies weights and biases to the activation values in this layer and passes them to the inputs of the next layer
    public void propagate()
    {
        double[] inputs = leftLayer.output();
        double[] outputs = new double[rightLayer.nodeCount];

        for (int i = 0; i < outputs.Length; i++)
        {
            outputs[i] = biases[i];

            for (int j = 0; j < inputs.Length; j++)
            {
                outputs[i] += inputs[j] * weights[j,i];
            }
        }

        rightLayer.input(outputs);

    }


    // calculates partial gradient values for the output layer
    public void calculateOutputLayerPartials(TrainingDataPoint dataPoint)
    {
        double[] outputLayerActivationDerivatives = rightLayer.getActivationDerivatives();
        double[] outputLayerCostDerivatives = rightLayer.getLayerCostDerivatives(dataPoint);
        double[] outputLayerPartialGradients = new double[rightLayer.nodeCount];

        for (int i = 0; i < rightLayer.nodeCount; i++)
        {
            outputLayerPartialGradients[i] = outputLayerActivationDerivatives[i] * outputLayerCostDerivatives[i];
        }

        rightLayer.storePartialGradients(outputLayerPartialGradients);
    }


    // calculates weight & bias gradients for hidden layers
    public void calculateGradients()
    {
        double[] leftLayerOutputs = leftLayer.output();

        // get right layer partials
        double[] rightLayerPartialGradients = rightLayer.getPartialGradients();

        // calculate and set weight & bias gradients
        for (int i = 0; i < rightLayer.nodeCount; i++)
        {
            biasCostGradients[i] += rightLayerPartialGradients[i];

            for (int j = 0; j < leftLayer.nodeCount; j++)
            {
                weightCostGradients[j,i] += rightLayerPartialGradients[i] * leftLayerOutputs[j];
            }
        }


        // set left layer partials
        double[] leftLayerPartialGradients = new double[leftLayer.nodeCount];
        double[] leftLayerActivationDerivatives = leftLayer.getActivationDerivatives();

        for (int i = 0; i < leftLayer.nodeCount; i++)
        {
            for (int j = 0; j < rightLayer.nodeCount; j++)
            {
                leftLayerPartialGradients[i] += weights[i,j] * rightLayerPartialGradients[j];
            }
        }

        for (int i = 0; i < leftLayer.nodeCount; i++)
        {
            leftLayerPartialGradients[i] *= leftLayerActivationDerivatives[i];
        }

        leftLayer.storePartialGradients(leftLayerPartialGradients);
    }


    // applies the weight/bias cost gradients to the weights and biases of the layer
    public void applyGradients(double learnRate)
    {
        double noiseValue = 0;

        for (int i = 0; i < rightLayer.nodeCount; i++)
        {
            biases[i] -= biasCostGradients[i] * learnRate;

            for (int j = 0; j < leftLayer.nodeCount; j++)
            {
                // small chance to add some random noise to the weight
                if (rand.NextDouble() < 0.05)
                {
                    noiseValue = rand.NextDouble() * 0.1 * learnRate;
                }

                weights[j,i] -= (weightCostGradients[j,i] + noiseValue) * learnRate;
            }
        }
    }


    // resets the weights and bias gradients to 0
    public void clearGradients()
    {
        for (int i = 0; i < weightCostGradients.GetLength(1); i++)
        {
            biasCostGradients[i] = 0;

            for (int j = 0; j < weightCostGradients.GetLength(0); j++)
            {
                weightCostGradients[j,i] = 0;
            }
        }
    }


}
