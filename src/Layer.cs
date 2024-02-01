using System;
using System.Globalization;

public class Layer
{

    private Node[] nodes;
    public int nodeCount {get; private set;}
    public Layer? previousLayer {get; private set;}
    public Layer? nextLayer {get; private set;}
    public double[,] weights { get; private set;}
    public double[] biases { get; private set;}
    public double[,] weightCostGradient;
    public double[] biasCostGradient;

    public Layer(int nodeCount)
    {

        this.nodeCount = nodeCount;
        nodes = new Node[nodeCount];


        for (int i = 0; i < nodeCount; i++)
        {
            nodes[i] = new Node();
        }

        weights = new double[0,0];
        biases = new double[0];
        weightCostGradient = new double[0,0];
        biasCostGradient = new double[0];

    }


    // attaches the previous and next layers, and creates the weights & biases array associated with this layer.
    public void connect(Layer? previousLayer, Layer? nextLayer)
    {
        this.previousLayer = previousLayer;
        this.nextLayer = nextLayer;

        initializeWeightsAndBiases();
    }


    // sets all weights and biases to random values.
    private void initializeWeightsAndBiases()
    {
        if (nextLayer == null)
            return;

        Random rand = new Random();

        weights = new double[nodeCount,nextLayer.nodeCount];
        biases = new double[nextLayer.nodeCount];
        weightCostGradient = new double[nodeCount, nextLayer.nodeCount];
        biasCostGradient = new double[nextLayer.nodeCount];

        for (int i = 0; i < nextLayer.nodeCount; i++)
        {
            biases[i] = rand.NextDouble();

            for (int j = 0; j < nodeCount; j++)
            {
                weights[j,i] = rand.NextDouble();
            }
        }
    }


    // takes an array of doubles and stores them in the nodes of this layer
    // isInputLayer adjusts whether the node inputs are passed through an activation function
    public void input(double[] inputs, bool isInputLayer)
    {
        if (inputs.Length != nodeCount)
            throw new InvalidInputLengthException();

        for (int i = 0; i < nodeCount; i++)
        {
            nodes[i].input(inputs[i], !isInputLayer);
        }
    }


    // applies weights and biases to the activation values in this layer and passes them to the inputs of the next layer
    public void propagate()
    {

        // if this is the output layer, do nothing
        if (nextLayer == null)
            return;

        double[] outputs = new double[nextLayer.nodeCount];

        for (int i = 0; i < nextLayer.nodeCount; i++)
        {
            outputs[i] = biases[i];
            for (int j = 0; j < nodeCount; j++)
            {
                outputs[i] += nodes[j].output() * weights[j,i];
            }
        }

        // previousLayer == null if this is the input layer
        nextLayer.input(outputs, previousLayer==null);

    }


    // returns an array containing the output values of each node in this layer
    public double[] output()
    {
        double[] outputs = new double[nodeCount];
        for (int i = 0; i < nodeCount; i++)
        {
            outputs[i] = nodes[i].output();
        }

        return outputs;
    }


    // takes a single training data point and calculates the total cost based on the difference between the actual and desired outputs
    public double layerCost(TrainingDataPoint dataPoint)
    {
        double cost = 0.0;

        for (int i = 0; i < dataPoint.expectedOutputs.Length; i++)
        {
            cost += nodes[i].nodeCost(dataPoint.expectedOutputs[i]);
        }

        return cost;
    }


    public void calculateGradients(double[] expectedOutput)
    {
        // output layer is starting point for backprop algorithm, so it gets treated a little bit differently from the others
        if (nextLayer == null && previousLayer != null)
        {
            for (int i = 0; i < previousLayer.nodeCount; i++)
            {
                for (int j = 0; j < nodeCount; j++)
                {
                    previousLayer.biasCostGradient[j] += nodes[j].partialGradient(expectedOutput[j]);
                    previousLayer.weightCostGradient[i,j] += previousLayer.nodes[i].output() * nodes[j].partialGradient(expectedOutput[j]);
                }
            }
        }


        else if (nextLayer != null && previousLayer != null)
        {

            for (int i = 0; i < nextLayer.nodeCount; i++)
            {
                for (int j = 0; j < nodeCount; j++)
                {
                    nodes[j].nodePartialGradient += weights[j,i] * nextLayer.nodes[i].nodePartialGradient;
                }
            }

            for (int i = 0; i < nodeCount; i++)
            {
                nodes[i].nodePartialGradient *= nodes[i].activationDerivative();

                for (int j = 0; j < previousLayer.nodeCount; j++)
                {
                    previousLayer.weightCostGradient[j,i] += nodes[i].nodePartialGradient * previousLayer.nodes[j].output();
                    previousLayer.biasCostGradient[i] += nodes[i].nodePartialGradient;
                }
            }
        }
    }


    // applies the weight/bias cost gradients to the weights and biases of the layer
    public void applyGradients(double learnRate)
    {
        if (nextLayer == null)
            return;

        for (int i = 0; i < nextLayer.nodeCount; i++)
        {
            biases[i] -= biasCostGradient[i] * learnRate;

            for (int j = 0; j < nodeCount; j++)
            {
                weights[j,i] -= weightCostGradient[j,i] * learnRate;
            }
        }

    }


    // resets the weights and bias gradients to 0
    public void clearGradients()
    {
        for (int i = 0; i < weightCostGradient.GetLength(0); i++)
        {
            for (int j = 0; j < weightCostGradient.GetLength(1); j++)
            {
                weightCostGradient[i,j] = 0;
            }
        }

        for (int i = 0; i < biasCostGradient.Length; i++)
        {
            biasCostGradient[i] = 0;
        }
    }

}


