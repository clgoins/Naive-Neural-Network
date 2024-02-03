using System;
using System.Globalization;

public class Layer
{

    private Node[] nodes;
    public int nodeCount {get; private set;}

    public Layer(int nodeCount)
    {

        this.nodeCount = nodeCount;
        nodes = new Node[nodeCount];


        for (int i = 0; i < nodeCount; i++)
        {
            nodes[i] = new Node();
        }


    }


    // takes an array of doubles and stores them in the nodes of this layer
    // isInputLayer adjusts whether the node inputs are passed through an activation function
    public void input(double[] inputs, bool isInputLayer = false)
    {
        if (inputs.Length != nodeCount)
            throw new InvalidInputLengthException();

        for (int i = 0; i < nodeCount; i++)
        {
            nodes[i].input(inputs[i], !isInputLayer);
        }
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


    // derivative of activation function w/ respect to weighted input
    public double[] getActivationDerivatives()
    {
        double[] output = new double[nodeCount];

        for (int i = 0; i < nodeCount; i++)
        {
            output[i] = nodes[i].activationDerivative();
        }

        return output;
    }


    // derivative of cost function w/ respect to outputs
    public double[] getLayerCostDerivatives(TrainingDataPoint dataPoint)
    {
        double[] output = new double[nodeCount];

        for (int i = 0; i < nodeCount; i++)
        {
            output[i] = nodes[i].nodeCostDerivative(dataPoint.expectedOutputs[i]);
        }

        return output;
    }


    // takes a list of doubles and stores them in the nodePartialGradient field in each node
    public void storePartialGradients(double[] partials)
    {
        for (int i = 0; i < nodeCount; i++)
        {
            nodes[i].nodePartialGradient = partials[i];
        }
    }


    // gets the nodePartialGradient values from each node and returns them as an array
    public double[] getPartialGradients()
    {
        double[] output = new double[nodeCount];

        for (int i = 0; i < nodeCount; i++)
        {
            output[i] = nodes[i].nodePartialGradient;
        }

        return output;
    }


    // chooses an activation function for the layer, sets each node in the layer accordingly
    public void setActivationFunction(activationFunctions function)
    {
        foreach (Node node in nodes)
        {
            node.chooseActivationFunction(function);
        }
    }

}


