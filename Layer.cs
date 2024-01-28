using System;

public class Layer
{

    private Node[] nodes;
    public int nodeCount {get; private set;}
    private Layer? previousLayer;
    private Layer? nextLayer;
    public double[,] weights { get; private set;}
    public double[] biases { get; private set;}

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
    public void input(double[] inputs)
    {
        if (inputs.Length != nodeCount)
            throw new InvalidInputLengthException();

        for (int i = 0; i < nodeCount; i++)
        {
            nodes[i].input(inputs[i]);
        }
    }


    // activates & weights the nodes in this layer and passes the outputs to the next layer
    public void propagate()
    {
        // if this is the input layer, skip the activation step
        if (previousLayer != null)
            activateNodes();

        // if this is not the output layer, weight the outputs of this layer and pass them to the input of the next layer
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

        nextLayer.input(outputs);

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


    // Iterates each node in the layer and calls its activation function
    private void activateNodes()
    {
        foreach (Node node in nodes)
        {
            node.activate();
        }
    }

}


