public class Node
{
    private double weightedInput;
    private double activationValue;
    public double nodePartialGradient;

    // Takes a weighted input in, activates the node, and stores the input & activation value
    public void input(double input, bool activateNode)
    {
        weightedInput = input;

        if(activateNode)
            activationValue = activate();
        else
            activationValue = weightedInput;
    }


    // returns the output value
    public double output()
    {
        return activationValue;
    }


    // passes input through sigmoid activation function, stores output value
    private double activate()
    {
        return 1 / (1 + Math.Exp(-weightedInput));    //sigmoid
        //return Math.Tanh(weightedInput);              //tanh
        //return Math.Max(0,weightedInput);             //relu
    }


    //TODO: tanh & relu derivatives
    public double activationDerivative()
    {
        return 1 / (1 + Math.Exp(-weightedInput)) * (1 - 1 / (1 + Math.Exp(-weightedInput)));   //sigmoid
        //return 1 - (Math.Tanh(weightedInput) * Math.Tanh(weightedInput));   //tanh
        //return weightedInput <= 0 ? 0 : 1;  //relu
    }


    // calculates the cost function for a single node, based on the difference between the actual and desired output
    public double nodeCost(double expectedOutput)
    {
        double cost = activationValue - expectedOutput;
        return cost*cost;
    }


    // derivative of node cost w/ respect to outputValue
    public double nodeCostDerr(double expectedOutput)
    {
        double cost = 2 * (activationValue - expectedOutput);
        return cost;
    }


    // calculates, stores and returns part of the weight/bias gradient calculation
    // used on output layer as the start of the backpropagation algorithm
    public double partialGradient(double expectedOutput)
    {
        nodePartialGradient = nodeCostDerr(expectedOutput) * activationDerivative();
        return nodePartialGradient;
    }

}