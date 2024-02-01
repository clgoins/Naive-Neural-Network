public class Node
{
    private double weightedInput;
    private double activationValue;
    private double nodePartialGradient;


    // Takes a weighted input in, activates the node, and stores the input & activation value
    public void input(double input, bool activateNode)
    {
        weightedInput = input;

        if(activateNode)
            activationValue = activate(weightedInput);
        else
            activationValue = weightedInput;
    }


    // returns the output value
    public double output()
    {
        return activationValue;
    }


    // passes input through sigmoid activation function, stores output value
    public double activate(double value)
    {
        return 1 / (1 + Math.Exp(-value));    //sigmoid
        //return Math.Tanh(value);              //tanh
        //return = Math.Max(0,value);             //relu
    }


    //TODO: tanh & relu derivatives
    public double activationDerivative(double value)
    {
        return 1 / (1 + Math.Exp(-value)) * (1 - 1 / (1 + Math.Exp(-value)));   //sigmoid

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

}