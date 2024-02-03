using System.Security.Cryptography.X509Certificates;

public class Node
{
    private double weightedInput;
    private double activationValue;
    public double nodePartialGradient {get; set;}
    public activationFunctions activationFunction;

    public Node()
    {
        activationFunction = activationFunctions.sigmoid;
    }

    // Takes a weighted input in, activates the node, and stores the input & activation value
    // ActivateNode should be false for nodes in the input layer, and true for all other layers
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
        switch(activationFunction)
        {
            case activationFunctions.sigmoid:
                return 1 / (1 + Math.Exp(-weightedInput));    //sigmoid

            case activationFunctions.tanh:
                return Math.Tanh(weightedInput);              //tanh

            case activationFunctions.relu:
                return Math.Max(0,weightedInput);             //reLU

            case activationFunctions.leaky:
                return weightedInput >= 0 ? weightedInput : weightedInput * 0.01;   //leaky reLU

            default:
                return 1 / (1 + Math.Exp(-weightedInput));    //default to sigmoid


        }
    }


    public double activationDerivative()
    {
        switch(activationFunction)
        {
            case activationFunctions.relu:
                return weightedInput <= 0 ? 0 : 1;  //reLU

            case activationFunctions.leaky:
                return weightedInput <= 0 ? 0.01 : 1;   //leaky reLU

            case activationFunctions.sigmoid:
                return 1 / (1 + Math.Exp(-weightedInput)) * (1 - 1 / (1 + Math.Exp(-weightedInput)));   //sigmoid

            case activationFunctions.tanh:
                return 1 - (Math.Tanh(weightedInput) * Math.Tanh(weightedInput));   //tanh

            default:
                return 1 / (1 + Math.Exp(-weightedInput)) * (1 - 1 / (1 + Math.Exp(-weightedInput)));   //default to sigmoid

        }
    }


    // calculates the cost function for a single node, based on the difference between the actual and desired output
    public double nodeCost(double expectedOutput)
    {
        double cost = activationValue - expectedOutput;
        return cost*cost;
    }


    // derivative of node cost w/ respect to outputValue
    public double nodeCostDerivative(double expectedOutput)
    {
        double cost = 2 * (activationValue - expectedOutput);
        return cost;
    }


    public void chooseActivationFunction(activationFunctions function)
    {
        activationFunction = function;
    }

}