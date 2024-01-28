public class Node
{
    private double inputValue;
    private double outputValue;


    // Takes a value in and stores it
    public void input(double input)
    {
        inputValue = input;
        outputValue = input;
    }


    // returns the output value
    public double output()
    {
        return outputValue;
    }


    // passes input through sigmoid activation function, stores output value
    public void activate()
    {

        outputValue = 1 / (1 + Math.Exp(-inputValue));
    }


    // calculates the cost function for a single node, based on the difference between the actual and desired output
    public double nodeCost(double expectedOutput)
    {
        double cost = outputValue - expectedOutput;
        return cost*cost;
    }

}