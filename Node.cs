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


    // passes input through linear activation function, stores output value
    public void activate()
    {
        outputValue = Math.Max(0,inputValue);
    }

}