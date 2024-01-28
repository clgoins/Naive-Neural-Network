
public class Program
{



    public static void Main()
    {
        Network network = new Network(1,2,1);

        double[] input = new double[1];

        input[0] = 1.0;

        double[] output = network.process(input);


        network.printDebugStuff();
        Console.WriteLine("Final Output: " + output[0]);

    }


}