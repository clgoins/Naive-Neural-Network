
using System.Runtime.InteropServices.ComTypes;
using System.Diagnostics.Metrics;

public class Program
{



    public static void Main(string[] args)
    {

        // create new network
        Network network = new Network(2,4,4,2);



        /////////////////////////////
        ////////TRAINING/////////////
        /////////////////////////////

        // load training data from file
        TrainingDataPoint[] trainingData = loadTrainingDataFromFile("./data/linear data/dataBIG0.csv", 2, 2);

        //network.process(trainingData[trainingData.Length-1].inputs);

        //network.printDebugStuff();

        // runs gradient descent algorithm until a key is pressed

        int batchSize = 100;

        while(! Console.KeyAvailable)
        {
            int batchCount = 0;

            while (batchCount * batchSize < trainingData.Length)
            {
                TrainingDataPoint[] batch = new TrainingDataPoint[batchSize];

                Array.Copy(trainingData, batchCount * batchSize, batch, 0, batchSize);

                double cost = network.train(batch, 0.0005);
                Console.WriteLine("Cost: " + cost);

                batchCount++;
            }
        }

        //network.printDebugStuff();






        /////////////////////////////
        ///////TESTING///////////////
        /////////////////////////////


        TrainingDataPoint[] testingData = loadTrainingDataFromFile("./data/quadratic data/dataBIG1.csv", 2, 2);

        double pointsCorrect = 0;
        double pointsCounted = 0;

        foreach (TrainingDataPoint point in testingData)
        {
            double[] actualOutputs = network.process(point.inputs);

            if (actualOutputs[0] < actualOutputs[1] && point.expectedOutputs[0] < point.expectedOutputs[1])
                pointsCorrect++;
            else if (actualOutputs[0] > actualOutputs[1] && point.expectedOutputs[0] > point.expectedOutputs[1])
                pointsCorrect++;

            pointsCounted++;

            Console.WriteLine("Accuracy: " + String.Format("{0:0.00}", pointsCorrect / pointsCounted));

        }

    }








    // Creates an array of TrainingDataPoints from a .csv file
    // Each point should be on its own line, each value separated by a comma, and with the inputs appearing first on the line.
    public static TrainingDataPoint[] loadTrainingDataFromFile(string filename, int inputCount, int outputCount)
    {
        List<TrainingDataPoint> data = new List<TrainingDataPoint>();

        using(StreamReader reader = new StreamReader(filename))
        {
            while (!reader.EndOfStream)
            {
                double[] inputs = new double [inputCount];
                double[] outputs = new double [outputCount];

                string? line = reader.ReadLine();
                if (line == null)
                    break;

                string[] values = line.Split(',');

                for(int i = 0; i < inputCount; i++)
                {
                    inputs[i] = Convert.ToDouble(values[i]);
                }

                for(int i = 0; i < outputCount; i++)
                {
                    outputs[i] = Convert.ToDouble(values[i + inputCount]);
                }

                TrainingDataPoint newData = new TrainingDataPoint(inputs,outputs);
                data.Add(newData);
            }
        }

        return data.ToArray();
    }


    // Shifts all training data equally such that all input values are positive
    public static void MakePositive(TrainingDataPoint[] data)
    {
        double[] smallestValues = new double[data[0].inputs.Length];

        // find smallest value on each axis
        foreach (TrainingDataPoint point in data)
        {
            for (int i = 0; i < point.inputs.Length; i++)
            {
                if (point.inputs[i] < smallestValues[i])
                    smallestValues[i] = point.inputs[i];
            }
        }

        // shift all values to above 0
        foreach (TrainingDataPoint point in data)
        {
            for (int i = 0; i < point.inputs.Length; i++)
            {
                if (smallestValues[i] >= 0)
                    continue;

                point.inputs[i] += Math.Abs(smallestValues[i]);
            }
        }


    }


    // Scales all training data such that all values are within 1 and 0
    //public static TrainingDataPoint[] Normalize(TrainingDataPoint[] data)
    //{

    //}
}