
public class Program
{



    // TODO: get the batch size, learn rate, and training data path as command line arguments
    // TODO: Split the input file into two to get the training and testing data from the same file
    // TODO: Write a couple input normalization functions (i.e. shift data so it's all positive, scale data so it falls between 0 and 1, etc)
    public static void Main(string[] args)
    {

        // create new network
        Network network = new Network(2,3,3,3,2);
        network.setActivationFunction(activationFunctions.tanh);



        /////////////////////////////
        ////////TRAINING/////////////
        /////////////////////////////

        // load training data from file
        //TrainingDataPoint[] trainingData = loadTrainingDataFromFile("./data/quadratic data/dataBIG0.csv", 2, 2);
        int[] trainingData = loadMNIST("./data/MNIST/");
        //trainingData = normalize(trainingData);

/*
        int batchSize = 100;

        // runs gradient descent algorithm until a key is pressed
        while(! Console.KeyAvailable)
        {
            int batchCount = 0;
            trainingData = shuffle(trainingData);

            while (batchCount * batchSize < trainingData.Length)
            {
                TrainingDataPoint[] batch = new TrainingDataPoint[batchSize];

                Array.Copy(trainingData, batchCount * batchSize, batch, 0, batchSize);

                double cost = network.train(trainingData, 0.05);
                Console.WriteLine("Cost: " + cost);

                batchCount++;
            }
        }



        /////////////////////////////
        ///////TESTING///////////////
        /////////////////////////////


        TrainingDataPoint[] testingData = loadTrainingDataFromFile("./data/quadratic data/dataBIG1.csv", 2, 2);

        double pointsCorrect = 0;
        double pointsCounted = 0;

        foreach (TrainingDataPoint point in testingData)
        {
            double[] actualOutputs = network.processRaw(point.inputs);

            if (actualOutputs[0] < actualOutputs[1] && point.expectedOutputs[0] < point.expectedOutputs[1])
                pointsCorrect++;
            else if (actualOutputs[0] > actualOutputs[1] && point.expectedOutputs[0] > point.expectedOutputs[1])
                pointsCorrect++;

            pointsCounted++;

            Console.WriteLine("Accuracy: " + String.Format("{0:0.00}", pointsCorrect / pointsCounted));

        }
*/

        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                Console.Write(trainingData[i]);
            }
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


    // Loads training data from the MNIST data set
    public static int[] loadMNIST(string path)
        {
        int[] data = new int[28*28];

        using(BinaryReader reader = new BinaryReader(new FileStream(path + "train-images.idx3-ubyte", FileMode.Open)))
        {
            //skip the file header
            reader.BaseStream.Seek(16, SeekOrigin.Begin);

            // temporary place to store the image data
            int[] nextImage = new int[28*28];

            // PeekChar() returns -1 at EOF
            while (reader.PeekChar() != -1)
            {

                for (int i = 0; i < 28*28; i++)
                {
                    Console.WriteLine(reader.Read());
                }

            }
        }

        return data;
    }


    //Shuffles the array of training data according to Fisher-Yates algorithm
    public static TrainingDataPoint[] shuffle(TrainingDataPoint[] data)
    {
        TrainingDataPoint[] shuffled = data;

        int n = shuffled.Length;
        Random rand = new Random();
        while(n > 1)
        {
            int k = rand.Next(n--);
            TrainingDataPoint temp = shuffled[n];
            shuffled[n] = shuffled[k];
            shuffled[k] = temp;
        }
        return shuffled;
    }



}