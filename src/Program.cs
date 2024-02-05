
using System.Diagnostics;

public class Program
{



    // TODO: get the batch size, learn rate, and training data path as command line arguments
    // TODO: Split the input file into two to get the training and testing data from the same file
    // TODO: Write a couple input normalization functions (i.e. shift data so it's all positive, scale data so it falls between 0 and 1, etc)
    public static void Main(string[] args)
    {

        // create new network
        Console.Write("Creating network");
        Network network = new Network(2,3,2);
        network.setActivationFunction(activationFunctions.tanh);
        Console.WriteLine("....success");



        /////////////////////////////
        ////////TRAINING/////////////
        /////////////////////////////

        // load training data from file
        Console.Write("Loading training data");
        TrainingDataPoint[] trainingData = loadTrainingDataFromFile("./data/linear data/dataBIG0.csv", 2, 2);
        //trainingData = normalize(trainingData);

        Console.WriteLine("....success");

        int batchSize = 25;
        int epoch = 0;
        double learnRate = 1;
        double costSum = 0;
        double cost = 0;
        double previousCost = 1;

        // runs gradient descent algorithm until a key is pressed
        Console.WriteLine("Training network...");
        while(! Console.KeyAvailable || previousCost < 0.001)
        {
            int batchCount = 0;

            //trainingData = shuffle(trainingData);

            while (batchCount * batchSize < trainingData.Length)
            {
                TrainingDataPoint[] batch = new TrainingDataPoint[batchSize];
                Array.Copy(trainingData, batchCount * batchSize, batch, 0, batchSize);
                costSum += network.train(trainingData, learnRate);
                Console.WriteLine($"Epoch {epoch} Cost: " + (costSum / (batchSize * batchCount)));
                batchCount++;
            }

            cost = costSum / trainingData.Length;
            Console.WriteLine($"Epoch {epoch} Cost: " + cost);

            // If the cost this epoch is within 10% of the cost last epoch, change isn't happening and the learnRate should be reduced
            if (cost / previousCost < 1.1 && cost / previousCost > 0.9)
                learnRate *= 0.75;

            previousCost = cost;
            costSum = 0;
            epoch++;

        }



        /////////////////////////////
        ///////TESTING///////////////
        /////////////////////////////


        TrainingDataPoint[] testingData = loadTrainingDataFromFile("./data/linear data/dataBIG1.csv", 2, 2);

        double pointsCorrect = 0;
        double pointsCounted = 0;

        foreach (TrainingDataPoint point in testingData)
        {
            int output = network.process(point.inputs);
            int expectedOutput = 0;

            for (int i = 0; i < point.expectedOutputs.Length; i++)
            {
                if (point.expectedOutputs[i] > 0)
                {
                    expectedOutput = i;
                    break;
                }
            }

            if (expectedOutput == output)
                pointsCorrect++;

            pointsCounted++;

            Console.WriteLine($"Accuracy: {pointsCorrect / (double)pointsCounted}");
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
    public static TrainingDataPoint[] loadMNIST()
    {
        TrainingDataPoint[] trainingData;

        using (BinaryReader labelFile = new BinaryReader(new FileStream("./data/MNIST/train-labels.idx1-ubyte", FileMode.Open)))
        {
            using (BinaryReader imageFile = new BinaryReader(new FileStream("./data/MNIST/train-images.idx3-ubyte", FileMode.Open)))
            {
                int magic1 = endianSwap(imageFile.ReadInt32());     //headers seem to be stored in Big Endian?
                int imageCount = endianSwap(imageFile.ReadInt32());
                int rowCount = endianSwap(imageFile.ReadInt32());
                int colCount = endianSwap(imageFile.ReadInt32());

                int magic2 = endianSwap(labelFile.ReadInt32());
                int labelCount = endianSwap(labelFile.ReadInt32());

                trainingData = new TrainingDataPoint[imageCount];

                // for each image
                for (int i = 0; i < imageCount; i++)
                {
                    // store image & label data in arrays
                    double[] inputs = new double[rowCount * colCount];
                    double[] expectedOutputs = new double[10];

                    for (int pixel = 0; pixel < rowCount * colCount; pixel++)
                    {
                        inputs[pixel] = (double)(imageFile.ReadByte() / 255);   // convert to double, scale down from 0-255 to 0-1
                    }

                    // each byte in label file will be 0-9. Treat the byte as the index of expected output that should be activated.
                    // All indicies of expected output should be 0 except the correct answer to the training data point.
                    expectedOutputs[labelFile.ReadByte()] = 1.0;

                    trainingData[i] = new TrainingDataPoint(inputs,expectedOutputs);
                }


            }
        }

        return trainingData;
    }


    // Converts a little endian 4-byte int and converts it to big endian
    private static int endianSwap(int value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes);
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