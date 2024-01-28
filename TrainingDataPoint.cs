public class TrainingDataPoint
{
    public double[] inputs;
    public double[] expectedOutputs;

    public TrainingDataPoint(double[] inputs, double[] expectedOutputs)
    {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }

}