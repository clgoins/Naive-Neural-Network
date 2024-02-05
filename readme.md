## Naive Neural Network

### What is it?
This project is a very naive implementation of a machine learning neural network.
It was built from scratch in C# to test my understanding of how the concepts work together
to make a program that could "learn" to approximate arbitrary functions, given a set of
inputs and expected outputs.


### How does it work?
This project is a work in progress and as such at the moment it's not flexible.
For the time being, the model configuration, training data, and testing data are hard coded
in `Program.cs`. In the future I intend for this program to take these parameters via command line or
GUI, and to give it the ability to export a models configuration to disk once it's been trained, such
that a previously trained model can be recalled as desired. There's still a lot of testing and
optimization to be done until then, since as of right now, while the model does work, the training process
is extremely slow.


### Why did I make it?
The main objective here was simply a better understanding of how neural networks function as a concept.
Potentially in the future I'd like to take what I've learned here and apply it in an audio processing
context. Ideally I'd like to be able to take a hardware guitar effects pedal that I've built, train
a model to approximate it, and integrate the model into a VST plugin, where hypothetically I can
create digital versions of hardware analog effects that I've built over the years.