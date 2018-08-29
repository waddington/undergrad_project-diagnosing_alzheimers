package waddington.kai.main.knnf.layers;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkHelper;
import waddington.kai.main.knnf.layers.neurons.Neuron;
import waddington.kai.main.knnf.layers.neurons.NeuronFactory;
import waddington.kai.main.knnf.layers.neurons.NeuronOutput;

import java.util.List;

/**
 * This is the class for the output layer. It extends the {@link Layer} class.
 */
public class LayerOutput extends Layer {
	/**
	 * The number of classes for the data, and the number of neurons in the layer.
	 */
    private int numberOfOutputs;
	/**
	 * The number of weights that each neuron will have (excluding weight to bias).
	 */
	private int numberOfWeights;

	/**
	 * See {@link Layer#Layer(int, LayerType, LayerActivation)}.
	 * @param id The layer ID.
	 */
    public LayerOutput(int id) {
        super(id, LayerType.output, null);
    }

    public void setNumberOfOutputs(int numberOfOutputs) {
        this.numberOfOutputs = numberOfOutputs;
    }

    public int getNumberOfOutputs() {
        return numberOfOutputs;
    }

    public void setNumberOfWeights(int number) {
        numberOfWeights = number;
    }

    public int getNumberOfWeights() {
        return numberOfWeights;
    }

	/**
	 * Creates the neurons for this layer.
	 * @param number The number of neurons to create.
	 */
    @Override
    public void createNeurons(int number) {
        for (int i=0; i<numberOfOutputs; i++) {
            Neuron neuron = NeuronFactory.getOutputNeuron(i, getLayerActivation(), numberOfWeights);
            addNeuron(neuron);
        }
    }

	/**
	 * Sets the input data to each neuron. Joins values from all inputs to a 1xN matrix.
	 * As setting the input data also triggers the neuron to calculate its' output, this method also retrieves the output from each neuron and applies the softmax function all outputs and then changes the outputs of each neuron from the original value to the respective value from the softmax function.
	 * @param inputNeurons The list of neurons to get the data from.
	 */
    @Override
    public void setInputData(List<Neuron> inputNeurons) {
        // Merge input data
        DoubleMatrix joinedInputs = NetworkHelper.joinInputVectors(inputNeurons);
        DoubleMatrix outputs = new DoubleMatrix(1, numberOfOutputs);

        // Set the input data for each neuron
        // and get their outputs so can apply softmax
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).setInputData(joinedInputs);
            outputs.put(0, i, getNeuron(i).getOutputData().get(0,0));
        }

        // Apply softmax function
        DoubleMatrix softmaxOutputs = NetworkHelper.applySoftmax(outputs);

        // Apply softmax outputs back to each neuron
        for (int i=0; i<softmaxOutputs.columns; i++) {
            double value = softmaxOutputs.get(i);
            getNeuron(i).setOutputData(new DoubleMatrix(new double[] {value}));
        }
    }

	/**
	 * Calculates and retrieves the MSE of the network.
	 * @param label The expected classification, so that the error can be calculated.
	 * @return The MSE of the network.
	 */
	public double getError(int label) {
        double sumSquaredError = 0;

        for (int i=0; i<getNumberOfNeurons(); i++) {
            int expectedOutput = 0;

            if (i == label)
                expectedOutput = 1;

            sumSquaredError += ((NeuronOutput) getNeuron(i)).setExpectedOutput(expectedOutput);
        }

        return sumSquaredError / numberOfOutputs;
    }

	/**
	 *  Triggers the weight delta calculations for each neuron in this layer.
	 * @param lowerNeurons The neurons in the layer below.
	 * @param upperNeurons The neurons in the layer above.
	 */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).calculateDeltas(lowerNeurons, upperNeurons);
        }
    }

	/**
	 * Triggers each neuron in this layer to apply their weight deltas.
	 */
    @Override
    public void applyDeltas() {
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).applyDeltas();
        }
    }

	/**
	 * Retrieves the outputs from each of the neurons in the layer.
	 * @return A 1xN matrix of the outputs from each neuron in the layer.
	 */
	public DoubleMatrix getOutputs() {
        double[] outputs = new double[getNumberOfOutputs()];

        for (int i=0; i<getNumberOfNeurons(); i++) {
            outputs[i] = getNeuron(i).getOutputData().get(0,0);
        }

        return new DoubleMatrix(outputs);
    }
}