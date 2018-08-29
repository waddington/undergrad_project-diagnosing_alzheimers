package waddington.kai.main.knnf.layers;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.exceptions.IllegalMethodCallException;
import waddington.kai.main.knnf.layers.neurons.Neuron;
import waddington.kai.main.knnf.layers.neurons.NeuronFactory;

import java.util.List;

/**
 * This is the class for the input layer. It extends the {@link Layer} class.
 */
public class LayerInput extends Layer {

    /**
     * See {@link Layer#Layer(int, LayerType, LayerActivation)}.
     * @param id The layer ID.
     */
    public LayerInput(int id) {
        super(id, LayerType.input, null);
    }

	/**
	 * Creates the neurons for this layer.
	 * @param number The number of neurons to create, also the number of channels in the input images.
	 */
    @Override
    public void createNeurons(int number) {
        // Neuron for each image channel
        for (int i=0; i<getInputSize()[0]; i++) {
            addNeuron(NeuronFactory.getInputNeuron(i, getInputSize(), getOutputSize()));
        }
    }

	/**
	 * Required implementation of abstract method - not used. See {@link #setInput(List)} for what should be used instead.
	 * @param inputNeurons The list of neurons to get the data from.
	 * @throws IllegalMethodCallException If this method is called as the input data is handled separately.
	 */
	@Override
    public void setInputData(List<Neuron> inputNeurons) {
        throw new IllegalMethodCallException("setInputData(List<Neuron> inputNeurons) should not be called inside LayerInput.");
    }

	/**
	 * Sets the input data for neurons in this layer.
	 * @param inputData The input data.
	 */
	public void setInput(List<DoubleMatrix> inputData) {
        for (int i=0; i<getNumberOfNeurons(); i++) {
            Neuron neuron = getNeuron(i);
            DoubleMatrix data = inputData.get(i);

            neuron.setInputData(data);
        }
    }

	/**
	 * Doesn't do anything in input layer...
	 */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        // Input layer doesn't need to do anything
    }

	/**
	 * Doesn't do anything in input layer...
	 */
    @Override
    public void applyDeltas() {
        // Input layer doesn't need to do anything
    }
}