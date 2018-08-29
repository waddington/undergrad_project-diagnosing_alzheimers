package waddington.kai.main.knnf.layers.neurons;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;

import java.util.List;

/**
 * This is the abstract class that all neuron-type-specific classes extend.
 */
public abstract class Neuron {
	/**
	 * Each neuron has an ID and is it's position in the network.
	 */
    private final int id;
	/**
	 * The specific type that the neuron is.
	 */
    private final LayerType type;
	/**
	 * The activation type that the neuron uses. Sometimes null.
	 */
    private final LayerActivation activation;

	/**
	 * The input data to the neuron.
	 */
	private DoubleMatrix inputData;
	/**
	 * The output data from the neuron, the neuron calculates this.
	 */
    private DoubleMatrix outputData;

	/**
	 * The error of the network that this neuron is responsible for.
	 */
	private DoubleMatrix error;
	/**
	 * The weight delta(s) for this neuron.
	 */
    private DoubleMatrix deltas;

	/**
	 * This is the only constructor that can be used. Sets the ID, layer type, and activation type of the neuron.
	 * @param id The ID for the neuron.
	 * @param type The {@link LayerType} that the new neuron is.
	 * @param activation The {@link LayerActivation} that the neuron will use.
	 */
    public Neuron(int id, LayerType type, LayerActivation activation) {
        this.id = id;
        this.type = type;
        this.activation = activation;
    }

	/**
	 * Get the ID of the neuron.
	 * @return The ID of the neuron.
	 */
	public int getId() {
        return id;
    }

	/**
	 * Get the type of neuron.
	 * @return The type of the neuron.
	 */
	public LayerType getType() {
        return type;
    }

	/**
	 * Get the activation type that the neuron uses.
	 * @return The activation type that the neuron uses.
	 */
    public LayerActivation getActivation() {
        return activation;
    }

	/**
	 * Sets the input data for the neuron.
	 * @param data A {@link DoubleMatrix} containing the input data.
	 */
	public void setInputData(DoubleMatrix data) {
        inputData = data;
        // Default the output data to the input data
        setOutputData(data);
    }

	/**
	 * Get the neurons input data.
	 * @return A {@link DoubleMatrix} containing the neurons input data.
	 */
	public DoubleMatrix getInputData() {
        return inputData;
    }

	/**
	 * Sets the output data that the neuron calculates.
	 * @param data A {@link DoubleMatrix} containing the calculated data.
	 */
    public void setOutputData(DoubleMatrix data) {
        outputData = data;
    }

	/**
	 * Gets the neurons output data.
	 * @return A {@link DoubleMatrix} containing the neurons output data.
	 */
	public DoubleMatrix getOutputData() {
        return outputData;
    }

	/**
	 * Sets the error for the neuron.
	 * @param error The error to set.
	 */
	public void setError(DoubleMatrix error) {
        this.error = error;
    }

	/**
	 * Gets the neurons error.
	 * @return The error of the neuron.
	 */
	public DoubleMatrix getError() {
        return error;
    }

	/**
	 * Sets the weight delatas for the neuron.
	 * @param deltas A {@link DoubleMatrix} containing the weight deltas.
	 */
	public void setDeltas(DoubleMatrix deltas) {
        this.deltas = deltas;
    }

	/**
	 * Gets the neurons weight deltas.
	 * @return A {@link DoubleMatrix} containing the weight deltas.
	 */
	public DoubleMatrix getDeltas() {
        return deltas;
    }

	/**
	 * Creates empty {@link DoubleMatrix} instances for the input and output data.
	 * @param inputSize The shape of the input data.
	 * @param outputSize The shape of the output data.
	 */
	public void initMemory(int[] inputSize, int[] outputSize) {
        initInputMemory(inputSize);
        initOutputMemory(outputSize);
    }

	/**
	 * Creates empty {@link DoubleMatrix} for the input data.
	 * @param inputSize The shape of the input data.
	 */
	private void initInputMemory(int[] inputSize) {
        inputData = new DoubleMatrix(inputSize[1], inputSize[2]);
    }

	/**
	 * Creates empty {@link DoubleMatrix} for the output data.
	 * @param outputSize The shape of the input data.
	 */
    private void initOutputMemory(int[] outputSize) {
        outputData = new DoubleMatrix(outputSize[1], outputSize[2]);
    }

	/**
	 * Method used to calculate the weight deltas for the neuron.
	 * @param lowerNeurons A list of neurons in the layer below.
	 * @param upperNeurons A list of neurons in the layer above.
	 */
	public abstract void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons);

	/**
	 * Applies the weight deltas.
	 */
    public abstract void applyDeltas();
}