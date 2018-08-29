package waddington.kai.main.knnf.layers.neurons;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkHelper;

import java.util.List;

/**
 * This is the class for output-layer neurons. It extends the {@link Neuron} class.
 */
public class NeuronPool extends Neuron {
	/**
	 * The size of the receptive field.
	 */
    private int poolSize;
	/**
	 * The stride value used.
	 */
	private int stride;
	/**
	 * A 2D array, the same size as the output, that maps the output cell locations to their respective locations in the input matrix.
	 */
    private int[][] poolLocations;

	/**
	 * See {@link Neuron#Neuron(int, LayerType, LayerActivation)}.
	 * @param id The neuron ID.
	 * @param type The type of pooling that this neuron will perform.
	 */
    public NeuronPool(int id, LayerType type) {
        super(id, type, null);
    }

	/**
	 * Sets the size of the receptive field.
	 * @param poolSize The size to use.
	 */
	public void setPoolSize(int poolSize) {
        this.poolSize = poolSize;
    }

    public int getPoolSize() {
        return poolSize;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public int getStride() {
        return stride;
    }

	/**
	 * Sets the input data for the neuron and triggers the neuron to calculate its output.
	 * @param data A {@link DoubleMatrix} containing the input data.
	 */
    @Override
    public void setInputData(DoubleMatrix data) {
        super.setInputData(data);

        calculateOutput();
    }

	/**
	 * Calculates the neurons output using the pooling process. Output data cells are mapped to a location in the input data using {@link #poolLocations}.
	 */
    private void calculateOutput() {
        // Preliminary data
        DoubleMatrix inputData = getInputData();
        int inputRows = inputData.rows;
        int inputCols = inputData.columns;
        int[] outputSize = NetworkHelper.calculatePoolOutputSize(poolSize, stride, new int[] {1, inputRows, inputCols});
        int outputRows = outputSize[1];
        int outputCols = outputSize[2];
        DoubleMatrix output = new DoubleMatrix(outputRows, outputCols);
        poolLocations = new int[outputRows][outputCols];
        LayerType type = getType();
        
        // Do the pooling
        for (int oy=0; oy<outputRows; oy++) {
            for (int ox=0; ox<outputCols; ox++) {
                // Create a matrix of the part of the input in the pooling view
                DoubleMatrix subView = NetworkHelper.createSubMatrix(inputData, oy, ox, poolSize, poolSize, stride);
                int index;
                double value;

                if (type == LayerType.maxPool) {
                    value = subView.max();
                    index = subView.argmax();
                } else {
                    value = subView.min();
                    index = subView.argmin();
                }

                // Translate linear index in subview you Y/X coordinates of entire input
                int indexY = (index % poolSize) + (oy * stride);
                int indexX = (index / poolSize) + (ox * stride);

                output.put(oy, ox, value);
                poolLocations[oy][ox] = NetworkHelper.poolEncodeLocation(indexY, indexX, inputRows, inputCols);

            }
        }

        super.setOutputData(output);
    }

	/**
	 * Back-propagates the error from the layer above for the layer below. Any cells in the input data that were not in the output data have an error of zero as they do not contribute.
	 * @param lowerNeurons A list of neurons in the layer below.
	 * @param upperNeurons A list of neurons in the layer above.
	 */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        // No deltas to calculate, just which errors to pass back where
        DoubleMatrix errorFromUp = NetworkHelper.getUpperError(this, lowerNeurons, upperNeurons);

        // Create matrix of errors to pass back 
        DoubleMatrix errors = DoubleMatrix.zeros(getInputData().rows, getInputData().columns);

        for (int y=0; y<errorFromUp.rows; y++) {
            for (int x=0; x<errorFromUp.columns; x++) {
                double error = errorFromUp.get(y, x);
                int encodedLocation = poolLocations[y][x];
                int[] decodedLocation = NetworkHelper.poolDecodeLocation(encodedLocation, getInputData().rows);

                errors.put(decodedLocation[0], decodedLocation[1], error);
            }
        }

        setError(errors);
    }

	/**
	 * The pooling-layer neurons don't do anything here.
	 */
	@Override
    public void applyDeltas() {}
}