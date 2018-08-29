package waddington.kai.main.knnf;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.exceptions.InvalidLayerOrderException;
import waddington.kai.main.knnf.exceptions.MissingActivationMethodException;
import waddington.kai.main.knnf.exceptions.UnknownActivationTypeException;
import waddington.kai.main.knnf.exceptions.UnknownPoolTypeException;
import waddington.kai.main.knnf.layers.neurons.Neuron;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Static utilities class that contains mostly mathematics operations.
 */
public class NetworkHelper {

    /**
     * Used to calculate the output shape/size from a convolution layer given its' input details.
     * @param numberOfFilters The number of filters in the convolution layer.
     * @param filterSize The size of the filters in the convolution layer.
     * @param stride The stride that the convolution layer uses.
     * @param inputSize The [x,y] size of the input to the convolution layer.
     * @return An int[] containing the output dimensions in the order: {z,y,x}
     */
    public static int[] calculateConvolutionOutputSize(int numberOfFilters, int filterSize, int stride, int[] inputSize) {
        int sizeZ = numberOfFilters;
        int sizeY = calculateConvolutionOutputSize(filterSize, stride, inputSize[1]);
        int sizeX = calculateConvolutionOutputSize(filterSize, stride, inputSize[2]);

        int[] out = new int[] {sizeZ, sizeY, sizeX};
        return out;
    }

    /**
     * Used to calculate the size of a single dimension of the output of a convolution layer.
     * @param filterSize The size of the filter used.
     * @param stride The stride used.
     * @param inputSize The input dimension along the axis that is to be calculated.
     * @return An int of the size of the output along the dimension that is to be calculated.
     */
    public static int calculateConvolutionOutputSize(int filterSize, int stride, int inputSize) {
        int out = ((inputSize - filterSize) / stride) + 1;
        return out;
    }

    /**
     * Used to calculate the output shape/size from a pooling layer.
     * @param poolSize The size of the receptive field of the pooling layer.
     * @param stride The stride used.
     * @param inputSize The [x,y] size of the input to the pool layer.
     * @return An int[] containing the output dimensions in the order: {z,y,x}
     */
    public static int[] calculatePoolOutputSize(int poolSize, int stride, int[] inputSize) {
        int sizeZ = inputSize[0];
        int sizeY = calculatePoolOutputSize(poolSize, stride, inputSize[1]);
        int sizeX = calculatePoolOutputSize(poolSize, stride, inputSize[2]);

        return new int[] {sizeZ, sizeY, sizeX};
    }

	/**
	 * Used to calculate the size of a single dimension of the output of a pooling layer.
	 * @param poolSize The size of the receptive field of the pooling layer.
	 * @param stride The stride used.
	 * @param inputSize The input dimension along the axis that is to be calculated.
	 * @return An int of the size of the output along the dimension that is to be calculated.
	 */
    public static int calculatePoolOutputSize(int poolSize, int stride, int inputSize) {

        int out = ((inputSize - poolSize) / stride) + 1;
        return out;
    }

	/**
	 * Calculates the output size (therefore number of neurons) of a flattening layer.
	 * @param inputSize The size/shape of the input to the flattening layer.
	 * @return An int[] containing the output dimensions in the order: {z,y,x}
	 */
	public static int[] caculateFlattenLayerOutputSize(int[] inputSize) {
        int size = inputSize[0] * inputSize[1] * inputSize[2]; 

        return new int[] {1, 1, size};
    }

    /**
     * Takes a string and matches it to a {@link LayerActivation} activation type.
     * @param activation The string that the user specifies as the activation type.
     * @return A {@link LayerActivation} type that matches the specified string.
     * @throws UnknownActivationTypeException if passed String cannot be matched to a LayerActivation.
     */
    public static LayerActivation matchActivation(String activation) {
        for (LayerActivation activation2: LayerActivation.values()) {
            if (activation.equals(activation2.toString()))
                return activation2;
        }

        throw new UnknownActivationTypeException(("\r\nUnknown activation: \"" + activation + "\"\r\n"));
    }

    /**
     * Takes a string and matches it to a {@link LayerType} pool type.
     * @param poolType The string that the user specifies as the pool type.
     * @return A {@link LayerType} type that matches the specified string.
     * @throws UnknownPoolTypeException if passed String cannot be matched to a LayerType.
     */
    public static LayerType matchPoolType(String poolType) {
        LayerType type;

        switch (poolType) {
            case "max": {
                type = LayerType.maxPool;
                break;
            }
            case "min": {
                type = LayerType.minPool;
                break;
            }
            default: {
                throw new UnknownPoolTypeException(("\r\nUnknown pool type: \"" + poolType + "\"\r\n"));
            }
        }

        return type;
    }

	/**
	 * Uses {@link LayerOrder} instances to set up a data structure used to check that layers are in a valid order.
	 * @return A Map between all {@link LayerType}'s and their {@link LayerOrder} instances.
	 */
	public static Map<LayerType, LayerOrder> createLayerOrderMap() {
        Map<LayerType, LayerOrder> layerOrderMap = new HashMap<>();

        if (layerOrderMap.get(LayerType.input) == null) {
			LayerOrder layerOrder = new LayerOrder(LayerType.input);
			layerOrder.addConsequents(LayerType.conv, LayerType.maxPool, LayerType.minPool, LayerType.flatten);
			layerOrderMap.put(LayerType.input, layerOrder);
		}
		if (layerOrderMap.get(LayerType.conv) == null) {
			LayerOrder layerOrder = new LayerOrder(LayerType.conv);
			layerOrder.addAntecedents(LayerType.input, LayerType.maxPool, LayerType.minPool, LayerType.conv);
			layerOrder.addConsequents(LayerType.maxPool, LayerType.minPool, LayerType.conv, LayerType.flatten);
			layerOrderMap.put(LayerType.conv, layerOrder);
		}
		if (layerOrderMap.get(LayerType.maxPool) == null) {
			LayerOrder layerOrder = new LayerOrder(LayerType.maxPool);
			layerOrder.addAntecedents(LayerType.input, LayerType.maxPool, LayerType.minPool, LayerType.conv);
			layerOrder.addConsequents(LayerType.maxPool, LayerType.minPool, LayerType.conv, LayerType.flatten);
			layerOrderMap.put(LayerType.maxPool, layerOrder);
		}
		if (layerOrderMap.get(LayerType.minPool) == null) {
			LayerOrder layerOrder = new LayerOrder(LayerType.minPool);
			layerOrder.addAntecedents(LayerType.input, LayerType.maxPool, LayerType.minPool, LayerType.conv);
			layerOrder.addConsequents(LayerType.maxPool, LayerType.minPool, LayerType.conv, LayerType.flatten);
			layerOrderMap.put(LayerType.minPool, layerOrder);
		}
		if (layerOrderMap.get(LayerType.flatten) == null) {
			LayerOrder layerOrder = new LayerOrder(LayerType.flatten);
			layerOrder.addAntecedents(LayerType.input, LayerType.conv, LayerType.maxPool, LayerType.minPool);
			layerOrder.addConsequents(LayerType.fc, LayerType.output);
			layerOrderMap.put(LayerType.flatten, layerOrder);
		}
		if (layerOrderMap.get(LayerType.fc) == null) {
			LayerOrder layerOrder = new LayerOrder(LayerType.fc);
			layerOrder.addAntecedents(LayerType.fc, LayerType.flatten);
			layerOrder.addConsequents(LayerType.fc, LayerType.output);
			layerOrderMap.put(LayerType.fc, layerOrder);
		}
		if (layerOrderMap.get(LayerType.output) == null) {
			LayerOrder layerOrder = new LayerOrder(LayerType.output);
			layerOrder.addAntecedents(LayerType.flatten, LayerType.fc);
			layerOrderMap.put(LayerType.output, layerOrder);
        }
        
        return layerOrderMap;
    }

    /**
     * Performs an element-wise addition of a list of matrices.
     * @param matrices A List containing the matrices to be summed.
     * @return A {@link DoubleMatrix} of the sum of the provided list of matrices.
     */
    public static DoubleMatrix sumMatrices(List<DoubleMatrix> matrices) {
        int rows = matrices.get(0).rows;
        int cols = matrices.get(0).columns;
        DoubleMatrix sum = new DoubleMatrix(rows, cols);

        for (int y=0; y<rows; y++) {
            for (int x=0; x<cols; x++) {
                sum.put(y, x, 0.0);
            }
        }

        for (DoubleMatrix matrix : matrices) {
            sum.addi(matrix);
        }

        return sum;
    }

    /**
     * Performs a summation of all elements in a matrix.
     * @param matrix The {@link DoubleMatrix} to be summed.
     * @return A double of the summed value.
     */
    public static double sumMatrix(DoubleMatrix matrix) {
        int rows = matrix.rows;
        int columns = matrix.columns;

        double sum = 0;

        for (int y=0; y<rows; y++) {
            for (int x=0; x<columns; x++) {
                sum += matrix.get(y, x);
            }
        }

        return sum;
    }

    /**
     * Creates a matrix from a portion of a provided matrix.
     * <p>
     * Used by convolution and pooling layers to get their receptive fields during their operations.
     * @param input The original {@link DoubleMatrix}.
     * @param startY The Y position that the sub-matrix should start from.
     * @param startX The X position that the sub-matrix should start from.
     * @param sizeY The Y size of the sub-matrix.
     * @param sizeX The X size of the sub-matrix.
     * @param stride The stride used by the calling layer type.
     * @return A {@link DoubleMatrix} containing the sub-matrix requested.
     */
    public static DoubleMatrix createSubMatrix(DoubleMatrix input, int startY, int startX, int sizeY, int sizeX, int stride) {
        DoubleMatrix output = new DoubleMatrix(sizeY, sizeX);

        for (int y=0; y<sizeY; y++) {
            for (int x=0; x<sizeX; x++) {
                double value = input.get((startY*stride)+y, (startX*stride)+x);
                output.put(y, x, value);
            }
        }

        return output;
    }

    /**
     * Applies a specified {@link LayerActivation} type to a matrix ({@link DoubleMatrix}) in an element-wise manner.
     * @param activation The {@link LayerActivation} activation type.
     * @param matrix The {@link DoubleMatrix} to apply the activation type to.
     * @return A {@link DoubleMatrix} with the activation type applied in an element-wise manner.
     */
    public static DoubleMatrix applyActivation(LayerActivation activation, DoubleMatrix matrix) {
        int rows = matrix.rows;
        int cols = matrix.columns;

        for (int y=0; y<rows; y++) {
            for (int x=0; x<cols; x++) {
                double value = applyActivation(activation, matrix.get(y, x));
                matrix.put(y, x, value);
            }
        }

        return matrix;
    }

    /**
     * Applies an {@link LayerActivation} to a value.
     * @param activation The activation type.
     * @param value The value to apply the activation type to.
     * @return The value after going through the activation function.
     */
    public static double applyActivation(LayerActivation activation, double value) {
        switch (activation) {
            case linear: {
                return value;
            }
            case sigmoid: {
                return sigmoid(value);
            }
            case tanh: {
                return tanh(value);
            }
            case relu: {
                return relu(value, 0.0);
            }
            case lrelu: {
                return relu(value, 0.01);
            }
            default: {
                throw new MissingActivationMethodException(activation.toString());
            }
        }
    }

    /**
     * Applies the derivative of a specified {@link LayerActivation} type to a matrix ({@link DoubleMatrix}) in an element-wise manner.
     * @param activation The {@link LayerActivation} activation type.
     * @param matrix The {@link DoubleMatrix} to apply the activation type derivative to.
     * @return A {@link DoubleMatrix} with the activation type derivative applied in an element-wise manner.
     */
    public static DoubleMatrix applyActivationDerivative(LayerActivation activation, DoubleMatrix matrix) {
        int rows = matrix.rows;
        int cols = matrix.columns;

        for (int y=0; y<rows; y++) {
            for (int x=0; x<cols; x++) {
                double value = applyActivationDerivative(activation, matrix.get(y, x));
                matrix.put(y, x, value);
            }
        }

        return matrix;
    }

    /**
     * Applies an {@link LayerActivation} derivative to a value.
     * @param activation The activation type to derive.
     * @param value The value to apply the activation type derivative to.
     * @return The value after going through the activation derivative function.
     */
    public static double applyActivationDerivative(LayerActivation activation, double value) {
        switch (activation) {
            case linear: {
                return value;
            }
            case sigmoid: {
                return sigmoidDerivative(value);
            }
            case tanh: {
                return tanhDerivative(value);
            }
            case relu: {
                return relu(value, 0.0);
            }
            case lrelu: {
                return relu(value, 0.01);
            }
            default: {
                throw new MissingActivationMethodException(activation.toString());
            }
        }
    }

    /**
     * Applies the Sigmoid function to a value.
     * @param a The value to apply the sigmoid function to.
     * @return The value after the Sigmoid function has been applied.
     */
    public static double sigmoid(double a) {
        double sig = 1.0 / (1.0 + Math.exp(a * -1.0));

        // Ensure number is not max/min so help with continuing gradient descent
        if (sig == 1)
            sig -= 0.00000001;
        else if (sig == 0)
            sig += 0.00000001;

		return sig;
    }

    /**
     * The derivative of the sigmoid function.
     * @param a The value to apply the sigmoid derivative function to.
     * @return The value after the function has been applied.
     */
    public static double sigmoidDerivative(double a) {
        double value = sigmoid(a) * (1 - sigmoid(a));

        return value;
    }

    /**
     * The Tanh function.
     * @param a The value to apply the tanh function to.
     * @return The value after the function has been applied.
     */
    public static double tanh(double a) {
        double tanh = 2 * sigmoid(2 * a) - 1;

        // Ensure number is max/min so help with continuing gradient descent
        if (tanh == 1)
            tanh -= 0.00000001;
        else if (tanh == -1)
            tanh += 0.00000001;

		return tanh;
    }

    /**
     * The derivative of the tanh function.
     * @param a The value to apply the tanh derivative function to.
     * @return The value after the function has been applied.
     */
    public static double tanhDerivative(double a) {
        double value = 1 - Math.pow(tanh(a), 2);

        return value;
    }

    /**
     * The ReLu (and Leaky ReLu) function.
     * <p>
     * The ReLu function is the same as the Leaky ReLu function just with a leak rate of 0.
     * @param value The value to apply the ReLu function to.
     * @param leakRate The leak rate.
     * @return The new value.
     */
    public static double relu(double value, double leakRate) {
        if (value > 0)
            return value;

        if (leakRate == 0)
            return 0;

        return value * leakRate;
    }

    /**
     * Encodes a cells location in a matrix as a number.
     * <p>
     * The number of rows and columns is required because the X and Y cell locations are zero-padded to be the length of the number of digits in the number of rows/cols.
     * This allows for the splitting of the number in the correct location to separate the X and Y locations.
     * The resulting number begins and ends with a 1 to preserve the zeros from the padding.
     * @param y The Y location of the cell.
     * @param x The X location of the cell.
     * @param inputRows The number of rows in the matrix that the cell is in.
     * @param inputCols The number of columns in the matrix that the cell is in.
     * @return An encoded version of the cells location.
     */
    public static int poolEncodeLocation(int y, int x, int inputRows, int inputCols) {
        StringBuilder sb = new StringBuilder();

        // Leading and trailing 1's to keep any 0's
        sb.append(1);

        sb.append(String.format("%0" + String.valueOf(inputRows).length() + "d", (y)));
        sb.append(String.format("%0" + String.valueOf(inputCols).length() + "d", (x)));

        sb.append(1);

        return Integer.valueOf(sb.toString());
    }

    /**
     * Decodes an encoded cell location.
     * @param encodedValue The encoded cell location generated from the {@link waddington.kai.main.knnf.NetworkHelper#poolEncodeLocation(int, int, int, int)} method.
     * @param inputRows The number of rows in the matrix that the cell belongs to.
     * @return The {Y, X} locations that the cell references.
     */
    public static int[] poolDecodeLocation(int encodedValue, int inputRows) {
        String encodedVal = Integer.toString(encodedValue);
        int ySize = String.valueOf(inputRows).length();

        // Trim the leading and trailing 1's
        encodedVal = encodedVal.substring(1, encodedVal.length()-1);

        String decodedY = encodedVal.substring(0, ySize);
        String decodedX = encodedVal.substring(ySize, encodedVal.length());

        return new int[] {Integer.valueOf(decodedY), Integer.valueOf(decodedX)};
    }

    /**
     * Joins the data from a List of Neurons into a single flat list.
     * @param inputNeurons The List of Neurons that data should be joined from.
     * @return A {@link DoubleMatrix} containing the data from all of the neurons.
     */
    public static DoubleMatrix joinInputVectors(List<Neuron> inputNeurons) {
        // Get the first neurons' data
        DoubleMatrix output = inputNeurons.get(0).getOutputData();

        // Start at 1 because we already have the first neurons data
        for (int i=1; i<inputNeurons.size(); i++) {
            Neuron neuron = inputNeurons.get(i);
            DoubleMatrix data = neuron.getOutputData();

            output = DoubleMatrix.concatHorizontally(output, data);
        }

        return output;
    }

    /**
     * Applies the softmax function to a set of numbers.
     * @param inputs A flat {@link DoubleMatrix} that contains the values to apply the softmax function to.
     * @return A new {@link DoubleMatrix} containing the values after applying the softmax function.
     */
    public static DoubleMatrix applySoftmax(DoubleMatrix inputs) {
        DoubleMatrix softmax = new DoubleMatrix(1, inputs.columns);
        DoubleMatrix expInputs = new DoubleMatrix(1, inputs.columns);
        double sumExp = 0;

        for (int i=0; i<inputs.columns; i++) {
            expInputs.put(i, Math.exp(inputs.get(0, i)));
            sumExp += expInputs.get(i);
        }

        for (int i=0; i<softmax.columns; i++) {
            softmax.put(i, expInputs.get(i) / sumExp);
        }

        return softmax;
    }

    /**
     * Gets a {@link DoubleMatrix} containing the backpropagated error from a list of neurons.
     * <p>
     * Works if the type of the layer above is flatten/conv/pool.
     * @param neuron The neuron that is calling this method.
     * @param lowerNeurons A list of neurons in the layer below.
     * @param upperNeurons A list of neurons in the layer above.
     * @return A {@link DoubleMatrix} containing the backpropagated error from the layer above.
     */
    public static DoubleMatrix getUpperError(Neuron neuron, List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        LayerType upperType = upperNeurons.get(0).getType();

        // Get error from upper layers
        switch (upperType) {
            case flatten: {
                return NetworkHelper.getErrorFromFlatten(upperNeurons, neuron.getId(), neuron.getOutputData().rows, neuron.getOutputData().columns);
            }
            case conv: {
                return NetworkHelper.getErrorFromConv(neuron, lowerNeurons, upperNeurons);
            }
            case minPool: {
                return NetworkHelper.getErrorFromPool(upperNeurons, neuron.getId());
            }
            case maxPool: {
                return NetworkHelper.getErrorFromPool(upperNeurons, neuron.getId());
            }
            default: {
                throw new InvalidLayerOrderException("No method found for backpropagation to this convolution layer.");
            }
        }
    }

    /**
     * Gets the backpropagated error from a flatten layer.
     * @param upperNeurons The list of neurons in the layer above.
     * @param id The ID of the current neuron.
     * @param rows The rows in the input data. Used to shape the output of this function.
     * @param cols The columns in the input data. Used to shape the output of this function.
     * @return A {@link DoubleMatrix} containing the backpropagated error from a flattening layer.
     */
    public static DoubleMatrix getErrorFromFlatten(List<Neuron> upperNeurons, int id, int rows, int cols) {
        // Create the matrix
        DoubleMatrix errors = new DoubleMatrix(rows, cols);

        // Translate neuron ID to range of neurons from upper layer
        // and add the errors to a matrix
        int startingIndex = id * errors.length;
        for (int i=0; i<errors.length; i++) {
            int index = startingIndex + i;
            double data = upperNeurons.get(index).getError().get(0,0);
            errors.put(i, data);
        }

        return errors;
    }

    /**
     * Gets the backpropagated error from a convolution layer.
     * @param neuron The neuron calling this method.
     * @param lowerNeurons A list containing references to the neurons in the layer below.
     * @param upperNeurons A list containing references to the neurons in the layer above.
     * @return A {@link DoubleMatrix} containing the backpropagated error from a convolution layer.
     */
    public static DoubleMatrix getErrorFromConv(Neuron neuron, List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        // Sum all upper errors
        DoubleMatrix sumUpperError = upperNeurons.get(0).getError();
        for (int i=1; i<upperNeurons.size(); i++) {
            sumUpperError.addi(upperNeurons.get(i).getError());
        }

        // Each conv. neuron holds a copy of merged input data
        DoubleMatrix sumOfInputLayer = upperNeurons.get(0).getInputData();

        // Calculate percentage that this neuron is of all input neurons
        DoubleMatrix inputEffect = neuron.getOutputData().div(sumOfInputLayer);

        // Get proportion of upper error
        DoubleMatrix proportionalisedUpperError = sumUpperError.mul(inputEffect);

        return proportionalisedUpperError;
    }

    /**
     * Gets the backpropagated error from a pooling layer.
     * @param upperNeurons A list containing references to the neurons in the layer above.
     * @param id The ID of the neuron calling this function.
     * @return A {@link DoubleMatrix} containing the backpropagated error from a pooling layer.
     */
    public static DoubleMatrix getErrorFromPool(List<Neuron> upperNeurons, int id) {
        return upperNeurons.get(id).getError();
    }
}