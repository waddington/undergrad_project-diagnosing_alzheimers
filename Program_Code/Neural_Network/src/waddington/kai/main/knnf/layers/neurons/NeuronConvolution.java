package waddington.kai.main.knnf.layers.neurons;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkHelper;
import waddington.kai.main.knnf.NetworkManager;

import java.util.List;

/**
 * This is the class for convolution neurons. It extends the {@link Neuron} class.
 */
public class NeuronConvolution extends Neuron {
    /**
     * The size of the receptive field.
     */
    private int filterSize;
    /**
     * The stride used.
     */
    private int stride;
    /**
     * A {@link DoubleMatrix} containing the values for the filter.
     */
    private DoubleMatrix filter;
    /**
     * A {@link DoubleMatrix} that will store the delta values for the filter.
     */
    private DoubleMatrix filterDelta;
    /**
     * The bias value.
     */
    private double bias;
    /**
     * The delta value for the bias.
     */
    private double biasDelta;

    /**
     * See {@link Neuron#Neuron(int, LayerType, LayerActivation)}.
     * @param id The neuron ID.
     * @param filterSize The size of the receptive field.
     * @param stride The stride used.
     * @param activation The {@link LayerActivation} for this neuron.
     */
    public NeuronConvolution(int id, int filterSize, int stride, LayerActivation activation) {
        super(id, LayerType.conv, activation);

        this.filterSize = filterSize;
        this.stride = stride;
    }

    /**
     * Initialises the values for the filter. The min/max values are dictated by {@link NetworkManager#MaximumInitialWeights} and are generated used {@link NetworkManager#random}.
     */
    public void initFilter() {
        filter = new DoubleMatrix(filterSize, filterSize);
        filterDelta = new DoubleMatrix(filterSize, filterSize);

        for (int y=0; y<filterSize; y++) {
            for (int x=0; x<filterSize; x++) {
                double value = NetworkManager.random.nextDouble() * NetworkManager.MaximumInitialWeights;
                filter.put(y, x, value);
            }
        }

        bias = NetworkManager.random.nextDouble() * NetworkManager.MaximumInitialWeights;
    }

    /**
     * Sets the size of the filter.
     * @param size The size of the filter.
     */
    public void setFilterSize(int size) {
        filterSize = size;
    }

    /**
     * Gets the size of the filter.
     * @return The size of the filter.
     */
    public int getFilterSize() {
        return filterSize;
    }

    /**
     * Sets the stride to use.
     * @param size The stride to use.
     */
    public void setStride(int size) {
        stride = size;
    }

    /**
     * Gets the stride used.
     * @return The stride used.
     */
    public int getStride() {
        return stride;
    }

    /**
     * Sets the filter for the neuron.
     * @param filter A correctly shaped {@link DoubleMatrix} containing the filter values.
     */
    public void setFilter(DoubleMatrix filter) {
        this.filter = filter;
    }

    /**
     * Gets the filter of the neuron.
     * @return The filter of the neuron.
     */
    public DoubleMatrix getFilter() {
        return filter;
    }

    /**
     * Sets the neurons bias value.
     * @param b The bias value.
     */
    public void setBias(double b) {
        bias = b;
    }

    /**
     * Gets the neurons bias value.
     * @return The neurons bias value.
     */
    public double getBias() {
        return bias;
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
     * Calculates the neurons output using the convolution process.
     */
    private void calculateOutput() {
        // Preliminary data
        DoubleMatrix inputData = getInputData();
        int inputRows = inputData.rows;
        int inputCols = inputData.columns;
        int filterSize = filter.rows;
        int[] outputSize = NetworkHelper.calculateConvolutionOutputSize(1, filterSize, stride, new int[] {1, inputRows, inputCols});
        int outputRows = outputSize[1];
        int outputCols = outputSize[2];
        DoubleMatrix output = new DoubleMatrix(outputRows, outputCols);
        double sumOfFilter = NetworkHelper.sumMatrix(filter);

        // Do the convolution
        for (int oy=0; oy<outputRows; oy++) {
            for (int ox=0; ox<outputCols; ox++) {
                // Create a matrix of the part of the input that the filter should be applied to
                DoubleMatrix subView = NetworkHelper.createSubMatrix(inputData, oy, ox, filterSize, filterSize, stride);
                DoubleMatrix filterApplied = filter.mul(subView);
                double value = (NetworkHelper.sumMatrix(filterApplied) / sumOfFilter) + bias;

                value = NetworkHelper.applyActivation(getActivation(), value);

                output.put(oy, ox, value);
            }
        }

        super.setOutputData(output);
    }

    /**
     * Calculates the bias delta and triggers the calculation of the filter deltas.
     * @param lowerNeurons A list of neurons in the layer below.
     * @param upperNeurons A list of neurons in the layer above.
     */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        DoubleMatrix upperError = NetworkHelper.getUpperError(this, lowerNeurons, upperNeurons);
        upperError = NetworkHelper.applyActivationDerivative(getActivation(), upperError);

        calculateFilterDeltas(upperNeurons, upperError);
        calculateInputError(upperNeurons, upperError);

        double biasMomentum = NetworkManager.Momentum * biasDelta;
        biasDelta = NetworkManager.LearningRate * (1 / NetworkHelper.sumMatrix(upperError)) + biasMomentum;
    }

    /***
     * Calculates the deltas for the filter.
     * @param upperNeurons A list of neurons in the layer above.
     * @param upperError A {@link DoubleMatrix} containing the errors from the layer above.
     */
    private void calculateFilterDeltas(List<Neuron> upperNeurons, DoubleMatrix upperError) {
        // Ensure deltas matrix exists
        if (filterDelta == null) {
            filterDelta = new DoubleMatrix(filterSize, filterSize);
        }
        
        DoubleMatrix filterMomentum = filterDelta.mul(NetworkManager.Momentum);
        DoubleMatrix errorDivideByOutput = upperError.div(getOutputData());

        // Calculate the delta
        // For each cell in the filter do:
        for (int y=0; y<filterSize; y++) {
            for (int x=0; x<filterSize; x++) {
                // Sum for all:
                // All output values divide by this filter value
                // Multiplied by corresponding upper errors divide by all outputs
                DoubleMatrix outputDivideByFilterCell = getOutputData().div(filter.get(y,x));
                DoubleMatrix filterBetaMatrix = errorDivideByOutput.mul(outputDivideByFilterCell);
                filterDelta.put(y, x, NetworkHelper.sumMatrix(filterBetaMatrix));
            }
        }

        filterDelta.addi(filterMomentum);
    }

    /**
     * Back-propagates the error through this neuron, calculating how much each input to this neuron is responsible for this neurons error.
     * @param upperNeurons A list of neurons in the layer above.
     * @param upperError A {@link DoubleMatrix} containing the errors from the layer above.
     */
    private void calculateInputError(List<Neuron> upperNeurons, DoubleMatrix upperError) {
        DoubleMatrix errorDivideByOutput = upperError.div(getOutputData());
        DoubleMatrix inputError = new DoubleMatrix(getInputData().rows, getInputData().columns);

        // For each input cell
        for (int y=0; y<getInputData().rows; y++) {
            for (int x=0; x<getInputData().columns; x++) {
                int[] effectedBounds = getEffectedBounds(y, x);
                DoubleMatrix effectedOutput = NetworkHelper.createSubMatrix(getOutputData(), effectedBounds[0], effectedBounds[1], effectedBounds[2], effectedBounds[3], 1);
                DoubleMatrix outputDividedByInput = effectedOutput.div(getInputData().get(y, x));
                double sumInputEffect = NetworkHelper.sumMatrix(outputDividedByInput);

                inputError.put(y, x, sumInputEffect);
            }
        }

        setError(inputError);
    }

    /**
     * Calculates the coords. of the receptive field.
     * @param y The Y coord. of the output position to calculate.
     * @param x The X coord. of the output position to calculate.
     * @return The bounds of the receptive field in order {upper bound, left bound, Y size, X size}.
     */
    private int[] getEffectedBounds(int y, int x) {
        int rows = getOutputData().rows;
        int cols = getOutputData().columns;

        int upper = y - (filterSize - 1);
        if (upper < 0) upper = 0;

        int lower = y + (filterSize - 1);
        if (lower > rows) lower = rows;

        int left = x - (filterSize - 1);
        if (left < 0) left = 0;

        int right = x + (filterSize - 1);
        if (right > cols) right = cols;

        int rangeY = lower - upper;
        int rangeX = right - left;

        return new int[] {upper, left, rangeY, rangeX};
    }

    /**
     * Applies the filter and bias deltas.
     */
    @Override
    public void applyDeltas() {
        bias += biasDelta;
        filter.addi(filterDelta);
    }
}