package waddington.kai.main.knnf.layers;

import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkHelper;
import waddington.kai.main.knnf.NetworkManager;

import java.util.Arrays;

/**
 * Static factory class for creating new {@link Layer} instances.
 */
public class LayerFactory {
    /**
     * The current number of layers created, used to generate the layer ID's.
     */
    public static int numberOfLayers = 0;

    /**
     * Creates and sets up an instance of {@link LayerInput}.
     * @param imgWidth The width of the input images.
     * @param imgHeight The height of the input images.
     * @param imgChannels The number of channels in the input images, used for the number of neurons in the layer.
     * @return An instance of {@link LayerInput}.
     */
    public static Layer getInputLayer(int imgWidth, int imgHeight, int imgChannels) {
        int layerId = numberOfLayers++;
        int[] inputSize = new int[] {imgChannels, imgHeight, imgWidth}; // {z,y,x}
        int[] outputSize = inputSize;

        LayerInput layer = new LayerInput(layerId);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);

        layer.createNeurons(1);

        StringBuilder sb = new StringBuilder();
        sb.append("Layer(" + layerId + ")(input), ");
        sb.append("input-size("+ Arrays.toString(inputSize) +"), ");
        sb.append("output-size("+Arrays.toString(outputSize)+").");
        System.out.println(sb);

        return layer;
    }

    /**
     * Creates and sets up an instance of {@link LayerConvolution}.
     * @param numFilters The number of neurons for this layer.
     * @param filterSize The size of the receptive field.
     * @param stride The stride that this layer will use.
     * @param activation The {@link LayerActivation} type that this layer will use.
     * @param numberOfNeuronsToLayer The number of inputs channels to the layer.
     * @return An instance of {@link LayerConvolution}.
     */
    public static Layer getConvLayer(int numFilters, int filterSize, int stride, String activation, int numberOfNeuronsToLayer) {
        int layerId = numberOfLayers++;
        int[] inputSize = NetworkManager.getLayerInputSize(layerId);
        int[] outputSize = NetworkHelper.calculateConvolutionOutputSize(numFilters, filterSize, stride, inputSize);
        LayerActivation layerActivation = NetworkHelper.matchActivation(activation);

        LayerConvolution layer = new LayerConvolution(layerId, layerActivation);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
        layer.setNumberOfFilters(numFilters);
        layer.setFilterSize(filterSize);
        layer.setStride(stride);

        layer.createNeurons(numberOfNeuronsToLayer);

        StringBuilder sb = new StringBuilder();
        sb.append("Layer(" + layerId + ")(convolution), ");
        sb.append("filters("+numFilters+"), ");
        sb.append("filter-size("+filterSize+"), ");
        sb.append("stride("+stride+"), ");
        sb.append("activation("+activation+"), ");
        sb.append("output-size("+Arrays.toString(NetworkHelper.calculateConvolutionOutputSize(numFilters, filterSize, stride, inputSize))+").");
        System.out.println(sb);

        return layer;
    }

    /**
     * Creates and sets up an instance of {@link LayerPool}.
     * @param poolSize The size of the receptive field.
     * @param stride The stride that this layer will use.
     * @param poolType The type of pooling operation, see {@link LayerType}.
     * @param numberOfNeuronsToLayer The number of neurons in this layer.
     * @return An instance of {@link LayerPool}.
     */
    public static Layer getPoolLayer(int poolSize, int stride, String poolType, int numberOfNeuronsToLayer) {
        int layerId = numberOfLayers++;
        int[] inputSize = NetworkManager.getLayerInputSize(layerId);
        int[] outputSize = NetworkHelper.calculatePoolOutputSize(poolSize, stride, inputSize);
        LayerType type = NetworkHelper.matchPoolType(poolType);

        LayerPool layer = new LayerPool(layerId, type);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
        layer.setPoolSize(poolSize);
        layer.setStride(stride);

        layer.createNeurons(numberOfNeuronsToLayer);

        StringBuilder sb = new StringBuilder();
        sb.append("Layer(" + layerId + ")(pool), ");
        sb.append("pools("+numberOfNeuronsToLayer+"), ");
        sb.append("pool-size("+poolSize+"), ");
        sb.append("stride("+stride+"), ");
        sb.append("output-size("+Arrays.toString(NetworkHelper.calculatePoolOutputSize(poolSize, stride, inputSize))+").");
        System.out.println(sb);

        return layer;
    }

    /**
     * Creates and sets up an instance of {@link LayerFlatten}.
     * @param numberOfNeuronsToLayer The number of neurons to the layer, used to calculate how many neurons this layer needs.
     * @return An instance of {@link LayerFlatten}.
     */
    public static Layer getFlattenLayer(int numberOfNeuronsToLayer) {
        int layerId = numberOfLayers++;
        int[] inputSize = NetworkManager.getLayerInputSize(layerId);
        int[] outputSize = NetworkHelper.caculateFlattenLayerOutputSize(inputSize);

        LayerFlatten layer = new LayerFlatten(layerId);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);

        layer.createNeurons(numberOfNeuronsToLayer);

        StringBuilder sb = new StringBuilder();
        sb.append("Layer(" + layerId + ")(flatten), ");
        sb.append("neurons("+outputSize[2]+").");
        System.out.println(sb);

        return layer;
    }

    /**
     * Creates and sets up an instance of {@link LayerFullyConnected}.
     * @param numberOfNeurons The number of neurons this layer should have.
     * @param activation The {@link LayerActivation} for this layer.
     * @param numberOfNeuronsToLayer The number of neurons leading to this layer, used to calculate how many weights each neuron will have.
     * @return An instance of {@link LayerFullyConnected}.
     */
    public static Layer getFCLayer(int numberOfNeurons, String activation, int numberOfNeuronsToLayer) {
        int layerId = numberOfLayers++;
        int[] inputSize = NetworkManager.getLayerInputSize(layerId);
        int[] outputSize = new int[] {1, 1, numberOfNeurons};
        LayerActivation layerActivation = NetworkHelper.matchActivation(activation);

        LayerFullyConnected layer = new LayerFullyConnected(layerId, layerActivation);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
        layer.setNumberOfNeurons(numberOfNeurons);
        layer.setNumberOfWeights(numberOfNeuronsToLayer);

        layer.createNeurons(numberOfNeurons);

        StringBuilder sb = new StringBuilder();
        sb.append("Layer(" + layerId + ")(fully-connected), ");
        sb.append("neurons("+numberOfNeurons+"), ");
        sb.append("weights("+numberOfNeurons*numberOfNeuronsToLayer+").");
        System.out.println(sb);

        return layer;
    }

    /**
     * Creates and sets up an instance of {@link LayerOutput}.
     * @param numberOfOutputs The number of neurons this layer should have.
     * @param numberOfNeuronsToLayer The number of neurons leading to this layer, used to calculate how many weights each neuron will have.
     * @return An instance of {@link LayerOutput}.
     */
    public static Layer getOutputLayer(int numberOfOutputs, int numberOfNeuronsToLayer) {
        int layerId = numberOfLayers++;
        int[] inputSize = NetworkManager.getLayerInputSize(layerId);
        int[] outputSize = new int[] {1, 1, numberOfOutputs};

        LayerOutput layer = new LayerOutput(layerId);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
        layer.setNumberOfOutputs(numberOfOutputs);
        layer.setNumberOfWeights(numberOfNeuronsToLayer);

        layer.createNeurons(numberOfOutputs);

        StringBuilder sb = new StringBuilder();
        sb.append("Layer(" + layerId + ")(output), ");
        sb.append("outputs("+numberOfOutputs+"), ");
        sb.append("weights("+numberOfOutputs*numberOfNeuronsToLayer+").");
        System.out.println(sb);
        
        return layer;
    }
}