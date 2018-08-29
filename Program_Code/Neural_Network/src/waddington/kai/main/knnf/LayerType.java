package waddington.kai.main.knnf;

/**
 * Used to track the types of layers in the neural network.
 */
public enum LayerType {
    input,
    conv,
    maxPool,
    minPool,
    flatten,
    fc,
    output
}