package waddington.kai.main.knnf;

/**
 * Used to track activation types used in layers.
 * <p>
 * Does not include the softmax activation type as it is currently the only option for and is unique to the output layer.
 */
public enum LayerActivation {
    linear,
    sigmoid,
    tanh,
    relu,
    lrelu
}