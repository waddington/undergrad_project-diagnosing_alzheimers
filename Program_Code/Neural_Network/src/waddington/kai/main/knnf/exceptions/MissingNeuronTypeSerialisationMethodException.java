/*
* Exception for if there is no serialisation method implemented for a given layer type.
*/

package waddington.kai.main.knnf.exceptions;

public class MissingNeuronTypeSerialisationMethodException extends RuntimeException {

    public MissingNeuronTypeSerialisationMethodException(String message) {
        super("No method found for serialising this neuron. " + message);
    }
}