/*
* Exception for when the calculated output size of a layer or neuron is found to be invalid during the network validity checks.
*/

package waddington.kai.main.knnf.exceptions;

public class InvalidOutputSizeException extends RuntimeException {

    public InvalidOutputSizeException(String message) {
        super("Calculated output size invalid. " + message);
    }
}