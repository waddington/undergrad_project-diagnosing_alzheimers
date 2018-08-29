/*
* Exception for if there is no implementation of an activation type.
*/

package waddington.kai.main.knnf.exceptions;

public class MissingActivationMethodException extends RuntimeException {

    public MissingActivationMethodException(String message) {
        super("No method for this activation type found. " + message);
    }
}