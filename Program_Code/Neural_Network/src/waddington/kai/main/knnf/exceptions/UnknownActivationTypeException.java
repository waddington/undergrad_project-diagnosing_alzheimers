/*
* Exception for if the user inputs an invalid activation type.
*/

package waddington.kai.main.knnf.exceptions;

public class UnknownActivationTypeException extends RuntimeException {
    
    public UnknownActivationTypeException(String message) {
        super("Selected activation type not known. " + message);
    }
}