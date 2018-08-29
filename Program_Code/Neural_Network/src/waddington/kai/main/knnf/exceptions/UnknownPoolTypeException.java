/*
* Exception for if the user inputs an invalid pooling type.
*/

package waddington.kai.main.knnf.exceptions;

public class UnknownPoolTypeException extends RuntimeException {
    
    public UnknownPoolTypeException(String message) {
        super("Selected pool type not known. " + message);
    }
}