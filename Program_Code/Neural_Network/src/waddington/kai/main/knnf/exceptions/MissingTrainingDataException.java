/*
* Exception for if the specified training data files cannot be found.
*/

package waddington.kai.main.knnf.exceptions;

public class MissingTrainingDataException extends RuntimeException {
    
    public MissingTrainingDataException(String message) {
        super("Cannot find training data at selected location. " + message);
    }
}