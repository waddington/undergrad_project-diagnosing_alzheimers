/*
* Exception for if the number of training examples and labels do not match.
*/

package waddington.kai.main.knnf.exceptions;

public class TrainingDataLengthException extends RuntimeException {
    
    public TrainingDataLengthException(String message) {
        super("Training data and label error. Number of training examples does not match number of training labels. " + message);
    }
}