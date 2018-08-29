/*
* Exception for if network training method is called without setting a training termination condition
*/

package waddington.kai.main.knnf.exceptions;

public class MissingTerminationConditionException extends RuntimeException {
    
    public MissingTerminationConditionException(String message) {
        super("No training termination condition set. At least one termination condition required. " + message);
    }
}