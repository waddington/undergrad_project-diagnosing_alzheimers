package waddington.kai.main.knnf.exceptions;

public class IllegalMethodCallException extends RuntimeException {

    public IllegalMethodCallException(String message) {
        super("This method should not be called. " + message);
    }
}