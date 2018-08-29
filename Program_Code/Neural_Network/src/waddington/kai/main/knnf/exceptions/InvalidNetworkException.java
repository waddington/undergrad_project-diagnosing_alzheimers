/*
* Exception for a when a network model is loaded from a file however the network is invalid.
* Also thrown if an invalid network somehow passes the other checks (which it shouldn't)
*/

package waddington.kai.main.knnf.exceptions;

public class InvalidNetworkException extends RuntimeException {
    
    public InvalidNetworkException(String message) {
        super("Network setup is invalid. " + message);
    }
}