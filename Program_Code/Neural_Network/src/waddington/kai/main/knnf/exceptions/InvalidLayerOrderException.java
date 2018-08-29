/*
* Exeption for when layers are created in an invalid order.
*/

package waddington.kai.main.knnf.exceptions;

public class InvalidLayerOrderException extends RuntimeException {

    public InvalidLayerOrderException(String message) {
        super("Layer order invalid. " + message);
    }
}