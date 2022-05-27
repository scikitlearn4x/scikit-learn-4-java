package ai.sklearn4j.core.libraries.numpy;

/**
 * A custom exception that provides information on what went wrong in numpy's processing.
 */
public class NumpyOperationException extends RuntimeException {
    /**
     * Instantiate a new exception object.
     *
     * @param message The content of the error.
     */
    public NumpyOperationException(String message) {
        super(message);
    }
}
