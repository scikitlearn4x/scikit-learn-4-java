package ai.sklearn4j.core;

/**
 * A custom exception that provides information on what went wrong in the library general processing.
 */
public class ScikitLearnCoreException extends RuntimeException {
    /**
     * Instantiate a new exception object.
     *
     * @param message The content of the error.
     */
    public ScikitLearnCoreException(String message) {
        super(message);
    }

}
