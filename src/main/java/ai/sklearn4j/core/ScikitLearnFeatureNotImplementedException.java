package ai.sklearn4j.core;

/**
 * A custom exception that indicates a feature has not been implemented yet.
 */
public class ScikitLearnFeatureNotImplementedException extends RuntimeException {
    /**
     * Instantiate a new exception object.
     */
    public ScikitLearnFeatureNotImplementedException() {
        super("This feature has not yet been implemented.");
    }

    /**
     * Instantiate a new exception object.
     *
     * @param message The content of the error.
     */
    public ScikitLearnFeatureNotImplementedException(String message) {
        super(message);
    }

}