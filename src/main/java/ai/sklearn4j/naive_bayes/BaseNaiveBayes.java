package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.core.NumpyArray;

/**
 * Abstract base class for naive Bayes estimators
 */
public abstract class BaseNaiveBayes {
    /**
     * Compute the unnormalized posterior log probability of X.
     *
     * I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of shape
     * (n_samples, n_classes).
     *
     * predict, predictProbabilities, and predictLogProbabilities pass the input through
     * checkX and handle it over to jointLogLikelihood.
     *
     * @param x An array-like of shape (n_samples, n_classes).
     */
    protected abstract NumpyArray jointLogLikelihood(NumpyArray x);

    /**
     * To be overridden in subclasses with the actual checks. Only used in predict* methods.
     * @param x An array-like of shape (n_samples, n_classes).
     */
    protected abstract NumpyArray checkX(NumpyArray x);

    /**
     * Perform classification on an array of test vectors X.
     *
     * @param x Array-like of shape (n_samples, n_features) The input samples.
     * @return NumpyArray of shape (n_samples,) Predicted target values for X.
     */
    public NumpyArray predict(NumpyArray x) {
        x = this.checkX(x);
        NumpyArray jll = jointLogLikelihood(x);
    }
}
