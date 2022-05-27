package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.base.ClassifierMixin;
import ai.sklearn4j.core.libraries.Scipy;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Abstract base class for naive Bayes estimators
 */
public abstract class BaseNaiveBayes extends ClassifierMixin {
    /**
     * Compute the unnormalized posterior log probability of X.
     * <p>
     * I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of shape
     * (n_samples, n_classes).
     * <p>
     * predict, predictProbabilities, and predictLogProbabilities pass the input over to
     * jointLogLikelihood.
     *
     * @param x An array-like of shape (n_samples, n_classes).
     */
    protected abstract NumpyArray<Double> jointLogLikelihood(NumpyArray<Double> x);

    /**
     * Perform classification on an array of test vectors X.
     *
     * @param x Array-like of shape (n_samples, n_features) The input samples.
     * @return NumpyArray of shape (n_samples,) Predicted target values for X.
     */
    public NumpyArray<Long> predict(NumpyArray<Double> x) {
        NumpyArray<Double> jll = jointLogLikelihood(x);
        return Numpy.argmax(jll, 1);
    }

    /**
     * Return log-probability estimates for the test vector X.
     *
     * @param x array-like of shape (n_samples, n_features) The input samples.
     * @return array-like of shape (n_samples, n_classes)
     * Returns the log-probability of the samples for each class in
     * the model. The columns correspond to the classes in sorted
     * order, as they appear in the attribute :term:`classes_`.
     */
    public NumpyArray<Double> predictLogProbabilities(NumpyArray<Double> x) {
        NumpyArray<Double> jll = jointLogLikelihood(x);
        NumpyArray<Double> logProbabilityOfX = Scipy.logSumExponent(jll, 1);

        return Numpy.subtract(jll, Numpy.atLeast2D(logProbabilityOfX).transpose());
    }

    /**
     * Return probability estimates for the test vector X.
     *
     * @param x array-like of shape (n_samples, n_features) The input samples.
     * @return array-like of shape (n_samples, n_classes)
     * Returns the probability of the samples for each class in
     * the model. The columns correspond to the classes in sorted
     * order, as they appear in the attribute :term:`classes_`.
     */
    public NumpyArray<Double> predictProbabilities(NumpyArray<Double> x) {
        return Numpy.exp(predictLogProbabilities(x));
    }
}
