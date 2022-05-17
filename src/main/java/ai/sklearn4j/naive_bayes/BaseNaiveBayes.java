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
     *
     * I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of shape
     * (n_samples, n_classes).
     *
     * predict, predictProbabilities, and predictLogProbabilities pass the input over to
     * jointLogLikelihood.
     *
     * @param x An array-like of shape (n_samples, n_classes).
     */
    protected abstract NumpyArray<Double> jointLogLikelihood(NumpyArray<Double> x);

    public NumpyArray<Integer> predict(NumpyArray<Double> x) {
        NumpyArray<Double> jll = jointLogLikelihood(x);
        return Numpy.argmax(jll, 1);
    }

    public NumpyArray<Double> predictLogProbabilities(NumpyArray<Double> x) {
        NumpyArray<Double> jll = jointLogLikelihood(x);
        NumpyArray<Double> logProbabilityOfX = Scipy.logSumExponent(jll, 1);

        return Numpy.subtract(jll, Numpy.atLeast2D(logProbabilityOfX).transpose());
    }

    public NumpyArray<Double> predictProbabilities(NumpyArray<Double> x) {
        return Numpy.exp(predictLogProbabilities(x));
    }
}
