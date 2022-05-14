package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.base.ClassifierMixin;
import ai.sklearn4j.core.numpy.NumpyArray;
import ai.sklearn4j.core.helpers.ArrayHelper;
import ai.sklearn4j.core.numpy.NumpyArrayFactory;

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
     * predict, predictProbabilities, and predictLogProbabilities pass the input through
     * checkX and handle it over to jointLogLikelihood.
     *
     * @param x An array-like of shape (n_samples, n_classes).
     */
    protected abstract NumpyArray<Double> jointLogLikelihood(NumpyArray<Double> x);

    /**
     * To be overridden in subclasses with the actual checks. Only used in predict* methods.
     * @param x An array-like of shape (n_samples, n_classes).
     */
    protected abstract NumpyArray<Double> checkX(NumpyArray<Double> x);

    public NumpyArray<Integer> predict(NumpyArray<Double> x) {
        x = this.checkX(x);
        NumpyArray<Double> jll = jointLogLikelihood(x);
        return NumpyArrayFactory.from(ArrayHelper.getArgmaxFromClassProbabilityDistribution(jll));
    }

    public NumpyArray<Double> predictLogProbabilities(NumpyArray<Double> x) {
        throw new RuntimeException();
    }

    public NumpyArray<Double> predictProbabilities(NumpyArray<Double> x) {
        return ArrayHelper.exponentOfLogProbabilities(predictLogProbabilities(x));
    }
}
