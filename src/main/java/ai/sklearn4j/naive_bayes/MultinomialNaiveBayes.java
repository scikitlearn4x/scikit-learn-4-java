package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.utils.ExtMath;

/**
 * Naive Bayes classifier for multinomial distributed models.
 */
public class MultinomialNaiveBayes extends BaseNaiveBayes {
    /**
     * Empirical log probability of features given a class, P(x_i|y).
     */
    private NumpyArray<Double> featureLogProbabilities = null;

    /**
     * Log probability of each class (smoothed).
     */
    private NumpyArray<Double> classLogPrior = null;

    /**
     * Number of samples encountered for each (class, feature) during fitting. This value is weighted by the sample weight when provided.
     */
    private NumpyArray<Double> featureCounts = null;

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
     * @return The unnormalized posterior log probability of X.
     */
    @Override
    protected NumpyArray<Double> jointLogLikelihood(NumpyArray<Double> x) {
        return Numpy.add(ExtMath.dot(x, featureLogProbabilities.transpose()), classLogPrior);
    }

    /**
     * Gets the empirical log probability of features given a class, P(x_i|y).
     *
     * @return Empirical log probability of features given a class, P(x_i|y).
     */
    public NumpyArray<Double> getFeatureLogProbabilities() {
        return featureLogProbabilities;
    }

    /**
     * Sets the empirical log probability of features given a class, P(x_i|y).
     *
     * @param featureLogProbabilities The empirical log probability of features given a class, P(x_i|y).
     */
    public void setFeatureLogProbabilities(NumpyArray<Double> featureLogProbabilities) {
        this.featureLogProbabilities = featureLogProbabilities;
    }

    /**
     * Gets the log probability of each class (smoothed).
     *
     * @return Log probability of each class (smoothed).
     */
    public NumpyArray<Double> getClassLogPrior() {
        return classLogPrior;
    }

    /**
     * Sets the log probability of each class (smoothed).
     *
     * @param classLogPrior The log probability of each class (smoothed).
     */
    public void setClassLogPrior(NumpyArray<Double> classLogPrior) {
        this.classLogPrior = classLogPrior;
    }

    /**
     * Gets the number of samples encountered for each (class, feature) during fitting. This value is weighted by the sample weight when provided.
     *
     * @return Number of samples encountered for each (class, feature) during fitting. This value is weighted by the sample weight when provided.
     */
    public NumpyArray<Double> getFeatureCounts() {
        return featureCounts;
    }

    /**
     * Sets the number of samples encountered for each (class, feature) during fitting. This value is weighted by the sample weight when provided.
     *
     * @param featureCounts The number of samples encountered for each (class, feature) during fitting. This value is weighted by the sample weight when provided.
     */
    public void setFeatureCount(NumpyArray<Double> featureCounts) {
        this.featureCounts = featureCounts;
    }

}