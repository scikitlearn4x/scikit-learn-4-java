package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.utils.ExtMath;
import ai.sklearn4j.utils.Preprocessing;

/**
 * Naive Bayes classifier for bernoulli distributed models.
 */
public class BernoulliNaiveBayes extends BaseNaiveBayes {
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
     * Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
     */
    private double binarizationThreshold = 0.0;

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
/**
 """Calculate the posterior log probability of the samples X"""
 n_features = self.feature_log_prob_.shape[1]
 n_features_X = X.shape[1]

 if n_features_X != n_features:
 raise ValueError(
 "Expected input with %d features, got %d instead"
 % (n_features, n_features_X)
 )

 neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
 # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
 jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
 jll += self.class_log_prior_ + neg_prob.sum(axis=1)

 return jll
 */
        x = Preprocessing.binarizeInput(x, binarizationThreshold);

        int n_features = this.featureLogProbabilities.getShape()[1];
        int n_features_X = x.getShape()[1];

        if (n_features != n_features_X) {
            throw new ScikitLearnCoreException(String.format("Expected input with %d features, got %d instead.", n_features, n_features_X));
        }

        NumpyArray<Double> featureProbabilities = Numpy.exp(featureLogProbabilities);
        NumpyArray<Double> negProb = Numpy.log(Numpy.add(Numpy.multiply(featureProbabilities, -1), 1.0));
        // Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        NumpyArray<Double> jll = ExtMath.dot(x, Numpy.subtract(featureLogProbabilities, negProb).transpose());

        jll = Numpy.add(jll, Numpy.add(this.classLogPrior, Numpy.sum(negProb, 1, false)));

        return jll;
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

    /**
     * Gets the threshold for binarizing (mapping to booleans) of sample features. If None, input
     * is presumed to already consist of binary vectors.
     *
     * @return Threshold for binarizing (mapping to booleans) of sample features. If None, input
     * is presumed to already consist of binary vectors.
     */
    public double getBinarizationThreshold() {
        return binarizationThreshold;
    }

    /**
     * Sets the threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
     *
     * @param binarizationThreshold The threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
     */
    public void setBinarizationThreshold(double binarizationThreshold) {
        this.binarizationThreshold = binarizationThreshold;
    }
}
