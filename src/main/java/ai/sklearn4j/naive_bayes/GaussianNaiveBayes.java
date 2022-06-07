package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.core.Constants;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.core.libraries.numpy.wrappers.Dim2DoubleNumpyWrapper;

/**
 * Naive Bayes classifier for normal distributed models.
 */
public class GaussianNaiveBayes extends BaseNaiveBayes {
    /**
     * The prior probability of each class.
     * Dimension: (class_count)
     */
    private NumpyArray<Double> classPriors = null;

    /**
     * The user provided class priors. If specified, the priors are not adjusted according to the data.
     * Dimension: (class_count)
     */
    private NumpyArray<Double> priors = null;

    /**
     * Variance of each feature per class.
     * Dimension: (n_classes, n_features)
     */
    private NumpyArray<Double> sigma = null;

    /**
     * Mean of each feature per class.
     * Dimension: (n_classes, n_features)
     */
    private NumpyArray<Double> theta = null;

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
        int count = x.getShape()[0];
        int classCount = classCounts.getShape()[0];
        int featureCount = sigma.getShape()[1];
        double[][] jointLogLikelihood = new double[count][classCount];

        double[][] variance = ((Dim2DoubleNumpyWrapper) sigma.getWrapper()).getArray();
        double[][] mean = ((Dim2DoubleNumpyWrapper) theta.getWrapper()).getArray();


        for (int cls = 0; cls < classCount; cls++) {
            double sumOfLogVariance = 0;

            for (int feature = 0; feature < featureCount; feature++) {
                sumOfLogVariance += Math.log(2.0 * Constants.PI * variance[cls][feature]);
            }

            double jointi = Math.log(classPriors.get(cls));

            for (int i = 0; i < count; i++) {
                double value = 0;

                for (int feature = 0; feature < featureCount; feature++) {
                    double diff = x.get(i, feature) - mean[cls][feature];
                    value += (Math.pow(x.get(i, feature) - mean[cls][feature], 2) / variance[cls][feature]);
                }

                value = -0.5 * (sumOfLogVariance + value);
                jointLogLikelihood[i][cls] = value + jointi;
            }
        }

        return NumpyArrayFactory.from(jointLogLikelihood);
    }

    /**
     * Gets the class priors.
     *
     * @return The value of class priors.
     */
    public NumpyArray<Double> getClassPriors() {
        return classPriors;
    }

    /**
     * Sets the class priors.
     *
     * @param classPriors New value to be stored.
     */
    public void setClassPriors(NumpyArray<Double> classPriors) {
        this.classPriors = classPriors;
    }

    /**
     * Gets the priors.
     *
     * @return The value of priors.
     */
    public NumpyArray<Double> getPriors() {
        return priors;
    }

    /**
     * Sets the priors.
     *
     * @param priors New value to be stored.
     */
    public void setPriors(NumpyArray<Double> priors) {
        this.priors = priors;
    }

    /**
     * Gets the variance of the features.
     *
     * @return The value of variance of the features.
     */
    public NumpyArray<Double> getSigma() {
        return sigma;
    }

    /**
     * Sets the variance of the features.
     *
     * @param sigma New value to be stored.
     */
    public void setSigma(NumpyArray<Double> sigma) {
        this.sigma = sigma;
    }

    /**
     * Gets the mean of the features.
     *
     * @return The value of mean of the features.
     */
    public NumpyArray<Double> getTheta() {
        return theta;
    }

    /**
     * Sets the mean of the features.
     *
     * @param theta New value to be stored.
     */
    public void setTheta(NumpyArray<Double> theta) {
        this.theta = theta;
    }
}
