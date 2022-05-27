package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.core.Constants;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.core.libraries.numpy.wrappers.Dim1DoubleNumpyWrapper;
import ai.sklearn4j.core.libraries.numpy.wrappers.Dim2DoubleNumpyWrapper;

import java.util.ArrayList;
import java.util.List;

/**
 * Naive Bayes classifier for normal distributed models.
 */
public class GaussianNaiveBayes extends BaseNaiveBayes {
    /**
     * The frequency of each class in the training set.
     * Dimension: (class_count)
     */
    private NumpyArray<Double> classCounts = null;

    /**
     * The prior probability of each class.
     * Dimension: (class_count)
     */
    private NumpyArray<Double> classPriors = null;

    /**
     * The list of class IDs.
     * Dimension: (class_count)
     */
    private NumpyArray<Long> classes = null;

    /**
     * The user provided class priors. If specified, the priors are not adjusted according to the data.
     * Dimension: (class_count)
     */
    private NumpyArray<Double> priors = null;

    /**
     * Names of features seen during training. Defined only when `X` has feature names that are all strings.
     */
    private String[] featureNamesIn = null;

    /**
     * Number of features seen during training.
     */
    private int numberOfFeatures = 0;

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
     *
     * I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of shape
     * (n_samples, n_classes).
     *
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
     * Gets the class counts.
     * @return The value of class counts.
     */
    public NumpyArray<Double> getClassCounts() {
        return classCounts;
    }

    /**
     * Sets the class counts.
     * @param classCounts New value to be stored.
     */
    public void setClassCounts(NumpyArray<Double> classCounts) {
        this.classCounts = classCounts;
    }

    /**
     * Gets the class priors.
     * @return The value of class priors.
     */
    public NumpyArray<Double> getClassPriors() {
        return classPriors;
    }

    /**
     * Sets the class priors.
     * @param classPriors New value to be stored.
     */
    public void setClassPriors(NumpyArray<Double> classPriors) {
        this.classPriors = classPriors;
    }

    /**
     * Gets the classes.
     * @return The value of classes.
     */
    public NumpyArray<Long> getClasses() {
        return classes;
    }

    /**
     * Sets the classes.
     * @param classes New value to be stored.
     */
    public void setClasses(NumpyArray<Long> classes) {
        this.classes = classes;
    }

    /**
     * Gets the priors.
     * @return The value of priors.
     */
    public NumpyArray<Double> getPriors() {
        return priors;
    }

    /**
     * Sets the priors.
     * @param priors New value to be stored.
     */
    public void setPriors(NumpyArray<Double> priors) {
        this.priors = priors;
    }

    /**
     * Gets the feature names.
     * @return The value of feature names.
     */
    public String[] getFeatureNamesIn() {
        return featureNamesIn;
    }

    /**
     * Sets the feature names.
     * @param featureNamesIn New value to be stored.
     */
    public void setFeatureNamesIn(String[] featureNamesIn) {
        this.featureNamesIn = featureNamesIn;
    }

    /**
     * Gets the number of features.
     * @return The value of number of features.
     */
    public int getNumberOfFeatures() {
        return numberOfFeatures;
    }

    /**
     * Sets the number of features.
     * @param numberOfFeatures New value to be stored.
     */
    public void setNumberOfFeatures(int numberOfFeatures) {
        this.numberOfFeatures = numberOfFeatures;
    }

    /**
     * Gets the variance of the features.
     * @return The value of variance of the features.
     */
    public NumpyArray<Double> getSigma() {
        return sigma;
    }

    /**
     * Sets the variance of the features.
     * @param sigma New value to be stored.
     */
    public void setSigma(NumpyArray<Double> sigma) {
        this.sigma = sigma;
    }

    /**
     * Gets the mean of the features.
     * @return The value of mean of the features.
     */
    public NumpyArray<Double> getTheta() {
        return theta;
    }

    /**
     * Sets the mean of the features.
     * @param theta New value to be stored.
     */
    public void setTheta(NumpyArray<Double> theta) {
        this.theta = theta;
    }
}
