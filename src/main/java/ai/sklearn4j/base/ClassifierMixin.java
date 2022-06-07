package ai.sklearn4j.base;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Mixin class for all classifiers in scikit-learn.
 */
public abstract class ClassifierMixin {
    /**
     * Names of features seen during training. Defined only when `X` has feature names that are all strings.
     */
    private String[] featureNamesIn = null;

    /**
     * Number of features seen during training.
     */
    private int numberOfFeatures = 0;

    /**
     * The frequency of each class in the training set.
     * Dimension: (class_count)
     */
    protected NumpyArray<Double> classCounts = null;

    /**
     * The list of class IDs.
     * Dimension: (class_count)
     */
    protected NumpyArray<Long> classes = null;

    /**
     * Perform classification on an array of test vectors X.
     *
     * @param x Array-like of shape (n_samples, n_features) The input samples.
     * @return NumpyArray of shape (n_samples,) Predicted target values for X.
     */
    public abstract NumpyArray<Long> predict(NumpyArray<Double> x);

    /**
     * Return log-probability estimates for the test vector X.
     *
     * @param x array-like of shape (n_samples, n_features) The input samples.
     * @return array-like of shape (n_samples, n_classes)
     * Returns the log-probability of the samples for each class in
     * the model. The columns correspond to the classes in sorted
     * order, as they appear in the attribute :term:`classes_`.
     */
    public abstract NumpyArray<Double> predictLogProbabilities(NumpyArray<Double> x);

    /**
     * Return probability estimates for the test vector X.
     *
     * @param x array-like of shape (n_samples, n_features) The input samples.
     * @return array-like of shape (n_samples, n_classes)
     * Returns the probability of the samples for each class in
     * the model. The columns correspond to the classes in sorted
     * order, as they appear in the attribute :term:`classes_`.
     */
    public abstract NumpyArray<Double> predictProbabilities(NumpyArray<Double> x);

    /**
     * Gets the feature names.
     *
     * @return The value of feature names.
     */
    public String[] getFeatureNamesIn() {
        return featureNamesIn;
    }

    /**
     * Sets the feature names.
     *
     * @param featureNamesIn New value to be stored.
     */
    public void setFeatureNamesIn(String[] featureNamesIn) {
        this.featureNamesIn = featureNamesIn;
    }

    /**
     * Gets the number of features.
     *
     * @return The value of number of features.
     */
    public int getNumberOfFeatures() {
        return numberOfFeatures;
    }

    /**
     * Sets the number of features.
     *
     * @param numberOfFeatures New value to be stored.
     */
    public void setNumberOfFeatures(int numberOfFeatures) {
        this.numberOfFeatures = numberOfFeatures;
    }

    /**
     * Gets the class counts.
     *
     * @return The value of class counts.
     */
    public NumpyArray<Double> getClassCounts() {
        return classCounts;
    }

    /**
     * Sets the class counts.
     *
     * @param classCounts New value to be stored.
     */
    public void setClassCounts(NumpyArray<Double> classCounts) {
        this.classCounts = classCounts;
    }

    /**
     * Gets the classes.
     *
     * @return The value of classes.
     */
    public NumpyArray<Long> getClasses() {
        return classes;
    }

    /**
     * Sets the classes.
     *
     * @param classes New value to be stored.
     */
    public void setClasses(NumpyArray<Long> classes) {
        this.classes = classes;
    }

}
