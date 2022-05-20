package ai.sklearn4j.base;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Mixin class for all classifiers in scikit-learn.
 */
public abstract class ClassifierMixin {
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
     *
     * @return array-like of shape (n_samples, n_classes)
     *         Returns the log-probability of the samples for each class in
     *         the model. The columns correspond to the classes in sorted
     *         order, as they appear in the attribute :term:`classes_`.
     */
    public abstract NumpyArray<Double> predictLogProbabilities(NumpyArray<Double> x);

    /**
     * Return probability estimates for the test vector X.
     *
     * @param x array-like of shape (n_samples, n_features) The input samples.
     *
     * @return array-like of shape (n_samples, n_classes)
     *         Returns the probability of the samples for each class in
     *         the model. The columns correspond to the classes in sorted
     *         order, as they appear in the attribute :term:`classes_`.
     */
    public abstract NumpyArray<Double> predictProbabilities(NumpyArray<Double> x);
}
