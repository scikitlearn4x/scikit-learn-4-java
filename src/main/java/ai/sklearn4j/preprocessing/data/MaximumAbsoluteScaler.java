// ==================================================================
// Inference for MaxAbsScaler
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler
// ==================================================================
package ai.sklearn4j.preprocessing.data;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Scale each feature by its maximum absolute value.
 * This estimator scales and translates each feature individually such
 * that the maximal absolute value of each feature in the training set
 * will be 1.0. It does not shift/center the data, and thus does not
 * destroy any sparsity.
 * This scaler can also be applied to sparse CSR or CSC matrices.
 */

public class MaximumAbsoluteScaler extends TransformerMixin<NumpyArray<Double>, NumpyArray<Double>> {
    /**
     * Instantiate a new object of MaximumAbsoluteScaler.
     */
    public MaximumAbsoluteScaler() {

    }

    /**
     * Per feature relative scaling of the data.
     */
    private NumpyArray scale = null;

    /**
     * Per feature maximum absolute value.
     */
    private NumpyArray maxAbs = null;

    /**
     * Number of features seen during `fit`.
     */
    private long nFeaturesIn = 0;

    /**
     * Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     */
    private String[] featureNamesIn = null;

    /**
     * The number of samples processed by the estimator. Will be reset on new
     * calls to fit, but increments across `partial_fit` calls.
     */
    private long nSamplesSeen = 0;

    /**
     * Sets the Per feature relative scaling of the data.
     *
     * @param value The new value for scale.
     */
    public void setScale(NumpyArray value) {
        this.scale = value;
    }


    /**
     * Gets the Per feature relative scaling of the data.
     */
    public NumpyArray getScale() {
        return this.scale;
    }


    /**
     * Sets the Per feature maximum absolute value.
     *
     * @param value The new value for maxAbs.
     */
    public void setMaxAbs(NumpyArray value) {
        this.maxAbs = value;
    }


    /**
     * Gets the Per feature maximum absolute value.
     */
    public NumpyArray getMaxAbs() {
        return this.maxAbs;
    }


    /**
     * Sets the Number of features seen during `fit`.
     *
     * @param value The new value for nFeaturesIn.
     */
    public void setNFeaturesIn(long value) {
        this.nFeaturesIn = value;
    }


    /**
     * Gets the Number of features seen during `fit`.
     */
    public long getNFeaturesIn() {
        return this.nFeaturesIn;
    }


    /**
     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     *
     * @param value The new value for featureNamesIn.
     */
    public void setFeatureNamesIn(String[] value) {
        this.featureNamesIn = value;
    }


    /**
     * Gets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     */
    public String[] getFeatureNamesIn() {
        return this.featureNamesIn;
    }


    /**
     * Sets the The number of samples processed by the estimator. Will be reset on new
     * calls to fit, but increments across `partial_fit` calls.
     *
     * @param value The new value for nSamplesSeen.
     */
    public void setNSamplesSeen(long value) {
        this.nSamplesSeen = value;
    }


    /**
     * Gets the The number of samples processed by the estimator. Will be reset on new
     * calls to fit, but increments across `partial_fit` calls.
     */
    public long getNSamplesSeen() {
        return this.nSamplesSeen;
    }


    @Override
    public NumpyArray<Double> transform(NumpyArray<Double> array) {
        return Numpy.divide(array, scale);
    }

    @Override
    public NumpyArray<Double> inverseTransform(NumpyArray<Double> array) {
        return Numpy.multiply(array, scale);
    }
}