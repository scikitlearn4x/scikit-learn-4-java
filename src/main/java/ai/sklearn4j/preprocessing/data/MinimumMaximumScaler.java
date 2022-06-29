// ==================================================================
// Inference for MinMaxScaler
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
// ==================================================================
package ai.sklearn4j.preprocessing.data;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Transform features by scaling each feature to a given range.
 * This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between zero
 * and one.
 * The transformation is given by::
 * X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
 * X_scaled = X_std * (max - min) + min
 * where min, max = feature_range.
 * This transformation is often used as an alternative to zero mean, unit
 * variance scaling.
 */
public class MinimumMaximumScaler extends TransformerMixin<NumpyArray<Double>, NumpyArray<Double>> {
    /**
     * Instantiate a new object of MinimumMaximumScaler.
     */
    public MinimumMaximumScaler() {

    }

    /**
     * Per feature adjustment for minimum. Equivalent to `min - X.min(axis=0)
     * * self.scale_`
     */
    private NumpyArray<Double> min = null;

    /**
     * Per feature relative scaling of the data. Equivalent to `(max - min) /
     * (X.max(axis=0) - X.min(axis=0))`
     */
    private NumpyArray<Double> scale = null;

    /**
     * Per feature minimum seen in the data
     */
    private NumpyArray<Double> dataMin = null;

    /**
     * Per feature maximum seen in the data
     */
    private NumpyArray<Double> dataMax = null;

    /**
     * Per feature range `(data_max_ - data_min_)` seen in the data
     */
    private NumpyArray<Double> dataRange = null;

    /**
     * Number of features seen during `fit`.
     */
    private long nFeaturesIn = 0;

    /**
     * The number of samples processed by the estimator. It will be reset on
     * new calls to fit, but increments across `partial_fit` calls.
     */
    private long nSamplesSeen = 0;

    /**
     * Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     */
    private String[] featureNamesIn = null;

    /**
     * Internal field of scikit-learn object.
     */
    private boolean clip = false;

    /**
     * Internal field of scikit-learn object.
     */
    private double[] featureRange = null;

    /**
     * Sets the Per feature adjustment for minimum. Equivalent to `min - X.min(axis=0)
     * * self.scale_`
     *
     * @param value The new value for min.
     */
    public void setMin(NumpyArray value) {
        this.min = value;
    }


    /**
     * Gets the Per feature adjustment for minimum. Equivalent to `min - X.min(axis=0)
     * * self.scale_`
     */
    public NumpyArray getMin() {
        return this.min;
    }


    /**
     * Sets the Per feature relative scaling of the data. Equivalent to `(max - min) /
     * (X.max(axis=0) - X.min(axis=0))`
     *
     * @param value The new value for scale.
     */
    public void setScale(NumpyArray value) {
        this.scale = value;
    }


    /**
     * Gets the Per feature relative scaling of the data. Equivalent to `(max - min) /
     * (X.max(axis=0) - X.min(axis=0))`
     */
    public NumpyArray getScale() {
        return this.scale;
    }


    /**
     * Sets the Per feature minimum seen in the data
     *
     * @param value The new value for dataMin.
     */
    public void setDataMin(NumpyArray value) {
        this.dataMin = value;
    }


    /**
     * Gets the Per feature minimum seen in the data
     */
    public NumpyArray getDataMin() {
        return this.dataMin;
    }


    /**
     * Sets the Per feature maximum seen in the data
     *
     * @param value The new value for dataMax.
     */
    public void setDataMax(NumpyArray value) {
        this.dataMax = value;
    }


    /**
     * Gets the Per feature maximum seen in the data
     */
    public NumpyArray getDataMax() {
        return this.dataMax;
    }


    /**
     * Sets the Per feature range `(data_max_ - data_min_)` seen in the data
     *
     * @param value The new value for dataRange.
     */
    public void setDataRange(NumpyArray value) {
        this.dataRange = value;
    }


    /**
     * Gets the Per feature range `(data_max_ - data_min_)` seen in the data
     */
    public NumpyArray getDataRange() {
        return this.dataRange;
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
     * Sets the The number of samples processed by the estimator. It will be reset on
     * new calls to fit, but increments across `partial_fit` calls.
     *
     * @param value The new value for nSamplesSeen.
     */
    public void setNSamplesSeen(long value) {
        this.nSamplesSeen = value;
    }


    /**
     * Gets the The number of samples processed by the estimator. It will be reset on
     * new calls to fit, but increments across `partial_fit` calls.
     */
    public long getNSamplesSeen() {
        return this.nSamplesSeen;
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
     * Sets the value of Clip
     *
     * @param value The new value for Clip.
     */
    public void setClip(boolean value) {
        this.clip = value;
    }


    /**
     * Gets the value of Clip
     */
    public boolean getClip() {
        return this.clip;
    }


    /**
     * Sets the value of FeatureRange
     *
     * @param value The new value for FeatureRange.
     */
    public void setFeatureRange(double[] value) {
        this.featureRange = value;
    }


    /**
     * Gets the value of FeatureRange
     */
    public double[] getFeatureRange() {
        return this.featureRange;
    }


    /**
     * Takes the input array and transforms it.
     *
     * @param array The array to transform.
     * @return The transformed array.
     */
    @Override
    public NumpyArray<Double> transform(NumpyArray<Double> array) {
        NumpyArray<Double> result = Numpy.multiply(array, scale);
        result = Numpy.add(result, min);

        if (clip) {
            result = Numpy.clip(result, featureRange[0], featureRange[1]);
        }

        return result;
    }

    /**
     * Takes a transformed array and reveres the transformation.
     *
     * @param array The array to apply reveres transform.
     * @return The inversed transform of array.
     */
    @Override
    public NumpyArray<Double> inverseTransform(NumpyArray<Double> array) {
        NumpyArray<Double> result = Numpy.subtract(array, min);
        result = Numpy.divide(result, scale);

        return result;
    }
}