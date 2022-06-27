// ==================================================================
// Inference for RobustScaler
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
// ==================================================================
package ai.sklearn4j.preprocessing.data;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Scale features using statistics that are robust to outliers.
 * This Scaler removes the median and scales the data according to the
 * quantile range (defaults to IQR: Interquartile Range). The IQR is the
 * range between the 1st quartile (25th quantile) and the 3rd quartile
 * (75th quantile).
 * Centering and scaling happen independently on each feature by
 * computing the relevant statistics on the samples in the training set.
 * Median and interquartile range are then stored to be used on later
 * data using the :meth:`transform` method.
 * Standardization of a dataset is a common requirement for many machine
 * learning estimators. Typically this is done by removing the mean and
 * scaling to unit variance. However, outliers can often influence the
 * sample mean / variance in a negative way. In such cases, the median
 * and the interquartile range often give better results.
 * .. versionadded:: 0.17
 */
public class RobustScaler extends TransformerMixin<NumpyArray<Double>, NumpyArray<Double>> {
    /**
     * Instantiate a new object of RobustScaler.
     */
    public RobustScaler() {

    }

    /**
     * The median value for each feature in the training set.
     */
    private NumpyArray center = null;

    /**
     * The (scaled) interquartile range for each feature in the training
     * set.
     */
    private NumpyArray scale = null;

    private boolean withCentering = true;

    private boolean withScaling = true;

    private boolean unitVariance = true;

    private double[] quantilesRange = null;

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
     * Sets the The median value for each feature in the training set.
     *
     * @param value The new value for center.
     */
    public void setCenter(NumpyArray value) {
        this.center = value;
    }


    /**
     * Gets the The median value for each feature in the training set.
     */
    public NumpyArray getCenter() {
        return this.center;
    }


    /**
     * Sets the The (scaled) interquartile range for each feature in the training
     * set.
     *
     * @param value The new value for scale.
     */
    public void setScale(NumpyArray value) {
        this.scale = value;
    }


    /**
     * Gets the The (scaled) interquartile range for each feature in the training
     * set.
     */
    public NumpyArray getScale() {
        return this.scale;
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

    public boolean isWithCentering() {
        return withCentering;
    }

    public void setWithCentering(boolean withCentering) {
        this.withCentering = withCentering;
    }

    public boolean isWithScaling() {
        return withScaling;
    }

    public void setWithScaling(boolean withScaling) {
        this.withScaling = withScaling;
    }

    public boolean isUnitVariance() {
        return unitVariance;
    }

    public void setUnitVariance(boolean unitVariance) {
        this.unitVariance = unitVariance;
    }

    public double[] getQuantilesRange() {
        return quantilesRange;
    }

    public void setQuantilesRange(double[] quantilesRange) {
        this.quantilesRange = quantilesRange;
    }

    /**
     * Takes the input array and transforms it.
     *
     * @param array The array to transform.
     * @return The transformed array.
     */
    @Override
    public NumpyArray<Double> transform(NumpyArray<Double> array) {
        NumpyArray<Double> result = array;

        if (withCentering) {
            result = Numpy.subtract(result, center);
        }

        if (withScaling) {
            result = Numpy.divide(result, scale);
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
        NumpyArray<Double> result = array;

        if (withScaling) {
            result = Numpy.multiply(result, scale);
        }

        if (withCentering) {
            result = Numpy.add(result, center);
        }

        return result;
    }
}