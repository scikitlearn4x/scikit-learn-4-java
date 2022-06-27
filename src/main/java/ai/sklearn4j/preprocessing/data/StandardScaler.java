// ==================================================================
// Inference for StandardScaler
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
// ==================================================================
package ai.sklearn4j.preprocessing.data;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Standardize features by removing the mean and scaling to unit
 * variance.
 * The standard score of a sample `x` is calculated as:
 * z = (x - u) / s
 * where `u` is the mean of the training samples or zero if
 * `with_mean=False`, and `s` is the standard deviation of the training
 * samples or one if `with_std=False`.
 * Centering and scaling happen independently on each feature by
 * computing the relevant statistics on the samples in the training set.
 * Mean and standard deviation are then stored to be used on later data
 * using :meth:`transform`.
 * Standardization of a dataset is a common requirement for many machine
 * learning estimators: they might behave badly if the individual
 * features do not more or less look like standard normally distributed
 * data (e.g. Gaussian with 0 mean and unit variance).
 * For instance many elements used in the objective function of a
 * learning algorithm (such as the RBF kernel of Support Vector Machines
 * or the L1 and L2 regularizers of linear models) assume that all
 * features are centered around 0 and have variance in the same order. If
 * a feature has a variance that is orders of magnitude larger than
 * others, it might dominate the objective function and make the
 * estimator unable to learn from other features correctly as expected.
 * This scaler can also be applied to sparse CSR or CSC matrices by
 * passing `with_mean=False` to avoid breaking the sparsity structure of
 * the data.
 */
public class StandardScaler extends TransformerMixin<NumpyArray<Double>, NumpyArray<Double>> {
    /**
     * Instantiate a new object of StandardScaler.
     */
    public StandardScaler() {

    }

    /**
     * Per feature relative scaling of the data to achieve zero mean and unit
     * variance. Generally this is calculated using `np.sqrt(var_)`. If a
     * variance is zero, we can't achieve unit variance, and the data is left
     * as-is, giving a scaling factor of 1. `scale_` is equal to `None` when
     * `with_std=False`.
     */
    private NumpyArray<Double> scale = null;

    /**
     * The mean value for each feature in the training set. Equal to `None`
     * when `with_mean=False`.
     */
    private NumpyArray<Double> mean = null;

    /**
     * The variance for each feature in the training set. Used to compute
     * `scale_`. Equal to `None` when `with_std=False`.
     */
    private NumpyArray<Double> variance = null;

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
     * The number of samples processed by the estimator for each feature. If
     * there are no missing samples, the `n_samples_seen` will be an integer,
     * otherwise it will be an array of dtype int. If `sample_weights` are
     * used it will be a float (if no missing data) or an array of dtype
     * float that sums the weights seen so far. Will be reset on new calls to
     * fit, but increments across `partial_fit` calls.
     */
    private NumpyArray<Long> nSamplesSeen = null;

    /**
     * Internal field of scikit-learn object.
     */
    private boolean withMean = true;

    /**
     * Internal field of scikit-learn object.
     */
    private boolean withStd = true;

    /**
     * Sets the Per feature relative scaling of the data to achieve zero mean and unit
     * variance. Generally this is calculated using `np.sqrt(var_)`. If a
     * variance is zero, we can't achieve unit variance, and the data is left
     * as-is, giving a scaling factor of 1. `scale_` is equal to `None` when
     * `with_std=False`.
     *
     * @param value The new value for scale.
     */
    public void setScale(NumpyArray<Double> value) {
        this.scale = value;
    }


    /**
     * Gets the Per feature relative scaling of the data to achieve zero mean and unit
     * variance. Generally this is calculated using `np.sqrt(var_)`. If a
     * variance is zero, we can't achieve unit variance, and the data is left
     * as-is, giving a scaling factor of 1. `scale_` is equal to `None` when
     * `with_std=False`.
     */
    public NumpyArray<Double> getScale() {
        return this.scale;
    }


    /**
     * Sets the The mean value for each feature in the training set. Equal to `None`
     * when `with_mean=False`.
     *
     * @param value The new value for mean.
     */
    public void setMean(NumpyArray<Double> value) {
        this.mean = value;
    }


    /**
     * Gets the The mean value for each feature in the training set. Equal to `None`
     * when `with_mean=False`.
     */
    public NumpyArray<Double> getMean() {
        return this.mean;
    }


    /**
     * Sets the The variance for each feature in the training set. Used to compute
     * `scale_`. Equal to `None` when `with_std=False`.
     *
     * @param value The new value for var.
     */
    public void setVariance(NumpyArray<Double> value) {
        this.variance = value;
    }


    /**
     * Gets the The variance for each feature in the training set. Used to compute
     * `scale_`. Equal to `None` when `with_std=False`.
     */
    public NumpyArray<Double> getVariance() {
        return this.variance;
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
     * Sets the The number of samples processed by the estimator for each feature. If
     * there are no missing samples, the `n_samples_seen` will be an integer,
     * otherwise it will be an array of dtype int. If `sample_weights` are
     * used it will be a float (if no missing data) or an array of dtype
     * float that sums the weights seen so far. Will be reset on new calls to
     * fit, but increments across `partial_fit` calls.
     *
     * @param value The new value for nSamplesSeen.
     */
    public void setNSamplesSeen(NumpyArray<Long> value) {
        this.nSamplesSeen = value;
    }


    /**
     * Gets the The number of samples processed by the estimator for each feature. If
     * there are no missing samples, the `n_samples_seen` will be an integer,
     * otherwise it will be an array of dtype int. If `sample_weights` are
     * used it will be a float (if no missing data) or an array of dtype
     * float that sums the weights seen so far. Will be reset on new calls to
     * fit, but increments across `partial_fit` calls.
     */
    public NumpyArray<Long> getNSamplesSeen() {
        return this.nSamplesSeen;
    }


    /**
     * Sets the value of WithMean
     *
     * @param value The new value for WithMean.
     */
    public void setWithMean(boolean value) {
        this.withMean = value;
    }


    /**
     * Gets the value of WithMean
     */
    public boolean getWithMean() {
        return this.withMean;
    }


    /**
     * Sets the value of WithStd
     *
     * @param value The new value for WithStd.
     */
    public void setWithStandardDeviation(boolean value) {
        this.withStd = value;
    }


    /**
     * Gets the value of WithStd
     */
    public boolean getWithStandardDeviation() {
        return this.withStd;
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

        if (withMean) {
            result = Numpy.subtract(result, mean);
        }

        if (withStd) {
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

        if (withStd) {
            result = Numpy.multiply(result, scale);
        }

        if (withMean) {
            result = Numpy.add(result, mean);
        }

        return result;
    }
}