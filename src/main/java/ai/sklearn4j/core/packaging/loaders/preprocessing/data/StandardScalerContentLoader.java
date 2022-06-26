// ==================================================================
// Deserialize StandardScaler
//
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
// ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.data;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.data.StandardScaler;


/**
 * StandardScaler object loader.
 */

public class StandardScalerContentLoader extends BaseScikitLearnContentLoader<StandardScaler> {
    /**
     * Instantiate a new object of StandardScalerContentLoader.
     */
    public StandardScalerContentLoader() {
        super("pp_standard_scaler");
    }

    /**
     * Instantiate an unloaded StandardScaler scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected StandardScaler createResultObject() {
        return new StandardScaler();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new StandardScalerContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained scikit-learn object.
     */
    @Override
    protected void registerSetters() {
        // Fields from the documentation
        registerNumpyArrayField("scale_", this::setScale);
        registerNumpyArrayField("mean_", this::setMean);
        registerNumpyArrayField("var_", this::setVar);
        registerLongField("n_features", this::setNFeaturesIn);
        registerStringArrayField("feature_names", this::setFeatureNamesIn);
        registerNumpyArrayField("n_samples_seen_", this::setNSamplesSeen);

        // Fields from the dir() method
        registerLongField("with_mean", this::setWithMean);
        registerLongField("with_std", this::setWithStd);
    }

    /**
     * Sets the Per feature relative scaling of the data to achieve zero mean and unit
     * variance. Generally this is calculated using `np.sqrt(var_)`. If a
     * variance is zero, we can't achieve unit variance, and the data is left
     * as-is, giving a scaling factor of 1. `scale_` is equal to `None` when
     * `with_std=False`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setScale(StandardScaler result, NumpyArray value) {
        result.setScale(value);
    }

    /**
     * Sets the The mean value for each feature in the training set. Equal to `None`
     * when `with_mean=False`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setMean(StandardScaler result, NumpyArray value) {
        result.setMean(value);
    }

    /**
     * Sets the The variance for each feature in the training set. Used to compute
     * `scale_`. Equal to `None` when `with_std=False`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setVar(StandardScaler result, NumpyArray value) {
        result.setVariance(value);
    }

    /**
     * Sets the Number of features seen during `fit`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNFeaturesIn(StandardScaler result, long value) {
        result.setNFeaturesIn(value);
    }

    /**
     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setFeatureNamesIn(StandardScaler result, String[] value) {
        result.setFeatureNamesIn(value);
    }

    /**
     * Sets the The number of samples processed by the estimator for each feature. If
     * there are no missing samples, the `n_samples_seen` will be an integer,
     * otherwise it will be an array of dtype int. If `sample_weights` are
     * used it will be a float (if no missing data) or an array of dtype
     * float that sums the weights seen so far. Will be reset on new calls to
     * fit, but increments across `partial_fit` calls.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNSamplesSeen(StandardScaler result, NumpyArray value) {
        result.setNSamplesSeen(value);
    }

    /**
     * Sets the with_mean field.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setWithMean(StandardScaler result, long value) {
        result.setWithMean(value == 1);
    }

    /**
     * Sets the with_std field.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setWithStd(StandardScaler result, long value) {
        result.setWithStandardDeviation(value == 1);
    }

}

