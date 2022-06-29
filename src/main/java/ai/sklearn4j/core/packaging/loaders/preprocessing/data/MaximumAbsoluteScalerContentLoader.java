// ==================================================================
// Deserialize MaxAbsScaler
//
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler
// ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.data;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.data.MaximumAbsoluteScaler;


/**
 * MaximumAbsoluteScaler object loader.
 */
public class MaximumAbsoluteScalerContentLoader extends BaseScikitLearnContentLoader<MaximumAbsoluteScaler> {
    /**
     * Instantiate a new object of MaximumAbsoluteScalerContentLoader.
     */
    public MaximumAbsoluteScalerContentLoader() {
        super("pp_max_abs_scaler");
    }

    /**
     * Instantiate an unloaded MaximumAbsoluteScaler scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected MaximumAbsoluteScaler createResultObject() {
        return new MaximumAbsoluteScaler();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new MaximumAbsoluteScalerContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained scikit-learn object.
     */
    @Override
    protected void registerSetters() {
        // Fields from the documentation
        registerNumpyArrayField("scale_", this::setScale);
        registerNumpyArrayField("max_abs_", this::setMaxAbs);
        registerLongField("n_features", this::setNFeaturesIn);
        registerStringArrayField("feature_names", this::setFeatureNamesIn);
        registerLongField("n_samples_seen_", this::setNSamplesSeen);
    }

    /**
     * Sets the Per feature relative scaling of the data.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setScale(MaximumAbsoluteScaler result, NumpyArray value) {
        result.setScale(value);
    }

    /**
     * Sets the Per feature maximum absolute value.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setMaxAbs(MaximumAbsoluteScaler result, NumpyArray value) {
        result.setMaxAbs(value);
    }

    /**
     * Sets the Number of features seen during `fit`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNFeaturesIn(MaximumAbsoluteScaler result, long value) {
        result.setNFeaturesIn(value);
    }

    /**
     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setFeatureNamesIn(MaximumAbsoluteScaler result, String[] value) {
        result.setFeatureNamesIn(value);
    }

    /**
     * Sets the The number of samples processed by the estimator. Will be reset on new
     * calls to fit, but increments across `partial_fit` calls.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNSamplesSeen(MaximumAbsoluteScaler result, long value) {
        result.setNSamplesSeen(value);
    }

}
