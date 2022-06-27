// ==================================================================
// Deserialize MinMaxScaler
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
// ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.data;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.data.MinimumMaximumScaler;

import java.util.List;


/**
 * MinimumMaximumScaler object loader.
 */
public class MinimumMaximumScalerContentLoader extends BaseScikitLearnContentLoader<MinimumMaximumScaler> {
    /**
     * Instantiate a new object of MinimumMaximumScalerContentLoader.
     */
    public MinimumMaximumScalerContentLoader() {
        super("pp_min_max_scaler");
    }

    /**
     * Instantiate an unloaded MinimumMaximumScaler scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected MinimumMaximumScaler createResultObject() {
        return new MinimumMaximumScaler();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new MinimumMaximumScalerContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained scikit-learn object.
     */
    @Override
    protected void registerSetters() {
        // Fields from the documentation
        registerNumpyArrayField("min_", this::setMin);
        registerNumpyArrayField("scale_", this::setScale);
        registerNumpyArrayField("data_min_", this::setDataMin);
        registerNumpyArrayField("data_max_", this::setDataMax);
        registerNumpyArrayField("data_range_", this::setDataRange);
        registerLongField("n_features", this::setNFeaturesIn);
        registerLongField("n_samples_seen_", this::setNSamplesSeen);
        registerStringArrayField("feature_names", this::setFeatureNamesIn);

        // Fields from the dir() method
        registerLongField("clip", this::setClip);
        registerListField("feature_range", this::setFeatureRange);
    }

    /**
     * Sets the Per feature adjustment for minimum. Equivalent to `min - X.min(axis=0)
     * * self.scale_`
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setMin(MinimumMaximumScaler result, NumpyArray value) {
        result.setMin(value);
    }

    /**
     * Sets the Per feature relative scaling of the data. Equivalent to `(max - min) /
     * (X.max(axis=0) - X.min(axis=0))`
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setScale(MinimumMaximumScaler result, NumpyArray value) {
        result.setScale(value);
    }

    /**
     * Sets the Per feature minimum seen in the data
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setDataMin(MinimumMaximumScaler result, NumpyArray value) {
        result.setDataMin(value);
    }

    /**
     * Sets the Per feature maximum seen in the data
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setDataMax(MinimumMaximumScaler result, NumpyArray value) {
        result.setDataMax(value);
    }

    /**
     * Sets the Per feature range `(data_max_ - data_min_)` seen in the data
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setDataRange(MinimumMaximumScaler result, NumpyArray value) {
        result.setDataRange(value);
    }

    /**
     * Sets the Number of features seen during `fit`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNFeaturesIn(MinimumMaximumScaler result, Object value) {
        result.setNFeaturesIn(value);
    }

    /**
     * Sets the The number of samples processed by the estimator. It will be reset on
     * new calls to fit, but increments across `partial_fit` calls.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNSamplesSeen(MinimumMaximumScaler result, long value) {
        result.setNSamplesSeen(value);
    }

    /**
     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setFeatureNamesIn(MinimumMaximumScaler result, Object value) {
        result.setFeatureNamesIn(value);
    }

    /**
     * Sets the clip field.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setClip(MinimumMaximumScaler result, long value) {
        result.setClip(value == 1);
    }

    /**
     * Sets the feature_range field.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setFeatureRange(MinimumMaximumScaler result, List<Object> value) {
        double[] data = new double[value.size()];
        for (int i = 0; i < data.length; i++) {
            data[i] = Double.valueOf(value.get(i).toString()); // WTF java?!
        }
        result.setFeatureRange(data);
    }

}