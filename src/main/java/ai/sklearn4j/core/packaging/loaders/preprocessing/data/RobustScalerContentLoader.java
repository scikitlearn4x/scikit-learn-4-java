// ==================================================================
// Deserialize RobustScaler
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
// ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.data;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.data.RobustScaler;

import java.util.List;


/**
 * RobustScaler object loader.
 */
public class RobustScalerContentLoader extends BaseScikitLearnContentLoader<RobustScaler> {
    /**
     * Instantiate a new object of RobustScalerContentLoader.
     */
    public RobustScalerContentLoader() {
        super("pp_robust_scaler");
    }

    /**
     * Instantiate an unloaded RobustScaler scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected RobustScaler createResultObject() {
        return new RobustScaler();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new RobustScalerContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained scikit-learn object.
     */
    @Override
    protected void registerSetters() {
        // Fields from the documentation
        registerNumpyArrayField("center_", this::setCenter);
        registerNumpyArrayField("scale_", this::setScale);
        registerLongField("n_features", this::setNFeaturesIn);
        registerStringArrayField("feature_names", this::setFeatureNamesIn);

        // Fields from dir()
        registerLongField("with_scaling", this::setWithScaling);
        registerLongField("with_centering", this::setWithCentering);
        registerLongField("unit_variance", this::setUnitVariance);
        registerListField("quantile_range", this::setQuantileRange);
    }

    /**
     * Sets the value of the field `quantile_range`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setQuantileRange(RobustScaler result, List<Object> value) {
        double[] quantilesRange = new double[2];
        quantilesRange[0] = (double) value.get(0);
        quantilesRange[1] = (double) value.get(1);

        result.setQuantilesRange(quantilesRange);
    }

    /**
     * Sets the value of the field `unit_variance`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setUnitVariance(RobustScaler result, long value) {
        result.setUnitVariance(value == 1);
    }

    /**
     * Sets the value of the field `with_centering`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setWithCentering(RobustScaler result, long value) {
        result.setWithCentering(value == 1);
    }

    /**
     * Sets the value of the field `with_scaling`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setWithScaling(RobustScaler result, long value) {
        result.setWithScaling(value == 1);
    }

    /**
     * Sets the The median value for each feature in the training set.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setCenter(RobustScaler result, NumpyArray value) {
        result.setCenter(value);
    }

    /**
     * Sets the The (scaled) interquartile range for each feature in the training
     * set.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setScale(RobustScaler result, NumpyArray value) {
        result.setScale(value);
    }

    /**
     * Sets the Number of features seen during `fit`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNFeaturesIn(RobustScaler result, long value) {
        result.setNFeaturesIn(value);
    }

    /**
     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setFeatureNamesIn(RobustScaler result, String[] value) {
        result.setFeatureNamesIn(value);
    }
}