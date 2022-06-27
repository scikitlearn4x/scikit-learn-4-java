// ==================================================================
// Deserialize Normalizer
//
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
// ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.data;

import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.data.Normalizer;


/**
 * Normalizer object loader.
 */
public class NormalizerContentLoader extends BaseScikitLearnContentLoader<Normalizer> {
    /**
     * Instantiate a new object of NormalizerContentLoader.
     */
    public NormalizerContentLoader() {
        super("pp_normalizer");
    }

    /**
     * Instantiate an unloaded Normalizer scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected Normalizer createResultObject() {
        return new Normalizer();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new NormalizerContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained scikit-learn object.
     */
    @Override
    protected void registerSetters() {
        // Fields from the documentation
        registerLongField("n_features", this::setNFeaturesIn);
        registerStringArrayField("feature_names", this::setFeatureNamesIn);

        // Fields from the dir() method
        registerStringField("norm", this::setNorm);
    }

    private void setNorm(Normalizer result, String value) {
        result.setNorm(value);
    }

    /**
     * Sets the Number of features seen during `fit`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNFeaturesIn(Normalizer result, long value) {
        result.setNFeaturesIn(value);
    }

    /**
     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setFeatureNamesIn(Normalizer result, String[] value) {
        result.setFeatureNamesIn(value);
    }


}
