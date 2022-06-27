// ==================================================================
// Deserialize Binarizer
//
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
// ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.data;

import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.data.Binarizer;


/**
 * Binarizer object loader.
 */
public class BinarizerContentLoader extends BaseScikitLearnContentLoader<Binarizer> {
    /**
     * Instantiate a new object of BinarizerContentLoader.
     */
    public BinarizerContentLoader() {
        super("pp_binarizer");
    }

    /**
     * Instantiate an unloaded Binarizer scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected Binarizer createResultObject() {
        return new Binarizer();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new BinarizerContentLoader();
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
        registerDoubleField("threshold", this::setThreshold);
    }

    /**
     * Feature values below or equal to this are replaced by 0, above it by 1. Threshold may not be
     * less than 0 for operations on sparse matrices.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setThreshold(Binarizer result, double value) {
        result.setThreshold(value);
    }

    /**
     * Sets the Number of features seen during `fit`.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNFeaturesIn(Binarizer result, long value) {
        result.setNFeaturesIn(value);
    }

    /**
     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setFeatureNamesIn(Binarizer result, String[] value) {
        result.setFeatureNamesIn(value);
    }


}
