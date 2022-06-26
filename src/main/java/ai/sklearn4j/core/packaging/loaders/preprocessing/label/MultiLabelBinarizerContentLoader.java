// ==================================================================
// Deserialize MultiLabelBinarizer
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
// ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.label;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.label.MultiLabelBinarizer;

import java.util.List;
import java.util.Map;


/**
 * MultiLabelBinarizer object loader.
 */

public class MultiLabelBinarizerContentLoader extends BaseScikitLearnContentLoader<MultiLabelBinarizer> {
    /**
     * Instantiate a new object of MultiLabelBinarizerContentLoader.
     */
    public MultiLabelBinarizerContentLoader() {
        super("pp_multilabel_binarizer");
    }

    /**
     * Instantiate an unloaded MultiLabelBinarizer scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected MultiLabelBinarizer createResultObject() {
        return new MultiLabelBinarizer();
    }
    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new MultiLabelBinarizerContentLoader();
    }
    /**
     * Defines the fields that are required to initialize a trained scikit-learn object.
     */
    @Override
    protected void registerSetters() {
        // Fields from the documentation
        registerListField("classes_", this::setClasses);

        // Fields from the dir() method
        registerDictionaryField("_cached_dict", this::setCachedDict);
    }

    /**
     * Sets the A copy of the `classes` parameter when provided. Otherwise it
     * corresponds to the sorted set of classes found when fitting.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setClasses(MultiLabelBinarizer result, List<Object> value) {
        result.setClasses(value);
    }

    /**
     * Sets the _cached_dict field.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setCachedDict(MultiLabelBinarizer result, Map<String, Object> value) {
        result.setCachedDict(value);
    }
}