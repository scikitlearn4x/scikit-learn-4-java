// # ==================================================================
// Deserialize LabelEncoder
//
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
// # ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.label;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.label.LabelEncoder;

import java.util.List;


public class LabelEncoderContentLoader extends BaseScikitLearnContentLoader<LabelEncoder> {
    /**
     * Instantiate a new object of LabelEncoderContentLoader.
     */
    public LabelEncoderContentLoader() {
        super("pp_label_encoder");
    }

    /**
     * Instantiate an unloaded LabelEncoder scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected LabelEncoder createResultObject() {
        return new LabelEncoder();
    }
    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new LabelEncoderContentLoader();
    }
    /**
     * Defines the fields that are required to initialize a trained scikit-learn object.
     */
    @Override
    protected void registerSetters() {
        // Fields from the documentation
        registerListField("classes_", this::setClasses);

        // Fields from the dir() method
    }

    /**
     * Sets the Holds the label for each class.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setClasses(LabelEncoder result, List<Object> value) {
        result.setClasses(value);
    }

}