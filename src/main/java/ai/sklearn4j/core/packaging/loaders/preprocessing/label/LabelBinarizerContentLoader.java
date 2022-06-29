// ==================================================================
// Deserialize LabelBinarizer
//
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
// ==================================================================
package ai.sklearn4j.core.packaging.loaders.preprocessing.label;

import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.preprocessing.label.LabelBinarizer;

import java.util.List;


/**
 * LabelBinarizer object loader.
 */
public class LabelBinarizerContentLoader extends BaseScikitLearnContentLoader<LabelBinarizer> {
    /**
     * Instantiate a new object of LabelBinarizerContentLoader.
     */
    public LabelBinarizerContentLoader() {
        super("pp_label_binarizer");
    }

    /**
     * Instantiate an unloaded LabelBinarizer scikit-learn object.
     *
     * @return The unloaded scikit-learn object.
     */
    @Override
    protected LabelBinarizer createResultObject() {
        return new LabelBinarizer();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new LabelBinarizerContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained scikit-learn object.
     */
    @Override
    protected void registerSetters() {
        // Fields from the documentation
        registerListField("classes_", this::setClasses);
        registerStringField("y_type_", this::setYType);

        // Fields from the dir() method
        registerLongField("neg_label", this::setNegLabel);
        registerLongField("pos_label", this::setPosLabel);
    }

    /**
     * Sets the Holds the label for each class.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setClasses(LabelBinarizer result, List<Object> value) {
        result.setClasses(value);
    }

    /**
     * Sets the Represents the type of the target data as evaluated by
     * utils.multiclass.type_of_target. Possible type are 'continuous',
     * 'continuous-multioutput', 'binary', 'multiclass',
     * 'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setYType(LabelBinarizer result, String value) {
        result.setYType(value);
    }

    /**
     * Sets the neg_label field.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setNegLabel(LabelBinarizer result, long value) {
        result.setNegativeLabel(value);
    }

    /**
     * Sets the pos_label field.
     *
     * @param result The scikit-learn object to be loaded.
     * @param value  The loaded value from stream.
     */
    private void setPosLabel(LabelBinarizer result, long value) {
        result.setPositiveLabel(value);
    }
}
