// ==================================================================
// Inference for LabelEncoder
//
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
// ==================================================================
package ai.sklearn4j.preprocessing.label;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Encode target labels with value between 0 and n_classes-1.
 * This transformer should be used to encode target values, *i.e.* `y`,
 * and not the input `X`.
 */
public class LabelEncoder extends TransformerMixin<List<Object>, NumpyArray<Long>> {
    /**
     * Instantiate a new object of LabelEncoder.
     */
    public LabelEncoder() {

    }

    /**
     * Holds the label for each class.
     */
    private List<Object> classes = null;

    /**
     * Sets the label for each class.
     * * @param value  The new value for classes.
     */
    public void setClasses(List<Object> value) {
        this.classes = value;
    }


    /**
     * Gets the label for each class.
     */
    public List<Object> getClasses() {
        return this.classes;
    }

    /**
     * Transform labels to normalized encoding.
     *
     * @param array array-like of shape (n_samples,) Target values.
     * @return array-like of shape (n_samples,) Labels as normalized encodings.
     */
    @Override
    public NumpyArray<Long> transform(List<Object> array) {
        Map<Object, Long> mapper = new HashMap<>();

        for (Object cls : classes) {
            mapper.put(cls, (long)mapper.size());
        }

        long[] result = new long[array.size()];

        int i = 0;
        for (Object obj : array) {
            if (mapper.containsKey(obj)) {
                result[i] = mapper.get(obj);
                i++;
            } else {
                throw new ScikitLearnCoreException(String.format("The class '%s' was not defined during the LabelEncoder training.", obj.toString()));
            }
        }

        return NumpyArrayFactory.from(result);
    }

    @Override
    public List<Object> inverseTransform(NumpyArray<Long> array) {
        List<Object> result = new ArrayList<>();
        long[] values = (long[]) array.getWrapper().getRawArray();

        for (int i = 0; i < values.length; i++) {
            int cls = (int) values[i];
            if (cls < 0 || cls >= classes.size()) {
                throw new ScikitLearnCoreException(String.format("The class '%d' is not in valid range.", cls));
            } else {
                result.add(classes.get(cls));
            }
        }

        return result;
    }
}