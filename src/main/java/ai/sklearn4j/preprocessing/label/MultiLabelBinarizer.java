// ==================================================================
// Inference for MultiLabelBinarizer
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
// ==================================================================
package ai.sklearn4j.preprocessing.label;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.ScikitLearnFeatureNotImplementedException;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

import java.util.*;

/**
 * Transform between iterable of iterables and a multilabel format.
 * Although a list of sets or tuples is a very intuitive format for
 * multilabel data, it is unwieldy to process. This transformer converts
 * between this intuitive format and the supported multilabel format: a
 * (samples x classes)
 */

public class MultiLabelBinarizer extends TransformerMixin<List<Set<Object>>, NumpyArray<Long>> {
    /**
     * Instantiate a new object of MultiLabelBinarizer.
     */
    public MultiLabelBinarizer() {

    }

    /**
     * A copy of the `classes` parameter when provided. Otherwise it
     * corresponds to the sorted set of classes found when fitting.
     */
    private List<Object> classes = null;

    /**
     * Internal field of scikit-learn object.
     */
    private Map<String, Object> cachedDict = null;

    /**
     * Sets the A copy of the `classes` parameter when provided. Otherwise it
     * corresponds to the sorted set of classes found when fitting.
     *
     * @param value The new value for classes.
     */
    public void setClasses(List<Object> value) {
        this.classes = value;
    }


    /**
     * Gets the A copy of the `classes` parameter when provided. Otherwise it
     * corresponds to the sorted set of classes found when fitting.
     */
    public List<Object> getClasses() {
        return this.classes;
    }


    /**
     * Sets the value of CachedDict
     *
     * @param value The new value for CachedDict.
     */
    public void setCachedDict(Map<String, Object> value) {
        this.cachedDict = value;
    }


    /**
     * Gets the value of CachedDict
     */
    public Map<String, Object> getCachedDict() {
        return this.cachedDict;
    }


    @Override
    public NumpyArray<Long> transform(List<Set<Object>> array) {
        Map<Object, Long> mapper = new HashMap<>();
        for (int i = 0; i < classes.size(); i++) {
            mapper.put(classes.get(i), (long)i);
        }

        NumpyArray<Long> result = NumpyArrayFactory.arrayOfInt64WithShape(new int[] {array.size(), classes.size()});

        for (int i = 0; i < array.size(); i++) {
            Set<Object> labels = array.get(i);

            for (Object label : labels) {
                if (mapper.containsKey(label)) {
                    long index = mapper.get(label);
                    result.set(1, i, (int)index);
                } else {
                    throw new ScikitLearnCoreException(String.format("The class '%s' was not defined during the MultiLabelBinarizer training.", label.toString()));
                }
            }
        }

        return result;
    }

    @Override
    public List<Set<Object>> inverseTransform(NumpyArray<Long> array) {
        List<Set<Object>> result = new ArrayList<>();

        for (int i = 0; i < array.getShape()[0]; i++) {
            Set<Object> labels = new HashSet<>();
            result.add(labels);

            for (int j = 0; j < classes.size(); j++) {
                long value = array.get(i, j);
                if (value != 0) {
                    labels.add(classes.get(j));
                }
            }
        }

        return result;
    }
}