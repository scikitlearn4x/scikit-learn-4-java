// ==================================================================
// Inference for LabelBinarizer
//
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
// ==================================================================
package ai.sklearn4j.preprocessing.label;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.ScikitLearnFeatureNotImplementedException;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Binarize labels in a one-vs-all fashion.
 * Several regression and binary classification algorithms are available
 * in scikit-learn. A simple way to extend these algorithms to the
 * multi-class classification case is to use the so-called one-vs-all
 * scheme.
 * At learning time, this simply consists in learning one regressor or
 * binary classifier per class. In doing so, one needs to convert
 * multi-class labels to binary labels (belong or does not belong to the
 * class). LabelBinarizer makes this process easy with the transform
 * method.
 * At prediction time, one assigns the class for which the corresponding
 * model gave the greatest confidence. LabelBinarizer makes this easy
 * with the inverse_transform method.
 */

public class LabelBinarizer extends TransformerMixin<List<Object>, NumpyArray<Long>> {
    private static final String Y_TYPE_BINARY = "binary";
    private static final String Y_TYPE_MULTICLASS = "multiclass";

    /**
     * Instantiate a new object of LabelBinarizer.
     */
    public LabelBinarizer() {

    }

    /**
     * Holds the label for each class.
     */
    private List<Object> classes = null;

    /**
     * Represents the type of the target data as evaluated by
     * utils.multiclass.type_of_target. Possible type are 'continuous',
     * 'continuous-multioutput', 'binary', 'multiclass',
     * 'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.
     */
    private String yType = null;

    /**
     * Internal field of scikit-learn object.
     */
    private long negLabel = 0;

    /**
     * Internal field of scikit-learn object.
     */
    private long posLabel = 1;

    /**
     * Sets the Holds the label for each class.
     *
     * @param value The new value for classes.
     */
    public void setClasses(List<Object> value) {
        this.classes = value;
    }


    /**
     * Gets the Holds the label for each class.
     */
    public List<Object> getClasses() {
        return this.classes;
    }


    /**
     * Sets the Represents the type of the target data as evaluated by
     * utils.multiclass.type_of_target. Possible type are 'continuous',
     * 'continuous-multioutput', 'binary', 'multiclass',
     * 'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.
     *
     * @param value The new value for yType.
     */
    public void setYType(String value) {
        this.yType = value;
    }


    /**
     * Gets the Represents the type of the target data as evaluated by
     * utils.multiclass.type_of_target. Possible type are 'continuous',
     * 'continuous-multioutput', 'binary', 'multiclass',
     * 'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.
     */
    public String getYType() {
        return this.yType;
    }


    /**
     * Sets the value of NegLabel
     *
     * @param value The new value for NegLabel.
     */
    public void setNegLabel(long value) {
        this.negLabel = value;
    }


    /**
     * Gets the value of NegLabel
     */
    public long getNegLabel() {
        return this.negLabel;
    }


    /**
     * Sets the value of PosLabel
     *
     * @param value The new value for PosLabel.
     */
    public void setPosLabel(long value) {
        this.posLabel = value;
    }


    /**
     * Gets the value of PosLabel
     */
    public long getPosLabel() {
        return this.posLabel;
    }

    @Override
    public NumpyArray<Long> transform(List<Object> array) {
        if (yType.equals(Y_TYPE_BINARY)) {
            return transformBinary(array);
        } else if (yType.equals(Y_TYPE_MULTICLASS)) {
            return transformMulticlass(array);
        } else {
            throw new ScikitLearnFeatureNotImplementedException(String.format("The yType=%s in LabelBinarizer is not implemented.", yType));
        }
    }

    private NumpyArray<Long> transformBinary(List<Object> array) {
        Map<Object, Integer> mapper = new HashMap<>();

        for (Object cls : classes) {
            mapper.put(cls, mapper.size());
        }

        NumpyArray<Long> result = NumpyArrayFactory.arrayOfInt64WithShape(new int[]{array.size(), 1});

        int i = 0;
        for (Object obj : array) {
            if (mapper.containsKey(obj)) {
                int index = mapper.get(obj);
                result.set(index == 0 ? negLabel : posLabel, i, 0);
                i++;
            } else {
                throw new ScikitLearnCoreException(String.format("The class '%s' was not defined during the LabelEncoder training.", obj.toString()));
            }
        }

        return result;
    }

    private NumpyArray<Long> transformMulticlass(List<Object> array) {
        Map<Object, Integer> mapper = new HashMap<>();

        for (Object cls : classes) {
            mapper.put(cls, mapper.size());
        }

        NumpyArray<Long> result = NumpyArrayFactory.arrayOfInt64WithShape(new int[]{array.size(), classes.size()});

        int i = 0;
        int classCount = classes.size();
        for (Object obj : array) {
            if (mapper.containsKey(obj)) {
                int index = mapper.get(obj);
                for (int j = 0; j < classCount; j++) {
                    if (j == index) {
                        result.set(posLabel, i, j);
                    } else {
                        result.set(negLabel, i, j);
                    }
                }
                i++;
            } else {
                throw new ScikitLearnCoreException(String.format("The class '%s' was not defined during the LabelEncoder training.", obj.toString()));
            }
        }

        return result;
    }

    @Override
    public List<Object> inverseTransform(NumpyArray<Long> array) {
        if (yType.equals(Y_TYPE_BINARY)) {
            return inverseTransformBinary(array);
        } else if (yType.equals(Y_TYPE_MULTICLASS)) {
            return inverseTransformMulticlass(array);
        } else {
            throw new ScikitLearnFeatureNotImplementedException(String.format("The yType=%s in LabelBinarizer is not implemented.", yType));
        }

    }

    private List<Object> inverseTransformBinary(NumpyArray<Long> array) {
        List<Object> result = new ArrayList<>();
        long[][] values = (long[][]) array.getWrapper().getRawArray();

        for (int i = 0; i < values.length; i++) {
            int cls = (int) values[i][0];
            result.add(cls == negLabel ? classes.get(0) : classes.get(1));
        }

        return result;
    }

    private List<Object> inverseTransformMulticlass(NumpyArray<Long> array) {
        List<Object> result = new ArrayList<>();
        long[][] values = (long[][]) array.getWrapper().getRawArray();

        for (int i = 0; i < values.length; i++) {
            int cls = getPosLabelIndex(values[i]);
            if (cls < 0 || cls >= classes.size()) {
                throw new ScikitLearnCoreException(String.format("The class '%d' is not in valid range.", cls));
            } else {
                result.add(classes.get(cls));
            }
        }

        return result;
    }

    private int getPosLabelIndex(long[] value) {
        int result = -1;

        for (int i = 0; i < value.length; i++) {
            if (value[i] == posLabel) {
                result = i;
                break;
            }
        }

        return result;
    }
}