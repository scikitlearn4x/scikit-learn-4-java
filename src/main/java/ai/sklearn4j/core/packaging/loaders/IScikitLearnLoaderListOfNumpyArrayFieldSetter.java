package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;

import java.util.List;

/**
 * A container for a method that sets a list of numpy array value into a scikit-learn object during deserialization.
 *
 * @param <ObjectType> The type of the scikit-learn object.
 * @param <ArrayType> The type of the numpy array's element.
 */
public interface IScikitLearnLoaderListOfNumpyArrayFieldSetter<ObjectType, ArrayType> {
    /**
     * Sets a numpy array value into a scikit-learn object.
     *
     * @param obj   The scikit-learn object.
     * @param value The value to be set.
     */
    void setListOfNumpyArrayField(ObjectType obj, List<NumpyArray<ArrayType>> value);
}
