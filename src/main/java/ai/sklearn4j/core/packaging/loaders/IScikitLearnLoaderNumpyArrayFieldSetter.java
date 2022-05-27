package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * A container for a method that sets a numpy array value into a scikit-learn object during deserialization.
 * @param <ObjectType> The type of the scikit-learn object.
 */
public interface IScikitLearnLoaderNumpyArrayFieldSetter<ObjectType> {
    /**
     * Sets a numpy array value into a scikit-learn object.
     * @param obj The scikit-learn object.
     * @param value The value to be set.
     */
    void setNumpyArrayField(ObjectType obj, NumpyArray value);
}
