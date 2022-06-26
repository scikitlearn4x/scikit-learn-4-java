package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;

import java.util.Map;

/**
 * A container for a method that sets a dictionary value into a scikit-learn object during deserialization.
 *
 * @param <ObjectType> The type of the scikit-learn object.
 */
public interface IScikitLearnLoaderDictionaryFieldSetter<ObjectType> {
    /**
     * Sets a dictionary value into a scikit-learn object.
     *
     * @param obj   The scikit-learn object.
     * @param value The value to be set.
     */
    void setDictionaryField(ObjectType obj, Map<String, Object> value);
}
