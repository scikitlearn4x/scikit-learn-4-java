package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;

import java.util.List;

/**
 * A container for a method that sets a list value into a scikit-learn object during deserialization.
 *
 * @param <ObjectType> The type of the scikit-learn object.
 */
public interface IScikitLearnLoaderListFieldSetter<ObjectType> {
    /**
     * Sets a value into a scikit-learn object.
     *
     * @param obj   The scikit-learn object.
     * @param value The value to be set.
     */
    void setListField(ObjectType obj, List<Object> value);
}
