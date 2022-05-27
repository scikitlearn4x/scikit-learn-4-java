package ai.sklearn4j.core.packaging.loaders;

/**
 * A container for a method that sets a string array value into a scikit-learn object during deserialization.
 *
 * @param <ObjectType> The type of the scikit-learn object.
 */
public interface IScikitLearnLoaderStringArrayFieldSetter<ObjectType> {
    /**
     * Sets a string array value into a scikit-learn object.
     *
     * @param obj   The scikit-learn object.
     * @param value The value to be set.
     */
    void setStringArrayField(ObjectType obj, String[] value);
}
