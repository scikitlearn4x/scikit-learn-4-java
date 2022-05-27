package ai.sklearn4j.core.packaging.loaders;

/**
 * A container for a method that sets a double value into a scikit-learn object during deserialization.
 *
 * @param <ObjectType> The type of the scikit-learn object.
 */
public interface IScikitLearnLoaderDoubleFieldSetter<ObjectType> {
    /**
     * Sets a double value into a scikit-learn object.
     *
     * @param obj   The scikit-learn object.
     * @param value The value to be set.
     */
    void setDoubleField(ObjectType obj, double value);
}
