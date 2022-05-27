package ai.sklearn4j.core.libraries.numpy;

/**
 * An interface to provide a unified view of the operations that can be performed on the
 * elements of a NumpyArray.
 *
 * @param <Type> The type of the NumpyArray.
 */
public interface INumpyArrayElementOperation<Type> {
    /**
     * The operation to be applied to each element in the NumpyArray.
     *
     * @param value The value of the element.
     *
     * @return The calculation on the element.
     */
    Type apply(Type value);
}
