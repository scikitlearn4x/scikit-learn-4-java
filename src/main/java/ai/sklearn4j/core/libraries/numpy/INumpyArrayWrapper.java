package ai.sklearn4j.core.libraries.numpy;

/**
 * Unlike python, Java is a strongly typed language and all the fields should have a predefined type.
 * This fact complicates the handling of data structures like Numpy arrays where the same interface
 * is used for all data types. The INumpyArrayWrapper provide a unified view of the underlying data
 * types.
 */
public interface INumpyArrayWrapper {
    /**
     * Gets the shape of the underlying Numpy array.
     *
     * @return The shape of the array as an int[].
     */
    int[] getShape();

    /**
     * Gets the value of a single element in the numpy array denoted by its indices.
     *
     * @param indices The index of the element to be retrieved.
     * @return An object-wrapped instance of the element in the Numpy array.
     */
    Object get(int... indices);

    /**
     * Sets the value of a single element in the numpy array denoted by its indices.
     *
     * @param value The new value to be assigned to the numpy element.
     * @param index The index of the element to be set.
     */
    void set(Object value, int... index);

    /**
     * Returns a boolean indicating that the underlying array type is a floating point one or not.
     *
     * @return A boolean value indicating floating point of the elements.
     */
    boolean isFloatingPoint();

    /**
     * The space allocated by each element of the numpy array in terms of bits.
     *
     * @return Number of bit needed for each element.
     */
    int numberOfBits();

    /**
     * Transposes a numpy array by reversing its dimensions.
     *
     * @return The transposed numpy array.
     */
    NumpyArray transpose();

    /**
     * Wraps a subset of the numpy array. This methods works only when slicing the most inner dimensions
     * of the array. For example, if the shape is [2, 6, 4, 8], wrapInnerSubsetArray(1, 3) is equivalent
     * to numpy array[1, 3, :, :].
     *
     * @param indices The indices of the first dimensions to keep.
     * @return A new INumpyArrayWrapper wrapping the inner dimensions. This array has the same reference.
     */
    NumpyArray wrapInnerSubsetArray(int... indices);

    /**
     * Gets the native array containing the numpy array data.
     *
     * @return The underlying native array.
     */
    Object getRawArray();
}
