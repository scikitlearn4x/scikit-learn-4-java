package ai.sklearn4j.core.libraries.numpy;

/**
 * Provide the functionality of the Numpy arrays and an abstraction over the element types.
 *
 * @param <Type> Type of the elements of the array.
 */
public class NumpyArray<Type> {
    /**
     * The internal data of the array.
     */
    private INumpyArrayWrapper data = null;

    /**
     * Instantiate a new object of NumpyArray.
     *
     * @param data The content of the numpy array.
     */
    protected NumpyArray(INumpyArrayWrapper data) {
        this.data = data;
    }

    /**
     * Gets the shape of the numpy array.
     *
     * @return The array shape.
     */
    public int[] getShape() {
        return data.getShape();
    }

    /**
     * Gets an element in the array specified by it index.
     *
     * @param indices The index of the element to be retrieved.
     * @return The element value in the array specified by its index.
     */
    public Type get(int... indices) {
        if (indices.length != getShape().length) {
            throw new NumpyOperationException("The number of indices provided doesn't match the number of dimensions.");
        }

        return (Type) data.get(indices);
    }

    /**
     * Applies a provided operation on every element of the array and stores it in the
     * same array.
     *
     * @param operation The operation to be applied on the elements.
     */
    public void applyToEachElement(INumpyArrayElementOperation<Type> operation) {
        int[] shape = data.getShape();
        int[] index = new int[shape.length];
        int[] counter = new int[shape.length + 1];
        counter[0] = -1;

        do {
            addCounter(counter, shape);

            for (int i = 0; i < index.length; i++) {
                index[i] = counter[i];
            }

            data.set(operation.apply((Type) data.get(index)), index);
        } while (counter[counter.length - 1] == 0);
    }

    /**
     * Applies a provided operation on every element of the array and stores it in a
     * specified target array.
     *
     * @param target    The array that the result should be stored into.
     * @param operation The operation to be applied on the elements.
     */
    public void applyToEachElementAnsSaveToTarget(NumpyArray target, INumpyArrayElementOperation<Type> operation) {
        int[] shape = data.getShape();
        int[] index = new int[shape.length];
        int[] counter = new int[shape.length + 1];
        counter[0] = -1;

        do {
            addCounter(counter, shape);

            for (int i = 0; i < index.length; i++) {
                index[i] = counter[i];
            }

            target.set(operation.apply((Type) data.get(index)), index);
        } while (counter[counter.length - 1] == 0);
    }

    /**
     * The numpy array could be multidimensional. To iterate over the element given that the
     * dimension is dynamic, a counter is used. This method increase the value of the counter
     * to move the index to the next element.
     *
     * @param counter The current index.
     * @param shape   The shape of the array.
     */
    public static void addCounter(int[] counter, int[] shape) {
        counter[0]++;

        for (int i = 0; i < shape.length; i++) {
            if (counter[i] == shape[i]) {
                counter[i] = 0;
                counter[i + 1]++;
            } else {
                break;
            }
        }
    }

    /**
     * Sets an element in the array specified by it index.
     *
     * @param value The value to be set.
     * @param indices The index of the element to be modified.
     */
    public void set(Object value, int... indices) {
        data.set(value, indices);
    }

    /**
     * Returns a boolean indicating if the underlying data type is floating numbers or not.
     *
     * @return A boolean indicating if the underlying data type is floating numbers or not.
     */
    public boolean isFloatingPoint() {
        return data.isFloatingPoint();
    }

    /**
     * Gets the number of bytes allocated in memory for each element.
     *
     * @return Number of bytes allocated in memory for each element.
     */
    public int numberOfBytes() {
        return data.numberOfBits() / 8;
    }

    /**
     * Transposes an array by reversing the order of its dimensions.
     *
     * @return The transposed array.
     */
    public NumpyArray<Type> transpose() {
        return data.transpose();
    }

    /**
     * Returns a boolean indicating if the array contains only a single element.
     *
     * @return A boolean indicating if the array contains only a single element.
     */
    public boolean isSingleValueArray() {
        int count = 1;
        int[] shape = getShape();

        for (int i = 0; i < shape.length; i++) {
            count = count * shape[i];
        }

        return count == 1;
    }

    /**
     * Gets the first element of the numpy array from its memory layout.
     *
     * @return The first element of the array.
     */
    public Type getSingleValue() {
        int[] shape = getShape();
        int[] index = new int[shape.length];

        return (Type) data.get(index);
    }

    /**
     * Wraps a subset of the numpy array. This methods works only when slicing the most inner dimensions
     * of the array. For example, if the shape is [2, 6, 4, 8], wrapInnerSubsetArray(1, 3) is equivalent
     * to numpy array[1, 3, :, :].
     *
     * @param indices The indices of the first dimensions to keep.
     * @return A new INumpyArrayWrapper wrapping the inner dimensions. This array has the same reference.
     */
    public NumpyArray<Type> wrapInnerSubsetArray(int... indices) {
        return data.wrapInnerSubsetArray(indices);
    }

    /**
     * Gets the number of dimensions of the numpy array.
     *
     * @return
     */
    public int numberOfDimensions() {
        return getShape().length;
    }

    /**
     * Gets the object that wraps the underlying array.
     *
     * @return The object that wraps the underlying array.
     */
    public INumpyArrayWrapper getWrapper() {
        return data;
    }
}

