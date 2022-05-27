package ai.sklearn4j.core.libraries.numpy;

/**
 * Base class for all the operations that performs an aggregation and reduction on a given
 * NumpyArray.
 *
 * @param <InputType>  The element type of the input numpy array.
 * @param <OutputType> The element type of the output numpy array.
 */
public abstract class NumpyArrayOperationWithAxisReduction<InputType, OutputType> {
    /**
     * The input numpy array to perform the operation on.
     */
    private NumpyArray<InputType> array;

    /**
     * Applies the operation on the numpy array.
     *
     * @param array Input array to perform operation on.
     * @param axis  The axis that should be reduced on.
     * @return The numpy array that contains the result of the reduction.
     */
    public NumpyArray<OutputType> apply(NumpyArray<InputType> array, int axis) {
        this.array = array;
        int[] inputShape = array.getShape();
        int countInAxis = inputShape[axis];

        int[] outputShape = new int[inputShape.length - 1];

        int temp = 0;
        for (int i = 0; i < inputShape.length; i++) {
            if (i != axis) {
                outputShape[temp] = inputShape[i];
                temp++;
            }
        }


        NumpyArray<OutputType> result = createInstanceResultNumpyArray(outputShape);
        int[] counter = new int[outputShape.length + 1];
        counter[0] = -1;

        do {
            NumpyArray.addCounter(counter, outputShape);
            int[] indexOnInput = new int[inputShape.length];
            temp = 0;
            for (int i = 0; i < indexOnInput.length; i++) {
                if (i != axis) {
                    indexOnInput[i] = counter[temp];
                    temp++;
                }
            }

            Object[] valuesInAxis = new Object[countInAxis];
            for (int i = 0; i < countInAxis; i++) {
                indexOnInput[axis] = i;
                valuesInAxis[i] = array.get(indexOnInput);
            }

            result.set(reduceAxisValues(valuesInAxis), counter);
        } while (counter[counter.length - 1] == 0);

        return result;
    }

    /**
     * Instantiate the result numpy array.
     *
     * @param shape The shape of the desired array.
     * @return An empty array with the desired specifications.
     */
    public NumpyArray<OutputType> createInstanceResultNumpyArray(int[] shape) {
        int size = array.numberOfBytes();
        boolean isFloatingPoint = array.isFloatingPoint();

        return (NumpyArray<OutputType>) NumpyArrayFactory.createArrayOfShapeAndTypeInfo(isFloatingPoint, size, shape);
    }

    /**
     * The core operation that does the aggregation.
     *
     * @param valuesInAxis The elements of the specified axis.
     * @return The aggregated value.
     */
    public abstract Object reduceAxisValues(Object[] valuesInAxis);
}
