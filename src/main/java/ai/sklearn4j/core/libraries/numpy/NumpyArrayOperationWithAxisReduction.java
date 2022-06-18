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
     * @param keepDimensions A flag indicating to keep the same number of dimension as the input.
     * @return The numpy array that contains the result of the reduction.
     */
    public NumpyArray<OutputType> apply(NumpyArray<InputType> array, int axis, boolean keepDimensions) {
        this.array = array;
        int[] inputShape = array.getShape();
        int countInAxis = inputShape[axis];

        int[] outputShape = getOutShape(axis, inputShape, false);
        int[] outputShapeWithKeepDimensions = getOutShape(axis, inputShape, keepDimensions);

        NumpyArray<OutputType> result = createInstanceResultNumpyArray(outputShapeWithKeepDimensions);
        int[] counter = new int[outputShape.length + 1];
        counter[0] = -1;
        int[] outputCounter = new int[outputShapeWithKeepDimensions.length + 1];
        outputCounter[0] = -1;

        do {
            NumpyArray.addCounter(counter, outputShape);
            NumpyArray.addCounter(outputCounter, outputShapeWithKeepDimensions);

            int[] indexOnInput = new int[inputShape.length];
            int temp = 0;
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

            result.set(reduceAxisValues(valuesInAxis), outputCounter);
        } while (counter[counter.length - 1] == 0);

        return result;
    }

    /**
     * Calculate the shape of the output array.
     *
     * @param axis The axis to reduce on.
     * @param inputShape The shape of the input array.
     * @param keepDimensions A boolean indicating to keep the original number of dimensions.
     *
     * @return The shape of the output based on the parameters.
     */
    private int[] getOutShape(int axis, int[] inputShape, boolean keepDimensions) {
        int offset = keepDimensions ? 0 : 1;
        int[] outputShape = new int[inputShape.length - offset];

        int temp = 0;
        for (int i = 0; i < inputShape.length; i++) {
            if (i != axis) {
                outputShape[temp] = inputShape[i];
                temp++;
            } else if (keepDimensions) {
                outputShape[temp] = 1;
                temp++;
            }
        }
        return outputShape;
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
