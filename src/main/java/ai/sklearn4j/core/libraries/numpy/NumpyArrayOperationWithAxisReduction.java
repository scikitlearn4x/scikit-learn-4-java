package ai.sklearn4j.core.libraries.numpy;

public abstract class NumpyArrayOperationWithAxisReduction<InputType, OutputType> {
    private NumpyArray<InputType> array;

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

    public NumpyArray<OutputType> createInstanceResultNumpyArray(int[] shape) {
        int size = array.numberOfBytes();
        boolean isFloatingPoint = array.isFloatingPoint();

        return (NumpyArray<OutputType>)NumpyArrayFactory.createArrayOfShapeAndTypeInfo(isFloatingPoint, size, shape);
    }

    public abstract Object reduceAxisValues(Object[] valuesInAxis);
}
