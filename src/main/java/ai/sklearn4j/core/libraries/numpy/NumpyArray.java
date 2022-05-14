package ai.sklearn4j.core.libraries.numpy;

import ai.sklearn4j.core.libraries.numpy.wrappers.*;

public class NumpyArray<Type> {
    private INumpyArrayWrapper data = null;

    protected NumpyArray(INumpyArrayWrapper data) {
        this.data = data;
    }

    public static <ArrayType> NumpyArray<ArrayType> withShape(int[] shape) {
        return null;
    }

    public int[] getShape() {
        return data.getShape();
    }

    public Type get(int... indices) {
        if (indices.length != getShape().length) {
            throw new RuntimeException();
        }

        return (Type) data.get(indices);
    }

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

    protected static void addCounter(int[] counter, int[] shape) {
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

    public NumpyArray<Integer> argmax(int axis) {
        NumpyArrayOperationWithAxisReduction<Type, Integer> operation = new NumpyArrayOperationWithAxisReduction<>() {
            @Override
            public NumpyArray<Integer> createInstanceResultNumpyArray(int[] shape) {
                return NumpyArrayFactory.arrayOfInt32WithShape(shape);
            }

            @Override
            public Object reduceAxisValues(Object[] valuesInAxis) {
                int result = 0;
                double max = (double)valuesInAxis[0];

                for (int i = 1; i < valuesInAxis.length; i++) {
                    double m = (double) valuesInAxis[i];
                    if (m > max) {
                        max = m;
                        result = i;
                    }
                }

                return result;
            }
        };
        
        return operation.apply(this, axis);
    }

    public void set(Object value, int... indices) {
        data.set(value, indices);
    }

    public NumpyArray<Double> sum(int axis) {
        NumpyArrayOperationWithAxisReduction<Type, Double> operation = new NumpyArrayOperationWithAxisReduction<>() {
            @Override
            public NumpyArray<Double> createInstanceResultNumpyArray(int[] shape) {
                return NumpyArrayFactory.arrayOfDoubleWithShape(shape);
            }

            @Override
            public Object reduceAxisValues(Object[] valuesInAxis) {
                double result = 0.0;

                for (int i = 0; i < valuesInAxis.length; i++) {
                    result += (double) valuesInAxis[i];
                }

                return result;
            }
        };

        return operation.apply(this, axis);
    }
}

abstract class NumpyArrayOperationWithAxisReduction<InputType, OutputType> {
    public NumpyArray<OutputType> apply(NumpyArray<InputType> array, int axis) {
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
            temp= 0;
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

    public abstract NumpyArray<OutputType> createInstanceResultNumpyArray(int[] shape);

    public abstract Object reduceAxisValues(Object[] valuesInAxis);
}

