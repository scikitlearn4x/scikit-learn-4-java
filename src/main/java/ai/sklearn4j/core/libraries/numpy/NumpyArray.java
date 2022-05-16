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


    public void set(Object value, int... indices) {
        data.set(value, indices);
    }

    public boolean isFloatingPoint() {
        return data.isFloatingPoint();
    }

    public int numberOfBytes() {
        return data.numberOfBits() / 8;
    }

    public NumpyArray<Type> transpose() {
        return data.transpose();
    }

    public boolean isSingleValueArray() {
        int count = 1;
        int[] shape = getShape();

        for (int i = 0; i < shape.length; i++) {
            count = count * shape[i];
        }

        return count == 1;
    }

    public Type getSingleValue() {
        int[] shape = getShape();
        int[] index = new int[shape.length];

        return (Type) data.get(index);
    }

    public NumpyArray<Type> wrapInnerSubsetArray(int... indices) {
        return data.wrapInnerSubsetArray(indices);
    }

    public int numberOfDimensions() {
        return getShape().length;
    }
}

