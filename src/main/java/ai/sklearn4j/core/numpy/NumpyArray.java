package ai.sklearn4j.core.numpy;

import ai.sklearn4j.core.numpy.wrappers.*;

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
        int[] counter = new int[shape.length + 1];
        int[] index = new int[shape.length];
        counter[0] = -1;

        do {
            addCounter(counter, shape);

            for (int i = 0; i < index.length; i++) {
                index[i] = counter[i];
            }

            data.set(operation.apply((Type) data.get(index)), index);
        } while (counter[counter.length - 1] == 0);
    }

    private void addCounter(int[] counter, int[] shape) {
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
}

