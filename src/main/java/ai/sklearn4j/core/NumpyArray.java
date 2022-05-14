package ai.sklearn4j.core;

public class NumpyArray<Type> {
    public static NumpyArray<Integer> from(int[] array) {
        throw new RuntimeException();
    }

    public static <ArrayType> NumpyArray<ArrayType> withShape(int[] shape) {
        return null;
    }

    public int[] getShape() {
        throw new RuntimeException();
    }

    public Type get(int... indices) {
        throw new RuntimeException();
    }

    public void applyToEachElement(INumpyArrayElementOperation<Type> operation) {
        throw new RuntimeException();
    }
}

