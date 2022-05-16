package ai.sklearn4j.core.libraries.numpy;

public interface INumpyArrayWrapper {
    int[] getShape();

    Object get(int... indices);

    void set(Object value, int... index);

    boolean isFloatingPoint();

    int numberOfBits();

    NumpyArray transpose();

    NumpyArray wrapInnerSubsetArray(int... indices);

    Object getRawArray();
}
