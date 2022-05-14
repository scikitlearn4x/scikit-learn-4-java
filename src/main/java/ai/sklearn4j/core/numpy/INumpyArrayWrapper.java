package ai.sklearn4j.core.numpy;

public interface INumpyArrayWrapper {
    int[] getShape();

    Object get(int... indices);

    void set(Object value, int... index);
}
