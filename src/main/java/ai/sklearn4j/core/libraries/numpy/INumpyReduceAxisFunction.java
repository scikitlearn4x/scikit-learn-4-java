package ai.sklearn4j.core.libraries.numpy;

/**
 * An interface to provide a unified view of the operations that can be performed on a
 * specified dimension of a NumpyArray.
 */
public interface INumpyReduceAxisFunction {
    /**
     * The aggregation function that can reduce the values of a NumpyArray dimension.
     *
     * @param valuesInAxis The values in the axis.
     *
     * @return A single value that contains the reduction of the axis.
     */
    Object reduceAxisValues(Object[] valuesInAxis);
}
