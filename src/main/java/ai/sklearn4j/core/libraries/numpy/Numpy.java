package ai.sklearn4j.core.libraries.numpy;

import java.util.HashMap;
import java.util.Map;

public final class Numpy {
    public static <Type> NumpyArray<Integer> argmax(NumpyArray<Type> array, int axis) {
        NumpyArrayOperationWithAxisReduction<Type, Integer> operation = new NumpyArrayOperationWithAxisReduction<>() {
            @Override
            public NumpyArray<Integer> createInstanceResultNumpyArray(int[] shape) {
                return NumpyArrayFactory.arrayOfInt32WithShape(shape);
            }

            @Override
            public Object reduceAxisValues(Object[] valuesInAxis) {
                int result = 0;
                double max = (double) valuesInAxis[0];

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

        return operation.apply(array, axis);
    }

    public static NumpyArray sum(NumpyArray array, int axis) {
        INumpyReduceAxisFunction function = null;

        if (!array.isFloatingPoint() && array.numberOfBytes() == 1) {
            function = (values) -> {
                byte result = 0;

                for (Object value : values) {
                    result += (byte) value;
                }

                return result;
            };
        } else if (!array.isFloatingPoint() && array.numberOfBytes() == 2) {
            function = (values) -> {
                short result = 0;

                for (Object value : values) {
                    result += (short) value;
                }

                return result;
            };
        } else if (!array.isFloatingPoint() && array.numberOfBytes() == 4) {
            function = (values) -> {
                int result = 0;

                for (Object value : values) {
                    result += (int) value;
                }

                return result;
            };

        } else if (!array.isFloatingPoint() && array.numberOfBytes() == 8) {
            function = (values) -> {
                long result = 0;

                for (Object value : values) {
                    result += (long) value;
                }

                return result;
            };

        } else if (array.isFloatingPoint() && array.numberOfBytes() == 4) {
            function = (values) -> {
                float result = 0;

                for (Object value : values) {
                    result += (float) value;
                }

                return result;
            };
        } else if (array.isFloatingPoint() && array.numberOfBytes() == 8) {
            function = (values) -> {
                double result = 0;

                for (Object value : values) {
                    result += (double) value;
                }

                return result;
            };
        }

        INumpyReduceAxisFunction finalFunction = function;
        NumpyArrayOperationWithAxisReduction<Double, Double> operation = new NumpyArrayOperationWithAxisReduction<>() {
            @Override
            public Object reduceAxisValues(Object[] valuesInAxis) {
                return finalFunction.reduceAxisValues(valuesInAxis);
            }
        };

        return operation.apply(array, axis);
    }

    /**
     * Calculate the exponential of all elements in the input array.
     * https://numpy.org/doc/stable/reference/generated/numpy.exp.html
     *
     * @param array Input values.
     *
     * @return Output array, element-wise exponential of x.
     */
    public static NumpyArray<Double> exp(NumpyArray array) {
        NumpyArray<Double> result = NumpyUtils.createArrayOfShapeAndTypeInfo(true, 8, array.getShape());

        array.applyToEachElementAnsSaveToTarget(result, value -> Math.exp((double) value));

        return result;
    }

    /**
     * Natural logarithm, element-wise.
     *
     * The natural logarithm log is the inverse of the exponential function, so that
     * log(exp(x)) = x. The natural logarithm is logarithm in base e.
     *
     * @param array Input values.
     *
     * @return Output array, element-wise log of x.
     */
    public static NumpyArray<Double> log(NumpyArray array) {
        NumpyArray<Double> result = NumpyUtils.createArrayOfShapeAndTypeInfo(true, 8, array.getShape());

        array.applyToEachElementAnsSaveToTarget(result, value -> Math.log((double) value));

        return result;
    }
}
