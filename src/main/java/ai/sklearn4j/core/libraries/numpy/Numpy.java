package ai.sklearn4j.core.libraries.numpy;

import java.util.HashMap;
import java.util.Map;

/**
 * Implementation of the Numpy library APIs.
 */
public final class Numpy {
    /**
     * Returns the indices of the maximum values along an axis.
     * See: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
     *
     * @param array The input multidimensional array.
     * @param axis The axis which the argmax should reduce to.
     *
     * @return Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.
     */
    public static <Type> NumpyArray<Long> argmax(NumpyArray<Type> array, int axis) {
        NumpyArrayOperationWithAxisReduction<Type, Long> operation = new NumpyArrayOperationWithAxisReduction<>() {
            @Override
            public NumpyArray<Long> createInstanceResultNumpyArray(int[] shape) {
                return NumpyArrayFactory.arrayOfInt64WithShape(shape);
            }

            @Override
            public Object reduceAxisValues(Object[] valuesInAxis) {
                long result = 0;
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

    /**
     * Performs an element-wise power operation on a given NumpyArray.
     * See: https://numpy.org/doc/stable/reference/generated/numpy.power.html
     *
     * @param array Input array.
     * @param power The value of the power.
     *
     * @return An array with same dimension with the requested power calculation.
     */
    public static NumpyArray<Double> pow(NumpyArray array, double power) {
        NumpyArray<Double> result = NumpyArrayFactory.arrayOfDoubleWithShape(array.getShape());

        array.applyToEachElementAnsSaveToTarget(result, value -> Math.pow((double) value, power));

        return result;
    }

    /**
     * Sums the values of a NumpyArray along a specified axis.
     * See: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
     *
     * @param array Input array.
     * @param axis Axis along which a sum is performed.
     *
     * @return An array with the same shape as a, with the specified axis removed.
     */
    public static NumpyArray sum(NumpyArray array, int axis) {
        INumpyReduceAxisFunction function = null;

        if (!array.isFloatingPoint() && array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
            function = (values) -> {
                byte result = 0;

                for (Object value : values) {
                    result += (byte) value;
                }

                return result;
            };
        } else if (!array.isFloatingPoint() && array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
            function = (values) -> {
                short result = 0;

                for (Object value : values) {
                    result += (short) value;
                }

                return result;
            };
        } else if (!array.isFloatingPoint() && array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
            function = (values) -> {
                int result = 0;

                for (Object value : values) {
                    result += (int) value;
                }

                return result;
            };

        } else if (!array.isFloatingPoint() && array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
            function = (values) -> {
                long result = 0;

                for (Object value : values) {
                    result += (long) value;
                }

                return result;
            };

        } else if (array.isFloatingPoint() && array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
            function = (values) -> {
                float result = 0;

                for (Object value : values) {
                    result += (float) value;
                }

                return result;
            };
        } else if (array.isFloatingPoint() && array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
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
     * @return Output array, element-wise exponential of x.
     */
    public static NumpyArray<Double> exp(NumpyArray array) {
        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(true, NumpyArrayFactory.SIZE_OF_DOUBLE, array.getShape());

        array.applyToEachElementAnsSaveToTarget(result, value -> Math.exp((double) value));

        return result;
    }

    /**
     * Natural logarithm, element-wise.
     * <p>
     * The natural logarithm log is the inverse of the exponential function, so that
     * log(exp(x)) = x. The natural logarithm is logarithm in base e.
     *
     * @param array Input values.
     * @return Output array, element-wise log of x.
     */
    public static NumpyArray<Double> log(NumpyArray array) {
        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(true, NumpyArrayFactory.SIZE_OF_DOUBLE, array.getShape());

        array.applyToEachElementAnsSaveToTarget(result, value -> Math.log((double) value));

        return result;
    }

    /**
     * To ease the add implementation, the code assumes that the array with higher dimensions is
     * on the left-hand side. In case it is not provided in this format, it should be swapped. This
     * function checks if it is the case or not.
     *
     * @param a1 Array on the left hand-side of the addition.
     * @param a2 Array on the right hand-side of the addition.
     * @return A boolean indicating if the values should be swapped or not.
     */
    private static boolean shouldSwapForAdd(NumpyArray a1, NumpyArray a2) {
        boolean result = false;
        int[] s1 = a1.getShape();
        int[] s2 = a2.getShape();

        if (getEffectiveShapeWithRemovingEndingDimensions(s1) < getEffectiveShapeWithRemovingEndingDimensions(s2)) {
            result = true;
        }

        return result;
    }

    /**
     * Gets the effective shape of the array. The effective shape is defined as the number of dimensions
     * on the left that is followed only by ones.
     * <p>
     * Example:
     * (2, 2) -> (2, 2): Will return 2
     * (1, 4) -> (1, 4): Will return 2
     * (3, 5, 1, 1) -> (3, 5): Will return 2
     *
     * @param shape Shape to evaluate the effective dimensions.
     * @return Number of effective dimensions.
     */
    private static int getEffectiveShapeWithRemovingEndingDimensions(int[] shape) {
        int lastOnes = 0;

        for (int i = 0; i < shape.length; i++) {
            if (shape[shape.length - i - 1] == 1) {
                lastOnes++;
            } else {
                break;
            }
        }
        return shape.length - lastOnes;
    }

    /**
     * Checks if two numpy arrays with the given dimensions could be added. If they are incompatible,
     * an exception is thrown.
     *
     * @param shape1 Shape of the array on the left hand-side.
     * @param shape2 Shape of the array on the right hand-side.
     */
    private static void validateDimensionsForAdd(int[] shape1, int[] shape2) {
        int effective1 = getEffectiveShapeWithRemovingEndingDimensions(shape1);
        int effective2 = getEffectiveShapeWithRemovingEndingDimensions(shape2);

        if (effective1 != effective2 && Math.abs(effective2 - effective1) != 1) {
            throw new NumpyOperationException("The effective shape of the two numpy array has different number of dimensions.");
        }

        for (int i = 0; i < effective1; i++) {
            if (shape1[i] != shape2[i] && (shape1[i] != 1 && shape2[i] != 1)) {
                throw new NumpyOperationException(String.format("Dimension %d of the two numpy arrays doesn't match.", i + 1));
            }
        }
    }

    /**
     * Subtract two numpy arrays.
     *
     * @param a1 Left-hand side of the expression.
     * @param a2 Right-hand side of the expression.
     *
     * @return The subtraction result.
     */
    public static NumpyArray subtract(NumpyArray a1, NumpyArray a2) {
        INumpyArrayElementOperation negate = null;
        if (a2.isFloatingPoint()) {
            if (a2.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                negate = value -> -((double) value);
            } else if (a2.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                negate = value -> -((float) value);
            }
        } else {
            if (a2.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                negate = value -> -((byte) value);
            } else if (a2.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                negate = value -> -((short) value);
            } else if (a2.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                negate = value -> -((int) value);
            } else if (a2.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                negate = value -> -((long) value);
            }
        }


        NumpyArray negA2 = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(a2);
        INumpyArrayElementOperation finalNegate = negate;
        a2.applyToEachElementAnsSaveToTarget(negA2, value -> finalNegate.apply(value));

        return add(a1, negA2);
    }

    /**
     * Adds two numpy arrays.
     *
     * @param a1 Left-hand side of the expression.
     * @param a2 Right-hand side of the expression.
     *
     * @return The addition result.
     */
    public static NumpyArray add(NumpyArray a1, NumpyArray a2) {
        validateDimensionsForAdd(a1.getShape(), a2.getShape());
        if (shouldSwapForAdd(a1, a2)) {
            return add(a2, a1);
        }

        boolean isFloatingPoint = a1.isFloatingPoint() || a2.isFloatingPoint();
        int size = Math.max(a1.numberOfBytes(), a2.numberOfBytes());

        if (!a1.isFloatingPoint()) {
            size = a2.numberOfBytes();
        } else if (!a2.isFloatingPoint()) {
            size = a1.numberOfBytes();
        }

        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(isFloatingPoint, size, a1.getShape());
        addInPlace(result, a1, a2, (byte) 1);
        return result;
    }


    /**
     * Adds a double value to numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The addition result.
     */
    public static NumpyArray<Double> add(NumpyArray array, double value) {
        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(true, NumpyArrayFactory.SIZE_OF_DOUBLE, array.getShape());
        addInPlace(result, array, value);

        return result;
    }

    /**
     * Subtract a double value from numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The subtraction result.
     */
    public static NumpyArray<Double> subtract(NumpyArray array, double value) {
        return add(array, -value);
    }

    /**
     * Adds a float value to numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The addition result.
     */
    public static NumpyArray<Float> add(NumpyArray array, float value) {
        NumpyArray<Float> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(true, NumpyArrayFactory.SIZE_OF_FLOAT, array.getShape());
        addInPlace(result, array, value);

        return result;
    }

    /**
     * Subtract a float value from numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The subtraction result.
     */
    public static NumpyArray<Float> subtract(NumpyArray array, float value) {
        return add(array, -value);
    }

    /**
     * Adds a byte value to numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The addition result.
     */
    public static NumpyArray add(NumpyArray array, byte value) {
        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(false, NumpyArrayFactory.SIZE_OF_INT_8, array.getShape());
        addInPlace(result, array, value);

        return result;
    }

    /**
     * Subtract a byte value from numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The subtraction result.
     */
    public static NumpyArray subtract(NumpyArray array, byte value) {
        return add(array, (byte) -value);
    }

    /**
     * Adds a short value to numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The addition result.
     */
    public static NumpyArray add(NumpyArray array, short value) {
        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(false, NumpyArrayFactory.SIZE_OF_INT_16, array.getShape());
        addInPlace(result, array, value);

        return result;
    }

    /**
     * Subtract a short value from numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The subtraction result.
     */
    public static NumpyArray subtract(NumpyArray array, short value) {
        return add(array, (short) -value);
    }

    /**
     * Adds a int value to numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The addition result.
     */
    public static NumpyArray add(NumpyArray array, int value) {
        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(false, NumpyArrayFactory.SIZE_OF_INT_32, array.getShape());
        addInPlace(result, array, value);

        return result;
    }

    /**
     * Subtract a int value from numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The subtraction result.
     */
    public static NumpyArray subtract(NumpyArray array, int value) {
        return add(array, -value);
    }

    /**
     * Adds a long value to numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The addition result.
     */
    public static NumpyArray add(NumpyArray array, long value) {
        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(false, NumpyArrayFactory.SIZE_OF_INT_64, array.getShape());
        addInPlace(result, array, value);

        return result;
    }

    /**
     * Subtract a long value from numpy arrays.
     *
     * @param array Left-hand side of the expression.
     * @param value Right-hand side of the expression.
     *
     * @return The subtraction result.
     */
    public static NumpyArray subtract(NumpyArray array, long value) {
        return add(array, -value);
    }

    /**
     * Adds two numpy array and stores the result into a target array.
     *
     * @param target The target array that stores the results.
     * @param a1 The left-hand side of the expression.
     * @param a2 The right-hand side of the expression.
     * @param sign A sign value to multiply by the right-hand side. The value of this parameter is either 1 or -1. The -1 is used to implement subtraction.
     */
    private static void addInPlace(NumpyArray target, NumpyArray a1, NumpyArray a2, byte sign) {
        if (a2.isSingleValueArray()) {
            Object singleValue = a2.getSingleValue();

            if (target.isFloatingPoint()) {
                if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                    addInPlace(target, a1, ((double) singleValue) * sign);
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                    addInPlace(target, a1, ((double) singleValue) * sign);
                }
            } else {
                if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                    addInPlace(target, a1, ((byte) singleValue) * sign);
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                    addInPlace(target, a1, ((short) singleValue) * sign);
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                    addInPlace(target, a1, ((int) singleValue) * sign);
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                    addInPlace(target, a1, ((long) singleValue) * sign);
                }
            }
        } else if (a1.numberOfDimensions() == 1 && a2.numberOfDimensions() == 1) {
            int firstDim = target.getShape()[0];

            for (int i = 0; i < firstDim; i++) {
                Object value = null;

                if (target.isFloatingPoint()) {
                    if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                        value = (double) a1.get(i) + sign * (double) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                        value = (float) a1.get(i) + sign * (float) a2.get(i);
                    }
                } else {
                    if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                        value = (byte) a1.get(i) + sign * (byte) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                        value = (short) a1.get(i) + sign * (short) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                        value = (int) a1.get(i) + sign * (int) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                        value = (long) a1.get(i) + sign * (long) a2.get(i);
                    }
                }

                target.set(value, i);
            }
        } else {
            int firstDim = target.getShape()[0];

            for (int i = 0; i < firstDim; i++) {
                addInPlace(target.wrapInnerSubsetArray(i), a1.wrapInnerSubsetArray(i), a2.wrapInnerSubsetArray(i), sign);
            }
        }
    }

    /**
     * Adds a double value to a numpy array.
     * @param target The array that stores the calculation.
     * @param array The left-hand side of the expression.
     * @param value The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, double value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (double) element);
    }

    /**
     * Adds a float value to a numpy array.
     * @param target The array that stores the calculation.
     * @param array The left-hand side of the expression.
     * @param value The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, float value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (float) element);
    }

    /**
     * Adds a long value to a numpy array.
     * @param target The array that stores the calculation.
     * @param array The left-hand side of the expression.
     * @param value The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, long value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (long) element);
    }

    /**
     * Adds a int value to a numpy array.
     * @param target The array that stores the calculation.
     * @param array The left-hand side of the expression.
     * @param value The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, int value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (int) element);
    }

    /**
     * Adds a short value to a numpy array.
     * @param target The array that stores the calculation.
     * @param array The left-hand side of the expression.
     * @param value The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, short value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (short) element);
    }

    /**
     * Adds a byte value to a numpy array.
     * @param target The array that stores the calculation.
     * @param array The left-hand side of the expression.
     * @param value The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, byte value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (byte) element);
    }

    /**
     * Multiplies a numpy array by a double value. The operation is element-wise.
     *
     * @param array The input array to be multiplied.
     * @param factor The value to be multiplied with.
     *
     * @return A numpy array of the calculation result.
     */
    public static NumpyArray<Double> multiply(NumpyArray<Double> array, double factor) {
        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array);

        array.applyToEachElementAnsSaveToTarget(result, value -> value * factor);

        return result;
    }

    /**
     * Multiplies a numpy array by a float value. The operation is element-wise.
     *
     * @param array The input array to be multiplied.
     * @param factor The value to be multiplied with.
     *
     * @return A numpy array of the calculation result.
     */
    public static NumpyArray<Float> multiply(NumpyArray<Float> array, float factor) {
        NumpyArray<Float> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array);

        array.applyToEachElementAnsSaveToTarget(result, value -> value * factor);

        return result;
    }

    /**
     * Divides a numpy array by a double value. The operation is element-wise.
     *
     * @param array The input array to be divided.
     * @param factor The value to be divided by.
     *
     * @return A numpy array of the calculation result.
     */
    public static NumpyArray<Double> divide(NumpyArray<Double> array, double factor) {
        return multiply(array, 1.0 / factor);
    }

    /**
     * Divides a numpy array by a float value. The operation is element-wise.
     *
     * @param array The input array to be divided.
     * @param factor The value to be divided by.
     *
     * @return A numpy array of the calculation result.
     */
    public static NumpyArray<Float> divide(NumpyArray<Float> array, float factor) {
        return multiply(array, 1.0f / factor);
    }

    /**
     * Wraps an atomic double value into a 2 dimensional array.
     *
     * @param value The value to be wrapped into an array.
     *
     * @return A two dimensional array that wraps the given value.
     */
    public static NumpyArray<Double> atLeast2D(double value) {
        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(true, NumpyArrayFactory.SIZE_OF_DOUBLE, new int[]{1, 1});
        result.set(value, 0, 0);

        return result;
    }

    /**
     * Wraps an atomic float value into a 2 dimensional array.
     *
     * @param value The value to be wrapped into an array.
     *
     * @return A two dimensional array that wraps the given value.
     */
    public static NumpyArray<Float> atLeast2D(float value) {
        NumpyArray<Float> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(true, NumpyArrayFactory.SIZE_OF_FLOAT, new int[]{1, 1});
        result.set(value, 0, 0);

        return result;
    }

    /**
     * Wraps an atomic long value into a 2 dimensional array.
     *
     * @param value The value to be wrapped into an array.
     *
     * @return A two dimensional array that wraps the given value.
     */
    public static NumpyArray<Long> atLeast2D(long value) {
        NumpyArray<Long> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(false, NumpyArrayFactory.SIZE_OF_INT_64, new int[]{1, 1});
        result.set(value, 0, 0);

        return result;
    }

    /**
     * Wraps an atomic int value into a 2 dimensional array.
     *
     * @param value The value to be wrapped into an array.
     *
     * @return A two dimensional array that wraps the given value.
     */
    public static NumpyArray<Integer> atLeast2D(int value) {
        NumpyArray<Integer> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(false, NumpyArrayFactory.SIZE_OF_INT_32, new int[]{1, 1});
        result.set(value, 0, 0);

        return result;
    }

    /**
     * Wraps an atomic short value into a 2 dimensional array.
     *
     * @param value The value to be wrapped into an array.
     *
     * @return A two dimensional array that wraps the given value.
     */
    public static NumpyArray<Short> atLeast2D(short value) {
        NumpyArray<Short> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(false, NumpyArrayFactory.SIZE_OF_INT_16, new int[]{1, 1});
        result.set(value, 0, 0);

        return result;
    }

    /**
     * Wraps an atomic byte value into a 2 dimensional array.
     *
     * @param value The value to be wrapped into an array.
     *
     * @return A two dimensional array that wraps the given value.
     */
    public static NumpyArray<Byte> atLeast2D(byte value) {
        NumpyArray<Byte> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(false, NumpyArrayFactory.SIZE_OF_INT_8, new int[]{1, 1});
        result.set(value, 0, 0);

        return result;
    }

    /**
     * Wraps a numpy array into a 2 dimensional array if the number dimensions is less than 2.
     *
     * @param array The array to be wrapped into a 2 dimensional array.
     *
     * @return A two dimensional array that wraps the given value.
     */
    public static <Type> NumpyArray<Type> atLeast2D(NumpyArray<Type> array) {
        // https://github.com/numpy/numpy/blob/v1.22.0/numpy/core/shape_base.py#L81-L132
        NumpyArray<Type> result = null;

        if (array.numberOfDimensions() == 1) {
            result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array.isFloatingPoint(), array.numberOfBytes(), new int[]{1, array.getShape()[0]});
            for (int i = 0; i < array.getShape()[0]; i++) {
                result.set(array.get(i), 0, i);
            }
        } else if (array.numberOfDimensions() > 1) {
            result = array;
        } else {
            throw new NumpyOperationException("The input for atLeast2D is invalid");
        }

        return result;
    }
}
