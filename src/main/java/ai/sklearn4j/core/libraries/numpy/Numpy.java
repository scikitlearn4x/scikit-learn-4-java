package ai.sklearn4j.core.libraries.numpy;

/**
 * Implementation of the Numpy library APIs.
 */
public final class Numpy {
    /**
     * Returns the indices of the maximum values along an axis.
     * See: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
     *
     * @param array          The input multidimensional array.
     * @param axis           The axis which the argmax should reduce to.
     * @param keepDimensions A flag to specify whether to keep the reduced dimension in the output.
     * @return Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.
     */
    public static <Type> NumpyArray<Long> argmax(NumpyArray<Type> array, int axis, boolean keepDimensions) {
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

        return operation.apply(array, axis, keepDimensions);
    }

    /**
     * Performs an element-wise power operation on a given NumpyArray.
     * See: https://numpy.org/doc/stable/reference/generated/numpy.power.html
     *
     * @param array Input array.
     * @param power The value of the power.
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
     * @param array          Input array.
     * @param axis           Axis along which a sum is performed.
     * @param keepDimensions A flag to specify whether to keep the reduced dimension in the output.
     * @return An array with the same shape as a, with the specified axis removed.
     */
    public static NumpyArray sum(NumpyArray array, int axis, boolean keepDimensions) {
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

        return operation.apply(array, axis, keepDimensions);
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

        // This check is for the case where 2 arrays are added with shapes: a1: [5,3], a2: [3]
        if ((shape1.length > shape2.length && isShapeEndingLike(shape1, shape2)) ||
                (shape2.length > shape1.length && isShapeEndingLike(shape2, shape1))) {
            return;
        }

        for (int i = 0; i < effective1; i++) {
            if (shape1[i] != shape2[i] && (shape1[i] != 1 && shape2[i] != 1)) {
                throw new NumpyOperationException(String.format("Dimension %d of the two numpy arrays doesn't match.", i + 1));
            }
        }
    }

    /**
     * Check if the shape 1 later dimensions are the shape as the one specified by shape 2.
     *
     * @param shape1 The base shape to check.
     * @param shape2 The shorter shape to check.
     * @return A boolean indicating if shape 1 ends with shape 2.
     */
    private static boolean isShapeEndingLike(int[] shape1, int[] shape2) {
        boolean result = true;

        for (int i = 0; i < shape2.length; i++) {
            if (shape2[shape2.length - i - 1] != shape1[shape1.length - i - 1]) {
                result = false;
                break;
            }
        }

        return result;
    }

    /**
     * Subtract two numpy arrays.
     *
     * @param a1 Left-hand side of the expression.
     * @param a2 Right-hand side of the expression.
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
     * @return The subtraction result.
     */
    public static NumpyArray subtract(NumpyArray array, long value) {
        return add(array, -value);
    }

    /**
     * Adds two numpy array and stores the result into a target array.
     *
     * @param target The target array that stores the results.
     * @param a1     The left-hand side of the expression.
     * @param a2     The right-hand side of the expression.
     * @param sign   A sign value to multiply by the right-hand side. The value of this parameter is either 1 or -1. The -1 is used to implement subtraction.
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
        } else if (a1.numberOfDimensions() > 1 && a2.numberOfDimensions() == 1) {
            int[] leftNoneCommonShape = new int[a1.numberOfDimensions() - 1];
            int[] index = new int[a1.numberOfDimensions()];
            for (int i = 0; i < leftNoneCommonShape.length; i++) {
                leftNoneCommonShape[i] = a1.getShape()[i];
            }

            int[] counter = new int[leftNoneCommonShape.length + 1];
            int rightShape = a2.getShape()[0];

            do {
                NumpyArray.addCounter(counter, leftNoneCommonShape);

                for (int i = 0; i < leftNoneCommonShape.length; i++) {
                    index[i] = counter[i];
                }

                for (int i = 0; i < rightShape; i++) {
                    index[index.length - 1] = i;
                    Object value = null;

                    if (target.isFloatingPoint()) {
                        if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                            value = (double) a1.get(index) + sign * (double) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                            value = (float) a1.get(index) + sign * (float) a2.get(i);
                        }
                    } else {
                        if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                            value = (byte) a1.get(index) + sign * (byte) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                            value = (short) a1.get(index) + sign * (short) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                            value = (int) a1.get(index) + sign * (int) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                            value = (long) a1.get(index) + sign * (long) a2.get(i);
                        }
                    }

                    target.set(value, index);
                }
            } while (counter[counter.length - 1] == 0);
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
                NumpyArray leftWrap = a1.wrapInnerSubsetArray(i);
                NumpyArray rightWrap = null;

                if (a2.getShape()[0] == 1) {
                    rightWrap = a2.wrapInnerSubsetArray(0);
                } else {
                    rightWrap = a2.wrapInnerSubsetArray(i);
                }

                addInPlace(target.wrapInnerSubsetArray(i), leftWrap, rightWrap, sign);
            }
        }
    }

    /**
     * Adds a double value to a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, double value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (double) element);
    }

    /**
     * Adds a float value to a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, float value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (float) element);
    }

    /**
     * Adds a long value to a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, long value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (long) element);
    }

    /**
     * Adds a int value to a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, int value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (int) element);
    }

    /**
     * Adds a short value to a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, short value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (short) element);
    }

    /**
     * Adds a byte value to a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be added.
     */
    private static void addInPlace(NumpyArray target, NumpyArray array, byte value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value + (byte) element);
    }

    /**
     * Multiplies two numpy arrays.
     *
     * @param a1 Left-hand side of the expression.
     * @param a2 Right-hand side of the expression.
     * @return The multiplication result.
     */
    public static NumpyArray multiply(NumpyArray a1, NumpyArray a2) {
        validateDimensionsForAdd(a1.getShape(), a2.getShape());
        if (shouldSwapForAdd(a1, a2)) {
            return multiply(a2, a1);
        }

        boolean isFloatingPoint = a1.isFloatingPoint() || a2.isFloatingPoint();
        int size = Math.max(a1.numberOfBytes(), a2.numberOfBytes());

        if (!a1.isFloatingPoint()) {
            size = a2.numberOfBytes();
        } else if (!a2.isFloatingPoint()) {
            size = a1.numberOfBytes();
        }

        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(isFloatingPoint, size, a1.getShape());
        multiplyInPlace(result, a1, a2);
        return result;
    }

    /**
     * Multiplies two numpy array and stores the result into a target array.
     *
     * @param target The target array that stores the results.
     * @param a1     The left-hand side of the expression.
     * @param a2     The right-hand side of the expression.
     */
    private static void multiplyInPlace(NumpyArray target, NumpyArray a1, NumpyArray a2) {
        if (a2.isSingleValueArray()) {
            Object singleValue = a2.getSingleValue();

            if (target.isFloatingPoint()) {
                if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                    multiplyInPlace(target, a1, ((double) singleValue));
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                    multiplyInPlace(target, a1, ((double) singleValue));
                }
            } else {
                if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                    multiplyInPlace(target, a1, ((byte) singleValue));
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                    multiplyInPlace(target, a1, ((short) singleValue));
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                    multiplyInPlace(target, a1, ((int) singleValue));
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                    multiplyInPlace(target, a1, ((long) singleValue));
                }
            }
        } else if (a1.numberOfDimensions() > 1 && a2.numberOfDimensions() == 1) {
            int[] leftNoneCommonShape = new int[a1.numberOfDimensions() - 1];
            int[] index = new int[a1.numberOfDimensions()];
            for (int i = 0; i < leftNoneCommonShape.length; i++) {
                leftNoneCommonShape[i] = a1.getShape()[i];
            }

            int[] counter = new int[leftNoneCommonShape.length + 1];
            int rightShape = a2.getShape()[0];

            do {
                NumpyArray.addCounter(counter, leftNoneCommonShape);

                for (int i = 0; i < leftNoneCommonShape.length; i++) {
                    index[i] = counter[i];
                }

                for (int i = 0; i < rightShape; i++) {
                    index[index.length - 1] = i;
                    Object value = null;

                    if (target.isFloatingPoint()) {
                        if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                            value = (double) a1.get(index) * (double) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                            value = (float) a1.get(index) * (float) a2.get(i);
                        }
                    } else {
                        if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                            value = (byte) a1.get(index) * (byte) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                            value = (short) a1.get(index) * (short) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                            value = (int) a1.get(index) * (int) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                            value = (long) a1.get(index) * (long) a2.get(i);
                        }
                    }

                    target.set(value, index);
                }
            } while (counter[counter.length - 1] == 0);
        } else if (a1.numberOfDimensions() == 1 && a2.numberOfDimensions() == 1) {
            int firstDim = target.getShape()[0];

            for (int i = 0; i < firstDim; i++) {
                Object value = null;

                if (target.isFloatingPoint()) {
                    if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                        value = (double) a1.get(i) * (double) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                        value = (float) a1.get(i) * (float) a2.get(i);
                    }
                } else {
                    if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                        value = (byte) a1.get(i) * (byte) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                        value = (short) a1.get(i) * (short) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                        value = (int) a1.get(i) * (int) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                        value = (long) a1.get(i) * (long) a2.get(i);
                    }
                }

                target.set(value, i);
            }
        } else {
            int firstDim = target.getShape()[0];

            for (int i = 0; i < firstDim; i++) {
                NumpyArray leftWrap = a1.wrapInnerSubsetArray(i);
                NumpyArray rightWrap = null;

                if (a2.getShape()[0] == 1) {
                    rightWrap = a2.wrapInnerSubsetArray(0);
                } else {
                    rightWrap = a2.wrapInnerSubsetArray(i);
                }

                multiplyInPlace(target.wrapInnerSubsetArray(i), leftWrap, rightWrap);
            }
        }
    }


    /**
     * Multiplies a double value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be multiplied by.
     */
    private static void multiplyInPlace(NumpyArray target, NumpyArray array, double value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value * (double) element);
    }

    /**
     * Multiplies a float value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be multiplied by.
     */
    private static void multiplyInPlace(NumpyArray target, NumpyArray array, float value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value * (float) element);
    }

    /**
     * Multiplies a long value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be multiplied by.
     */
    private static void multiplyInPlace(NumpyArray target, NumpyArray array, long value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value * (long) element);
    }

    /**
     * Multiplies a int value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be multiplied by.
     */
    private static void multiplyInPlace(NumpyArray target, NumpyArray array, int value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value * (int) element);
    }

    /**
     * Multiplies a short value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be multiplied by.
     */
    private static void multiplyInPlace(NumpyArray target, NumpyArray array, short value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value * (short) element);
    }

    /**
     * Multiplies a byte value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be multiplied by.
     */
    private static void multiplyInPlace(NumpyArray target, NumpyArray array, byte value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> value * (byte) element);
    }


    /**
     * Multiplies a numpy array by a double value. The operation is element-wise.
     *
     * @param array  The input array to be multiplied.
     * @param factor The value to be multiplied with.
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
     * @param array  The input array to be multiplied.
     * @param factor The value to be multiplied with.
     * @return A numpy array of the calculation result.
     */
    public static NumpyArray<Float> multiply(NumpyArray<Float> array, float factor) {
        NumpyArray<Float> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array);

        array.applyToEachElementAnsSaveToTarget(result, value -> value * factor);

        return result;
    }

    /**
     * Divides two numpy arrays.
     *
     * @param a1 Left-hand side of the expression.
     * @param a2 Right-hand side of the expression.
     * @return The multiplication result.
     */
    public static NumpyArray divide(NumpyArray a1, NumpyArray a2) {
        validateDimensionsForAdd(a1.getShape(), a2.getShape());
        if (shouldSwapForAdd(a1, a2)) {
            throw new NumpyOperationException("This division is not supported.");
        }

        boolean isFloatingPoint = a1.isFloatingPoint() || a2.isFloatingPoint();
        int size = Math.max(a1.numberOfBytes(), a2.numberOfBytes());

        if (!a1.isFloatingPoint()) {
            size = a2.numberOfBytes();
        } else if (!a2.isFloatingPoint()) {
            size = a1.numberOfBytes();
        }

        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(isFloatingPoint, size, a1.getShape());
        divideInPlace(result, a1, a2);
        return result;
    }

    /**
     * Divides two numpy array and stores the result into a target array.
     *
     * @param target The target array that stores the results.
     * @param a1     The left-hand side of the expression.
     * @param a2     The right-hand side of the expression.
     */
    private static void divideInPlace(NumpyArray target, NumpyArray a1, NumpyArray a2) {
        if (a2.isSingleValueArray()) {
            Object singleValue = a2.getSingleValue();

            if (target.isFloatingPoint()) {
                if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                    divideInPlace(target, a1, ((double) singleValue));
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                    divideInPlace(target, a1, ((double) singleValue));
                }
            } else {
                if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                    divideInPlace(target, a1, ((byte) singleValue));
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                    divideInPlace(target, a1, ((short) singleValue));
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                    divideInPlace(target, a1, ((int) singleValue));
                } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                    divideInPlace(target, a1, ((long) singleValue));
                }
            }
        } else if (a1.numberOfDimensions() > 1 && a2.numberOfDimensions() == 1) {
            int[] leftNoneCommonShape = new int[a1.numberOfDimensions() - 1];
            int[] index = new int[a1.numberOfDimensions()];
            for (int i = 0; i < leftNoneCommonShape.length; i++) {
                leftNoneCommonShape[i] = a1.getShape()[i];
            }

            int[] counter = new int[leftNoneCommonShape.length + 1];
            int rightShape = a2.getShape()[0];

            do {
                NumpyArray.addCounter(counter, leftNoneCommonShape);

                for (int i = 0; i < leftNoneCommonShape.length; i++) {
                    index[i] = counter[i];
                }

                for (int i = 0; i < rightShape; i++) {
                    index[index.length - 1] = i;
                    Object value = null;

                    if (target.isFloatingPoint()) {
                        if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                            value = (double) a1.get(index) / (double) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                            value = (float) a1.get(index) / (float) a2.get(i);
                        }
                    } else {
                        if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                            value = (byte) a1.get(index) / (byte) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                            value = (short) a1.get(index) / (short) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                            value = (int) a1.get(index) / (int) a2.get(i);
                        } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                            value = (long) a1.get(index) / (long) a2.get(i);
                        }
                    }

                    target.set(value, index);
                }
            } while (counter[counter.length - 1] == 0);
        } else if (a1.numberOfDimensions() == 1 && a2.numberOfDimensions() == 1) {
            int firstDim = target.getShape()[0];

            for (int i = 0; i < firstDim; i++) {
                Object value = null;

                if (target.isFloatingPoint()) {
                    if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                        value = (double) a1.get(i) / (double) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                        value = (float) a1.get(i) / (float) a2.get(i);
                    }
                } else {
                    if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                        value = (byte) a1.get(i) / (byte) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                        value = (short) a1.get(i) / (short) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                        value = (int) a1.get(i) / (int) a2.get(i);
                    } else if (target.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                        value = (long) a1.get(i) / (long) a2.get(i);
                    }
                }

                target.set(value, i);
            }
        } else {
            int firstDim = target.getShape()[0];

            for (int i = 0; i < firstDim; i++) {
                NumpyArray leftWrap = a1.wrapInnerSubsetArray(i);
                NumpyArray rightWrap = null;

                if (a2.getShape()[0] == 1) {
                    rightWrap = a2.wrapInnerSubsetArray(0);
                } else {
                    rightWrap = a2.wrapInnerSubsetArray(i);
                }

                divideInPlace(target.wrapInnerSubsetArray(i), leftWrap, rightWrap);
            }
        }
    }


    /**
     * Divides a double value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be divided by.
     */
    private static void divideInPlace(NumpyArray target, NumpyArray array, double value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> ((double) element) / value);
    }

    /**
     * Divides a float value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be divided by.
     */
    private static void divideInPlace(NumpyArray target, NumpyArray array, float value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> ((float) element) / value);
    }

    /**
     * Divides a long value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be divided by.
     */
    private static void divideInPlace(NumpyArray target, NumpyArray array, long value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> ((long) element) / value);
    }

    /**
     * Divides a int value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be divided by.
     */
    private static void divideInPlace(NumpyArray target, NumpyArray array, int value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> ((int) element) / value);
    }

    /**
     * Divides a short value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be divided by.
     */
    private static void divideInPlace(NumpyArray target, NumpyArray array, short value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> ((short) element) / value);
    }

    /**
     * Divides a byte value by a numpy array.
     *
     * @param target The array that stores the calculation.
     * @param array  The left-hand side of the expression.
     * @param value  The value to be divided by.
     */
    private static void divideInPlace(NumpyArray target, NumpyArray array, byte value) {
        array.applyToEachElementAnsSaveToTarget(target, element -> ((byte) element) / value);
    }

    /**
     * Divides a numpy array by a double value. The operation is element-wise.
     *
     * @param array  The input array to be divided.
     * @param factor The value to be divided by.
     * @return A numpy array of the calculation result.
     */
    public static NumpyArray<Double> divide(NumpyArray<Double> array, double factor) {
        return multiply(array, 1.0 / factor);
    }

    /**
     * Divides a numpy array by a float value. The operation is element-wise.
     *
     * @param array  The input array to be divided.
     * @param factor The value to be divided by.
     * @return A numpy array of the calculation result.
     */
    public static NumpyArray<Float> divide(NumpyArray<Float> array, float factor) {
        return multiply(array, 1.0f / factor);
    }

    /**
     * Wraps an atomic double value into a 2 dimensional array.
     *
     * @param value The value to be wrapped into an array.
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

    /**
     * Returns the maximum values along an axis.
     * See: https://numpy.org/doc/stable/reference/generated/numpy.amax.html
     *
     * @param array          The input multidimensional array.
     * @param axis           The axis which the amax should reduce to.
     * @param keepDimensions A flag to specify whether to keep the reduced dimension in the output.
     * @return Array of maximum into the array. It has the same shape as a.shape with the dimension along axis removed.
     */
    public static NumpyArray<Double> arrayMax(NumpyArray<Double> array, int axis, boolean keepDimensions) {
        NumpyArrayOperationWithAxisReduction<Double, Double> operation = new NumpyArrayOperationWithAxisReduction<>() {
            @Override
            public NumpyArray<Double> createInstanceResultNumpyArray(int[] shape) {
                return NumpyArrayFactory.arrayOfDoubleWithShape(shape);
            }

            @Override
            public Object reduceAxisValues(Object[] valuesInAxis) {
                double max = (double) valuesInAxis[0];

                for (int i = 1; i < valuesInAxis.length; i++) {
                    double m = (double) valuesInAxis[i];
                    if (m > max) {
                        max = m;
                    }
                }

                return max;
            }
        };

        return operation.apply(array, axis, keepDimensions);
    }

    /**
     * Remove axes of length one from the array.
     *
     * @param array The array to squeeze.
     * @return An array without any dimension of length 1.
     */
    public static NumpyArray squeeze(NumpyArray array) {
        int desiredDimensions = 0;
        for (int i = 0; i < array.getShape().length; i++) {
            int dim = array.getShape()[i];
            if (dim > 1) {
                desiredDimensions++;
            }
        }

        int[] shape = new int[desiredDimensions];
        int[] mapper = new int[desiredDimensions];
        int indexOnShape = 0;
        for (int i = 0; i < array.getShape().length; i++) {
            int dim = array.getShape()[i];
            if (dim > 1) {
                shape[indexOnShape] = dim;
                mapper[indexOnShape] = i;
                indexOnShape++;
            }
        }

        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array.isFloatingPoint(), array.numberOfBytes(), shape);
        int[] counter = new int[shape.length + 1];
        int[] indexOnInput = new int[array.getShape().length];

        do {
            NumpyArray.addCounter(counter, shape);
            for (int i = 0; i < mapper.length; i++) {
                indexOnInput[mapper[i]] = counter[i];
            }
            result.set(array.get(indexOnInput), counter);
        } while (counter[counter.length - 1] == 0);

        return result;
    }

    /**
     * Clip (limit) the values in an array.
     * <p>
     * Given an interval, values outside the interval are clipped to the interval edges.
     * For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
     * and values larger than 1 become 1.
     * <p>
     * Equivalent to but faster than np.minimum(a_max, np.maximum(a, a_min)).
     *
     * @param array Array containing elements to clip.
     * @param min   The minimum value to clip.
     * @param max   The maximum value to clip.
     * @return An array with the elements of array, but where values less than min are replaced
     * with min, and those greater than max with max.
     */
    public static NumpyArray<Double> clip(NumpyArray<Double> array, double min, double max) {
        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array);

        array.applyToEachElementAnsSaveToTarget(result, value -> {
            if (value > max) {
                return max;
            } else if (value < min) {
                return min;
            }

            return value;
        });

        return result;
    }

    /**
     * Calculate the absolute value element-wise.
     *
     * @param array Input array.
     * @return An ndarray containing the absolute value of each element in x.
     */
    public static NumpyArray abs(NumpyArray<Double> array) {
        NumpyArray result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array);
        INumpyArrayElementOperation absOperation = null;

        if (array.isFloatingPoint()) {
            if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                absOperation = value -> Math.abs((double) value);
            } else if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                absOperation = value -> Math.abs((float) value);
            }
        } else {
            if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                absOperation = value -> Math.abs((byte) value);
            } else if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                absOperation = value -> Math.abs((short) value);
            } else if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                absOperation = value -> Math.abs((int) value);
            } else if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                absOperation = value -> Math.abs((long) value);
            }
        }

        array.applyToEachElementAnsSaveToTarget(result, absOperation);

        return result;
    }

    /**
     * Return the non-negative square-root of an array, element-wise.
     *
     * @param array The values whose square-roots are required.
     * @return An array of the same shape as x, containing the positive square-root
     * of each element in x.
     */
    public static NumpyArray<Double> sqrt(NumpyArray array) {
        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(true, NumpyArrayFactory.SIZE_OF_DOUBLE, array.getShape());
        INumpyArrayElementOperation sqrtOperation = null;

        if (array.isFloatingPoint()) {
            if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                sqrtOperation = value -> Math.sqrt((double) value);
            } else if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_FLOAT) {
                sqrtOperation = value -> Math.sqrt((float) value);
            }
        } else {
            if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_8) {
                sqrtOperation = value -> Math.sqrt((byte) value);
            } else if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_16) {
                sqrtOperation = value -> Math.sqrt((short) value);
            } else if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_32) {
                sqrtOperation = value -> Math.sqrt((int) value);
            } else if (array.numberOfBytes() == NumpyArrayFactory.SIZE_OF_INT_64) {
                sqrtOperation = value -> Math.sqrt((long) value);
            }
        }

        array.applyToEachElementAnsSaveToTarget(result, sqrtOperation);

        return result;
    }

}
