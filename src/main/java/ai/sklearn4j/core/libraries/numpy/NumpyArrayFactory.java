package ai.sklearn4j.core.libraries.numpy;

import ai.sklearn4j.core.libraries.numpy.wrappers.*;

/**
 * A factory class that facilitate the creation of NumpyArrays.
 */
public class NumpyArrayFactory {
    /**
     * Number of bytes used to allocate a byte.
     */
    public static final int SIZE_OF_INT_8 = 1;

    /**
     * Number of bytes used to allocate a short.
     */
    public static final int SIZE_OF_INT_16 = 2;

    /**
     * Number of bytes used to allocate an int.
     */
    public static final int SIZE_OF_INT_32 = 4;

    /**
     * Number of bytes used to allocate a long.
     */
    public static final int SIZE_OF_INT_64 = 8;

    /**
     * Number of bytes used to allocate a float.
     */
    public static final int SIZE_OF_FLOAT = 4;

    /**
     * Number of bytes used to allocate a double.
     */
    public static final int SIZE_OF_DOUBLE = 8;

    /**
     * Creates a numpy array of the same dimension and data type of the provided one.
     *
     * @param other The other array to create a similar array to.
     * @return An empty array with the same shape and data type.
     */
    public static NumpyArray createArrayOfShapeAndTypeInfo(NumpyArray other) {
        return createArrayOfShapeAndTypeInfo(other.isFloatingPoint(), other.numberOfBytes(), other.getShape());
    }

    /**
     * Creates a numpy array with a specified shape that can store values with specified
     * characteristics.
     *
     * @param isFloatingPoint Indicates if the elements are floating point.
     * @param size            Indicates the number of bytes each element should allocate in memory.
     * @param shape           Shape of the desired new array.
     * @return An array of the specified shape and data characteristics.
     */
    public static NumpyArray createArrayOfShapeAndTypeInfo(boolean isFloatingPoint, int size, int[] shape) {
        NumpyArray result = null;

        if (isFloatingPoint) {
            if (size == NumpyArrayFactory.SIZE_OF_DOUBLE) {
                result = NumpyArrayFactory.arrayOfDoubleWithShape(shape);
            } else {
                result = NumpyArrayFactory.arrayOfFloatWithShape(shape);
            }
        } else {
            if (size == NumpyArrayFactory.SIZE_OF_INT_64) {
                result = NumpyArrayFactory.arrayOfInt64WithShape(shape);
            } else if (size == NumpyArrayFactory.SIZE_OF_INT_32) {
                result = NumpyArrayFactory.arrayOfInt32WithShape(shape);
            } else if (size == NumpyArrayFactory.SIZE_OF_INT_16) {
                result = NumpyArrayFactory.arrayOfInt16WithShape(shape);
            } else if (size == NumpyArrayFactory.SIZE_OF_INT_8) {
                result = NumpyArrayFactory.arrayOfInt8WithShape(shape);
            }
        }

        return result;
    }

    /**
     * Converts a value into a byte.
     *
     * @param o The value to be converted.
     * @return A byte value.
     */
    public static byte toByte(Object o) {
        byte result = 0;

        if (o instanceof Byte) {
            result = (byte) o;
        } else if (o instanceof Short) {
            result = (byte) ((short) o);
        } else if (o instanceof Integer) {
            result = (byte) ((int) o);
        } else if (o instanceof Long) {
            result = (byte) ((long) o);
        } else {
            throw new NumpyOperationException("Invalid casting value to byte.");
        }

        return result;
    }

    /**
     * Converts a value into a double.
     *
     * @param value The value to be converted.
     * @return A double value.
     */
    public static double toDouble(Object value) {
        return (double) value;
    }

    /**
     * Converts a value into a float.
     *
     * @param value The value to be converted.
     * @return A float value.
     */
    public static float toFloat(Object value) {
        return (float) value;
    }

    /**
     * Converts a value into a short.
     *
     * @param o The value to be converted.
     * @return A short value.
     */
    public static short toShort(Object o) {
        short result = 0;

        if (o instanceof Byte) {
            result = (short) o;
        } else if (o instanceof Short) {
            result = (short) ((short) o);
        } else if (o instanceof Integer) {
            result = (short) ((int) o);
        } else if (o instanceof Long) {
            result = (short) ((long) o);
        } else {
            throw new NumpyOperationException("Invalid casting value to short.");
        }

        return result;

    }

    /**
     * Converts a value into an int.
     *
     * @param o The value to be converted.
     * @return A int value.
     */
    public static int toInteger(Object o) {
        int result = 0;

        if (o instanceof Byte) {
            result = (int) o;
        } else if (o instanceof Short) {
            result = (int) ((short) o);
        } else if (o instanceof Integer) {
            result = (int) ((int) o);
        } else if (o instanceof Long) {
            result = (int) ((long) o);
        } else {
            throw new NumpyOperationException("Invalid casting value to int.");
        }

        return result;
    }

    /**
     * Converts a value into a long.
     *
     * @param o The value to be converted.
     * @return A long value.
     */
    public static long toLong(Object o) {
        long result = 0;

        if (o instanceof Byte) {
            result = (long) o;
        } else if (o instanceof Short) {
            result = (long) ((short) o);
        } else if (o instanceof Integer) {
            result = (long) ((int) o);
        } else if (o instanceof Long) {
            result = (long) ((long) o);
        } else {
            throw new NumpyOperationException("Invalid casting value to long.");
        }

        return result;

    }

    /**
     * Create a numpy array wrapper over a 1 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[] array) {
        return new NumpyArray<>(new Dim1Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 2 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][] array) {
        return new NumpyArray<>(new Dim2Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 3 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][][] array) {
        return new NumpyArray<>(new Dim3Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 4 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][][][] array) {
        return new NumpyArray<>(new Dim4Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 5 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][][][][] array) {
        return new NumpyArray<>(new Dim5Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 6 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][][][][][] array) {
        return new NumpyArray<>(new Dim6Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 7 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][][][][][][] array) {
        return new NumpyArray<>(new Dim7Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 8 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 9 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 10 dimension byte array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Byte> from(byte[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10Int8NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 1 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[] array) {
        return new NumpyArray<>(new Dim1Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 2 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][] array) {
        return new NumpyArray<>(new Dim2Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 3 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][][] array) {
        return new NumpyArray<>(new Dim3Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 4 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][][][] array) {
        return new NumpyArray<>(new Dim4Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 5 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][][][][] array) {
        return new NumpyArray<>(new Dim5Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 6 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][][][][][] array) {
        return new NumpyArray<>(new Dim6Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 7 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][][][][][][] array) {
        return new NumpyArray<>(new Dim7Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 8 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 9 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 10 dimension short array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Short> from(short[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10Int16NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 1 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[] array) {
        return new NumpyArray<>(new Dim1Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 2 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][] array) {
        return new NumpyArray<>(new Dim2Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 3 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][][] array) {
        return new NumpyArray<>(new Dim3Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 4 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][][][] array) {
        return new NumpyArray<>(new Dim4Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 5 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][][][][] array) {
        return new NumpyArray<>(new Dim5Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 6 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][][][][][] array) {
        return new NumpyArray<>(new Dim6Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 7 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][][][][][][] array) {
        return new NumpyArray<>(new Dim7Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 8 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 9 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 10 dimension int array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Integer> from(int[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10Int32NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 1 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[] array) {
        return new NumpyArray<>(new Dim1Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 2 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][] array) {
        return new NumpyArray<>(new Dim2Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 3 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][][] array) {
        return new NumpyArray<>(new Dim3Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 4 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][][][] array) {
        return new NumpyArray<>(new Dim4Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 5 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][][][][] array) {
        return new NumpyArray<>(new Dim5Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 6 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][][][][][] array) {
        return new NumpyArray<>(new Dim6Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 7 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][][][][][][] array) {
        return new NumpyArray<>(new Dim7Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 8 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 9 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 10 dimension long array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Long> from(long[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10Int64NumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 1 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[] array) {
        return new NumpyArray<>(new Dim1FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 2 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][] array) {
        return new NumpyArray<>(new Dim2FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 3 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][][] array) {
        return new NumpyArray<>(new Dim3FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 4 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][][][] array) {
        return new NumpyArray<>(new Dim4FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 5 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][][][][] array) {
        return new NumpyArray<>(new Dim5FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 6 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][][][][][] array) {
        return new NumpyArray<>(new Dim6FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 7 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][][][][][][] array) {
        return new NumpyArray<>(new Dim7FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 8 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 9 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 10 dimension float array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Float> from(float[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10FloatNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 1 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[] array) {
        return new NumpyArray<>(new Dim1DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 2 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][] array) {
        return new NumpyArray<>(new Dim2DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 3 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][][] array) {
        return new NumpyArray<>(new Dim3DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 4 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][][][] array) {
        return new NumpyArray<>(new Dim4DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 5 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][][][][] array) {
        return new NumpyArray<>(new Dim5DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 6 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][][][][][] array) {
        return new NumpyArray<>(new Dim6DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 7 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][][][][][][] array) {
        return new NumpyArray<>(new Dim7DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 8 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 9 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array wrapper over a 10 dimension double array.
     *
     * @param array The array to be wrapped.
     * @return The wrapped numpy array.
     */
    public static NumpyArray<Double> from(double[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10DoubleNumpyWrapper(array));
    }

    /**
     * Create a numpy array of byte with specified shape.
     *
     * @param shape The shape of the new array.
     * @return The new numpy array with desired shape.
     */
    public static NumpyArray<Byte> arrayOfInt8WithShape(int[] shape) {
        if (shape.length == 1) {
            return from(new byte[shape[0]]);
        } else if (shape.length == 2) {
            return from(new byte[shape[0]][shape[1]]);
        } else if (shape.length == 3) {
            return from(new byte[shape[0]][shape[1]][shape[2]]);
        } else if (shape.length == 4) {
            return from(new byte[shape[0]][shape[1]][shape[2]][shape[3]]);
        } else if (shape.length == 5) {
            return from(new byte[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]]);
        } else if (shape.length == 6) {
            return from(new byte[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]]);
        } else if (shape.length == 7) {
            return from(new byte[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]]);
        } else if (shape.length == 8) {
            return from(new byte[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]]);
        } else if (shape.length == 9) {
            return from(new byte[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]]);
        } else if (shape.length == 10) {
            return from(new byte[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]][shape[9]]);
        }

        throw new NumpyOperationException("The shape specified is not supported; only arrays less than 10 dimensions are supported.");
    }

    /**
     * Create a numpy array of short with specified shape.
     *
     * @param shape The shape of the new array.
     * @return The new numpy array with desired shape.
     */
    public static NumpyArray<Short> arrayOfInt16WithShape(int[] shape) {
        if (shape.length == 1) {
            return from(new short[shape[0]]);
        } else if (shape.length == 2) {
            return from(new short[shape[0]][shape[1]]);
        } else if (shape.length == 3) {
            return from(new short[shape[0]][shape[1]][shape[2]]);
        } else if (shape.length == 4) {
            return from(new short[shape[0]][shape[1]][shape[2]][shape[3]]);
        } else if (shape.length == 5) {
            return from(new short[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]]);
        } else if (shape.length == 6) {
            return from(new short[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]]);
        } else if (shape.length == 7) {
            return from(new short[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]]);
        } else if (shape.length == 8) {
            return from(new short[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]]);
        } else if (shape.length == 9) {
            return from(new short[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]]);
        } else if (shape.length == 10) {
            return from(new short[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]][shape[9]]);
        }

        throw new NumpyOperationException("The shape specified is not supported; only arrays less than 10 dimensions are supported.");
    }

    /**
     * Create a numpy array of int with specified shape.
     *
     * @param shape The shape of the new array.
     * @return The new numpy array with desired shape.
     */
    public static NumpyArray<Integer> arrayOfInt32WithShape(int[] shape) {
        if (shape.length == 1) {
            return from(new int[shape[0]]);
        } else if (shape.length == 2) {
            return from(new int[shape[0]][shape[1]]);
        } else if (shape.length == 3) {
            return from(new int[shape[0]][shape[1]][shape[2]]);
        } else if (shape.length == 4) {
            return from(new int[shape[0]][shape[1]][shape[2]][shape[3]]);
        } else if (shape.length == 5) {
            return from(new int[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]]);
        } else if (shape.length == 6) {
            return from(new int[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]]);
        } else if (shape.length == 7) {
            return from(new int[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]]);
        } else if (shape.length == 8) {
            return from(new int[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]]);
        } else if (shape.length == 9) {
            return from(new int[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]]);
        } else if (shape.length == 10) {
            return from(new int[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]][shape[9]]);
        }

        throw new NumpyOperationException("The shape specified is not supported; only arrays less than 10 dimensions are supported.");
    }

    /**
     * Create a numpy array of long with specified shape.
     *
     * @param shape The shape of the new array.
     * @return The new numpy array with desired shape.
     */
    public static NumpyArray<Long> arrayOfInt64WithShape(int[] shape) {
        if (shape.length == 1) {
            return from(new long[shape[0]]);
        } else if (shape.length == 2) {
            return from(new long[shape[0]][shape[1]]);
        } else if (shape.length == 3) {
            return from(new long[shape[0]][shape[1]][shape[2]]);
        } else if (shape.length == 4) {
            return from(new long[shape[0]][shape[1]][shape[2]][shape[3]]);
        } else if (shape.length == 5) {
            return from(new long[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]]);
        } else if (shape.length == 6) {
            return from(new long[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]]);
        } else if (shape.length == 7) {
            return from(new long[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]]);
        } else if (shape.length == 8) {
            return from(new long[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]]);
        } else if (shape.length == 9) {
            return from(new long[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]]);
        } else if (shape.length == 10) {
            return from(new long[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]][shape[9]]);
        }

        throw new NumpyOperationException("The shape specified is not supported; only arrays less than 10 dimensions are supported.");
    }

    /**
     * Create a numpy array of float with specified shape.
     *
     * @param shape The shape of the new array.
     * @return The new numpy array with desired shape.
     */
    public static NumpyArray<Float> arrayOfFloatWithShape(int[] shape) {
        if (shape.length == 1) {
            return from(new float[shape[0]]);
        } else if (shape.length == 2) {
            return from(new float[shape[0]][shape[1]]);
        } else if (shape.length == 3) {
            return from(new float[shape[0]][shape[1]][shape[2]]);
        } else if (shape.length == 4) {
            return from(new float[shape[0]][shape[1]][shape[2]][shape[3]]);
        } else if (shape.length == 5) {
            return from(new float[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]]);
        } else if (shape.length == 6) {
            return from(new float[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]]);
        } else if (shape.length == 7) {
            return from(new float[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]]);
        } else if (shape.length == 8) {
            return from(new float[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]]);
        } else if (shape.length == 9) {
            return from(new float[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]]);
        } else if (shape.length == 10) {
            return from(new float[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]][shape[9]]);
        }

        throw new NumpyOperationException("The shape specified is not supported; only arrays less than 10 dimensions are supported.");
    }

    /**
     * Create a numpy array of double with specified shape.
     *
     * @param shape The shape of the new array.
     * @return The new numpy array with desired shape.
     */
    public static NumpyArray<Double> arrayOfDoubleWithShape(int[] shape) {
        if (shape.length == 1) {
            return from(new double[shape[0]]);
        } else if (shape.length == 2) {
            return from(new double[shape[0]][shape[1]]);
        } else if (shape.length == 3) {
            return from(new double[shape[0]][shape[1]][shape[2]]);
        } else if (shape.length == 4) {
            return from(new double[shape[0]][shape[1]][shape[2]][shape[3]]);
        } else if (shape.length == 5) {
            return from(new double[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]]);
        } else if (shape.length == 6) {
            return from(new double[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]]);
        } else if (shape.length == 7) {
            return from(new double[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]]);
        } else if (shape.length == 8) {
            return from(new double[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]]);
        } else if (shape.length == 9) {
            return from(new double[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]]);
        } else if (shape.length == 10) {
            return from(new double[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]][shape[9]]);
        }

        throw new NumpyOperationException("The shape specified is not supported; only arrays less than 10 dimensions are supported.");
    }
}
