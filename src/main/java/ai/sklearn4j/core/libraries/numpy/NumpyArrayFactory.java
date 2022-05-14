package ai.sklearn4j.core.libraries.numpy;

import ai.sklearn4j.core.libraries.numpy.wrappers.*;

public class NumpyArrayFactory {
    public static NumpyArray<Byte> from(byte[] array) {
        return new NumpyArray<>(new Dim1Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][] array) {
        return new NumpyArray<>(new Dim2Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][][] array) {
        return new NumpyArray<>(new Dim3Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][][][] array) {
        return new NumpyArray<>(new Dim4Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][][][][] array) {
        return new NumpyArray<>(new Dim5Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][][][][][] array) {
        return new NumpyArray<>(new Dim6Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][][][][][][] array) {
        return new NumpyArray<>(new Dim7Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9Int8NumpyWrapper(array));
    }

    public static NumpyArray<Byte> from(byte[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10Int8NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[] array) {
        return new NumpyArray<>(new Dim1Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][] array) {
        return new NumpyArray<>(new Dim2Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][][] array) {
        return new NumpyArray<>(new Dim3Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][][][] array) {
        return new NumpyArray<>(new Dim4Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][][][][] array) {
        return new NumpyArray<>(new Dim5Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][][][][][] array) {
        return new NumpyArray<>(new Dim6Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][][][][][][] array) {
        return new NumpyArray<>(new Dim7Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9Int16NumpyWrapper(array));
    }

    public static NumpyArray<Short> from(short[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10Int16NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[] array) {
        return new NumpyArray<>(new Dim1Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][] array) {
        return new NumpyArray<>(new Dim2Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][][] array) {
        return new NumpyArray<>(new Dim3Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][][][] array) {
        return new NumpyArray<>(new Dim4Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][][][][] array) {
        return new NumpyArray<>(new Dim5Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][][][][][] array) {
        return new NumpyArray<>(new Dim6Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][][][][][][] array) {
        return new NumpyArray<>(new Dim7Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9Int32NumpyWrapper(array));
    }

    public static NumpyArray<Integer> from(int[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10Int32NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[] array) {
        return new NumpyArray<>(new Dim1Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][] array) {
        return new NumpyArray<>(new Dim2Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][][] array) {
        return new NumpyArray<>(new Dim3Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][][][] array) {
        return new NumpyArray<>(new Dim4Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][][][][] array) {
        return new NumpyArray<>(new Dim5Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][][][][][] array) {
        return new NumpyArray<>(new Dim6Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][][][][][][] array) {
        return new NumpyArray<>(new Dim7Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9Int64NumpyWrapper(array));
    }

    public static NumpyArray<Long> from(long[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10Int64NumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[] array) {
        return new NumpyArray<>(new Dim1FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][] array) {
        return new NumpyArray<>(new Dim2FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][][] array) {
        return new NumpyArray<>(new Dim3FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][][][] array) {
        return new NumpyArray<>(new Dim4FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][][][][] array) {
        return new NumpyArray<>(new Dim5FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][][][][][] array) {
        return new NumpyArray<>(new Dim6FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][][][][][][] array) {
        return new NumpyArray<>(new Dim7FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9FloatNumpyWrapper(array));
    }

    public static NumpyArray<Float> from(float[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10FloatNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[] array) {
        return new NumpyArray<>(new Dim1DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][] array) {
        return new NumpyArray<>(new Dim2DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][][] array) {
        return new NumpyArray<>(new Dim3DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][][][] array) {
        return new NumpyArray<>(new Dim4DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][][][][] array) {
        return new NumpyArray<>(new Dim5DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][][][][][] array) {
        return new NumpyArray<>(new Dim6DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][][][][][][] array) {
        return new NumpyArray<>(new Dim7DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][][][][][][][] array) {
        return new NumpyArray<>(new Dim8DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][][][][][][][][] array) {
        return new NumpyArray<>(new Dim9DoubleNumpyWrapper(array));
    }

    public static NumpyArray<Double> from(double[][][][][][][][][][] array) {
        return new NumpyArray<>(new Dim10DoubleNumpyWrapper(array));
    }

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

        throw new RuntimeException();
    }

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

        throw new RuntimeException();
    }

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

        throw new RuntimeException();
    }

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

        throw new RuntimeException();
    }

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

        throw new RuntimeException();
    }

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

        throw new RuntimeException();
    }
}
