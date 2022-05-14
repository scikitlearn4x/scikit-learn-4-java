package ai.sklearn4j.core.numpy;

import ai.sklearn4j.core.numpy.wrappers.*;

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
}
