package test.sklearn4j.core.libraries;

import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import test.sklearn4j.core.TestHelper;

public class TestNumpyArray {
    @Test
    public void testArgmax() {
        double[][][] data = {{{0.34207587, 0.59631829, 0.59525696, 0.95059543, 0.39912264}, {0.85464008, 0.90788009, 0.82057904, 0.43492519, 0.43897654}, {0.80624335, 0.80625582, 0.04637509, 0.21079158, 0.2956869}}, {{0.53870154, 0.54324711, 0.86099808, 0.21863662, 0.2951117}, {0.45954226, 0.1318496, 0.94744519, 0.95518557, 0.16132475}, {0.67286602, 0.2399391, 0.40854949, 0.8254005, 0.47404008}}};
        NumpyArray<Double> array = NumpyArrayFactory.from(data);

        NumpyArray<Integer> axis0 = Numpy.argmax(array, 0);
        int[][] axis0Expected = {{1, 0, 1, 0, 0}, {0, 0, 1, 1, 0,}, {0, 0, 1, 1, 1}};
        TestHelper.assertEqualData(axis0, axis0Expected);

        NumpyArray<Integer> axis1 = Numpy.argmax(array, 1);
        int[][] axis1Expected = {{1, 1, 1, 0, 1}, {2, 0, 1, 1, 2}};
        TestHelper.assertEqualData(axis1, axis1Expected);

        NumpyArray<Integer> axis2 = Numpy.argmax(array, 2);
        int[][] axis2Expected = {{3, 1, 1}, {2, 3, 3}};
        TestHelper.assertEqualData(axis2, axis2Expected);
    }

    @Test
    public void testSum() {
        double[][] data = {{0.817, 0.721, 0.67, 0.3}, {0.524, 0.935, 0.883, 0.866}, {0.134, 0.849, 0.578, 0.142}};

        NumpyArray<Double> array = NumpyArrayFactory.from(data);

        double[] axis0Expected = {1.475, 2.505, 2.131, 1.308};
        NumpyArray<Double> axis0 = Numpy.sum(array, 0);
        TestHelper.assertEqualData(axis0, axis0Expected);

        double[] axis1Expected = {2.508, 3.208, 1.703};
        NumpyArray<Double> axis1 = Numpy.sum(array, 1);
        TestHelper.assertEqualData(axis1, axis1Expected);
    }

    @Test
    public void testOneDimensionIntegerArray() {
        int dim1 = 10;
        int[] data = new int[dim1];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }

        NumpyArray<Integer> np = NumpyArrayFactory.from(data);
        Assertions.assertArrayEquals(new int[]{dim1}, np.getShape());

        for (int i = 0; i < data.length; i++) {
            Assertions.assertEquals(i, np.get(i));
        }

        np.applyToEachElement((value) -> value * 2);

        for (int i = 0; i < data.length; i++) {
            Assertions.assertEquals(i * 2, np.get(i));
        }
    }

    @Test
    public void testTwoDimensionIntegerArray() {
        int dim1 = 10;
        int dim2 = 5;
        int[][] data = new int[dim1][dim2];
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                data[i][j] = i * dim2 + j;
            }
        }

        NumpyArray<Integer> np = NumpyArrayFactory.from(data);
        Assertions.assertArrayEquals(new int[]{dim1, dim2}, np.getShape());

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                Assertions.assertEquals(i * dim2 + j, np.get(i, j));
            }
        }

        np.applyToEachElement((value) -> value * 2);

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                Assertions.assertEquals(2 * (i * dim2 + j), np.get(i, j));
            }
        }

    }

    @Test
    public void testOneDimensionDoubleArray() {
        int dim1 = 10;
        double[] data = new double[dim1];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }

        NumpyArray<Double> np = NumpyArrayFactory.from(data);
        Assertions.assertArrayEquals(new int[]{dim1}, np.getShape());

        for (int i = 0; i < data.length; i++) {
            Assertions.assertEquals(i, np.get(i));
        }

        np.applyToEachElement((value) -> value * 2);

        for (int i = 0; i < data.length; i++) {
            Assertions.assertEquals(i * 2, np.get(i));
        }
    }

    @Test
    public void testTwoDimensionDoubleArray() {
        int dim1 = 10;
        int dim2 = 5;
        double[][] data = new double[dim1][dim2];
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                data[i][j] = i * dim2 + j;
            }
        }

        NumpyArray<Double> np = NumpyArrayFactory.from(data);
        Assertions.assertArrayEquals(new int[]{dim1, dim2}, np.getShape());

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                Assertions.assertEquals(i * dim2 + j, np.get(i, j));
            }
        }

        np.applyToEachElement((value) -> value * 2);

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                Assertions.assertEquals(2 * (i * dim2 + j), np.get(i, j));
            }
        }

    }
}
