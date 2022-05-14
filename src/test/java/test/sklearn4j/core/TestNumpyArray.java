package test.sklearn4j.core;

import ai.sklearn4j.core.numpy.NumpyArray;
import ai.sklearn4j.core.numpy.NumpyArrayFactory;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestNumpyArray {
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
