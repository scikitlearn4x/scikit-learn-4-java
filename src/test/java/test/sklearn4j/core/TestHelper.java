package test.sklearn4j.core;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import org.junit.jupiter.api.Assertions;

public class TestHelper {

    public static final double DOUBLE_COMPARE_EPSILON = 0.0000001;

    public static void assertEqualData(NumpyArray<Integer> numpyArray, int[][] array) {
        Assertions.assertEquals(2, numpyArray.getShape().length);
        Assertions.assertEquals(array.length, numpyArray.getShape()[0]);
        Assertions.assertEquals(array[0].length, numpyArray.getShape()[1]);

        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                Assertions.assertEquals(array[i][j], numpyArray.get(i, j));
            }
        }
    }

    public static void assertEqualData(NumpyArray<Double> numpyArray, double[][] array) {
        Assertions.assertEquals(2, numpyArray.getShape().length);
        Assertions.assertEquals(array.length, numpyArray.getShape()[0]);
        Assertions.assertEquals(array[0].length, numpyArray.getShape()[1]);

        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                double diff = Math.abs(array[i][j] - numpyArray.get(i, j));
                boolean check = diff < DOUBLE_COMPARE_EPSILON;

                Assertions.assertTrue(check);
            }
        }
    }

    public static void assertEqualData(NumpyArray<Double> numpyArray, double[] array) {
        Assertions.assertEquals(1, numpyArray.getShape().length);
        Assertions.assertEquals(array.length, numpyArray.getShape()[0]);

        for (int i = 0; i < array.length; i++) {
            double diff = Math.abs(array[i] - numpyArray.get(i));
            boolean check = diff < DOUBLE_COMPARE_EPSILON;

            Assertions.assertTrue(check);
        }
    }
}
