package test.sklearn4j.core.libraries;

import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import org.junit.jupiter.api.Test;
import test.sklearn4j.TestHelper;

public class TestNumpy {
    @Test
    public void testExp() {
        double[] data = {0.285, 0.342, 0.763, 0.492, 0.962, 0.876, 0.487, 0.605, 0.507, 0.187, 0.051, 0.266, 0.619, 0.748, 0.78, 0.706, 0.881, 0.772, 0.022, 0.057};
        NumpyArray<Double> array = NumpyArrayFactory.from(data);

        NumpyArray<Double> actual = Numpy.exp(array);
        double[] expected = {1.32976203, 1.4077603, 2.14470068, 1.63558412, 2.61692509, 2.40127537, 1.62742661, 1.83125221, 1.66030281, 1.20562729, 1.05232289, 1.30473506, 1.85707004, 2.11277025, 2.18147227, 2.02587154, 2.41331181, 2.16409011, 1.02224378, 1.05865581};
        TestHelper.assertEqualData(actual, expected);
    }

    @Test
    public void testLog() {
        double[] data = {0.754, 0.635, 0.754, 0.453, 0.441};
        NumpyArray<Double> array = NumpyArrayFactory.from(data);

        NumpyArray<Double> actual = Numpy.log(array);
        double[] expected = {-0.28236291, -0.45413028, -0.28236291, -0.79186315, -0.8187104};
        TestHelper.assertEqualData(actual, expected);
    }

    @Test
    public void testAtLeast2D() {
        double vd = 10;
        double[][] vda = {{vd}};
        TestHelper.assertEqualData(Numpy.atLeast2D(vd), vda);

        float vf = 10;
        float[][] vfa = {{vf}};
        TestHelper.assertEqualData(Numpy.atLeast2D(vf), vfa);

        long vl = 10;
        long[][] vla = {{vl}};
        TestHelper.assertEqualData(Numpy.atLeast2D(vl), vla);

        int vi = 10;
        int[][] via = {{vi}};
        TestHelper.assertEqualData(Numpy.atLeast2D(vi), via);

        short vs = 10;
        short[][] vsa = {{vs}};
        TestHelper.assertEqualData(Numpy.atLeast2D(vs), vsa);

        byte vb = 10;
        byte[][] vba = {{vb}};
        TestHelper.assertEqualData(Numpy.atLeast2D(vb), vba);

        double[] array = {0, 1, 2, 3, 4, 5, 6};
        NumpyArray<Double> numpyArray = NumpyArrayFactory.from(array);

        TestHelper.assertEqualData(Numpy.atLeast2D(numpyArray), new double[][]{array});
    }

    @Test
    public void testNumpyArraySqueeze() {
        double[][][][][] a = {{{{{0.162, 0.533, 0.247, 0.386}}, {{0.608, 0.069, 0.434, 0.237}}}}, {{{{0.327, 0.803, 0.603, 0.499}}, {{0.765, 0.925, 0.715, 0.93}}}}, {{{{0.781, 0.037, 0.512, 0.693}}, {{0.771, 0.91, 0.029, 0.844}}}}};
        double[][][] expected = {{{0.162, 0.533, 0.247, 0.386}, {0.608, 0.069, 0.434, 0.237}}, {{0.327, 0.803, 0.603, 0.499}, {0.765, 0.925, 0.715, 0.93}}, {{0.781, 0.037, 0.512, 0.693}, {0.771, 0.91, 0.029, 0.844}}};

        NumpyArray<Double> numpyA = NumpyArrayFactory.from(a);
        NumpyArray<Double> numpySqueezed = Numpy.squeeze(numpyA);

        TestHelper.assertEqualData(numpySqueezed, expected);
    }
}
