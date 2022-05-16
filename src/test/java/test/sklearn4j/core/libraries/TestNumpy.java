package test.sklearn4j.core.libraries;

import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import org.junit.jupiter.api.Test;
import test.sklearn4j.core.TestHelper;

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
}
