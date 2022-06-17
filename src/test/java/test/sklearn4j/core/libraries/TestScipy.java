package test.sklearn4j.core.libraries;

import ai.sklearn4j.core.libraries.Scipy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import org.junit.jupiter.api.Test;
import test.sklearn4j.TestHelper;

public class TestScipy {
    @Test
    public void testLogSumExponent() {
        double[][][] data = {{{0.739, 0.008, 0.367}, {0.509, 0.031, 0.262}, {0.628, 0.572, 0.441}, {0.09, 0.754, 0.565}, {0.734, 0.135, 0.79}, {0.436, 0.709, 0.064}}, {{0.78, 0.252, 0.057}, {0.541, 0.7, 0.32}, {0.6, 0.372, 0.033}, {0.96, 0.872, 0.5}, {0.453, 0.795, 0.235}, {0.227, 0.797, 0.113}}, {{0.78, 0.998, 0.858}, {0.393, 0.068, 0.459}, {0.077, 0.47, 0.191}, {0.488, 0.31, 0.185}, {0.464, 0.342, 0.166}, {0.95, 0.394, 0.887}}, {{0.099, 0.945, 0.478}, {0.668, 0.643, 0.227}, {0.256, 0.84, 0.014}, {0.043, 0.368, 0.322}, {0.998, 0.175, 0.979}, {0.953, 0.116, 0.116}}};

        NumpyArray<Double> array = NumpyArrayFactory.from(data);

        double[][] axis0Expected = {{2.02298381, 2.02555175, 1.86769194}, {1.91884747, 1.79471882, 1.70728576}, {1.8029916, 1.96554487, 1.57118951}, {1.85298995, 1.99135344, 1.79030379}, {2.07439073, 1.78473222, 1.98933512}, {2.07700442, 1.92523051, 1.74685893}};
        NumpyArray<Double> axis0 = Scipy.logSumExponent(array, 0);
        TestHelper.assertEqualData(axis0, axis0Expected);

        double[][] axis1Expected = {{2.33750881, 2.20973541, 2.23281926}, {2.41241349, 2.44893282, 2.01524911}, {2.35594766, 2.26564059, 2.29887984}, {2.36990899, 2.35616788, 2.20237215}};
        NumpyArray<Double> axis1 = Scipy.logSumExponent(array, 1);
        TestHelper.assertEqualData(axis1, axis1Expected);

        double[][] axis2Expected = {{1.51408727, 1.38495316, 1.64864458, 1.60525057, 1.69218418, 1.53569653}, {1.50999852, 1.63092622, 1.46007961, 1.89493368, 1.61985822, 1.52460658}, {1.98138237, 1.41931184, 1.35853658, 1.43405981, 1.43001368, 1.87125517}, {1.66576629, 1.63067901, 1.53087471, 1.35290941, 1.88188807, 1.57680525}};
        NumpyArray<Double> axis2 = Scipy.logSumExponent(array, 2);
        TestHelper.assertEqualData(axis2, axis2Expected);
    }
}
