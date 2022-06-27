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

    @Test
    public void testNumpyArrayAbsoluteValue() {
        double[][][] a = {{{2.2021807376623004, -2.371934168187786, -1.2303305322177822}, {-1.4145853395562886, 2.2189727100067644, 1.3616225507706203}, {1.650369009667708, -1.5138657258918848, 0.9495624707581896}}, {{0.6942375411280279, 0.7031752340165349, 1.1236726833703594}, {-1.2088253170059255, -2.3731534212128795, 2.201735303066867}, {2.0747940248392025, -1.8719669542142596, -1.2139129447818109}}, {{2.471844613158825, 2.436473690209085, -1.8835979673451464}, {1.8858781531861348, -1.6891113483514886, 1.207727197330739}, {-2.0304249021758936, -2.299433004919129, 1.0798879246898574}}};
        double[][][] expected = {{{2.2021807376623004, 2.371934168187786, 1.2303305322177822}, {1.4145853395562886, 2.2189727100067644, 1.3616225507706203}, {1.650369009667708, 1.5138657258918848, 0.9495624707581896}}, {{0.6942375411280279, 0.7031752340165349, 1.1236726833703594}, {1.2088253170059255, 2.3731534212128795, 2.201735303066867}, {2.0747940248392025, 1.8719669542142596, 1.2139129447818109}}, {{2.471844613158825, 2.436473690209085, 1.8835979673451464}, {1.8858781531861348, 1.6891113483514886, 1.207727197330739}, {2.0304249021758936, 2.299433004919129, 1.0798879246898574}}};

        NumpyArray<Double> numpyA = NumpyArrayFactory.from(a);
        NumpyArray<Double> numpyAbs = Numpy.abs(numpyA);

        TestHelper.assertEqualData(numpyAbs, expected);
    }

    @Test
    public void testNumpyArraySqrt() {
        double[][][] array = {{{4.760063093516445, 3.8439497991784006, 3.0776809453102056}, {3.4150132636395414, 2.5300637455269968, 1.9771414541525516}, {2.571432141751086, 0.4394586875508866, 3.9371745731239183}}, {{1.3961357125162492, 0.9633971942731845, 2.6167689528536187}, {1.9842424198632547, 2.9684580191651815, 0.18820704490001283}, {1.749505002489724, 3.388219138964079, 4.5699733281820105}}, {{4.9142242617244385, 1.1983514385822958, 2.527980567253314}, {3.978316200072288, 0.006321589675757355, 1.1202633899625374}, {3.5866022929535344, 4.443235848376891, 2.8623522069961993}}};
        double[][][] expected = {{{2.1817568823121527, 1.9605993469289946, 1.754332051041138}, {1.847975449955854, 1.5906174101671957, 1.406108621036281}, {1.6035685647177942, 0.6629168028877278, 1.9842314817389421}}, {{1.1815818687320185, 0.9815279895515892, 1.6176430239251238}, {1.4086313995730944, 1.7229213618633852, 0.4338283588010503}, {1.3226885508273383, 1.8407115849486249, 2.1377495943589864}}, {{2.2168049670019325, 1.09469239450281, 1.589962442088905}, {1.9945716833626932, 0.07950842518725519, 1.0584249571710491}, {1.8938326993041212, 2.107898443563373, 1.6918487541728424}}};

        NumpyArray<Double> numpyA = NumpyArrayFactory.from(array);
        NumpyArray<Double> numpySqrt = Numpy.sqrt(numpyA);

        TestHelper.assertEqualData(numpySqrt, expected);
    }
}
