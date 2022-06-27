package ai.sklearn4j.core.libraries;

import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Function of the scipy library that was used in scikit-learn.
 */
public class Scipy {
    /**
     * Compute the log of the sum of exponentials of input elements.
     * See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
     *
     * @param data Input array to apply the calculations on.
     * @param axis The axis used by sum for reduction.
     * @return The result, np.log(np.sum(np.exp(a))) calculated in a numerically more stable way.
     */
    public static NumpyArray<Double> logSumExponent(NumpyArray<Double> data, int axis) {
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        // Calculates np.log(np.sum(np.exp(a)))
//        return Numpy.log(Numpy.sum(Numpy.exp(data), axis));
        NumpyArray<Double> aMax = Numpy.arrayMax(data, axis, true);
        aMax.applyToEachElement(value -> {
            if (!Double.isFinite(value)) {
                return 0.0;
            } else {
                return value;
            }
        });

        NumpyArray<Double> tmp = Numpy.exp(Numpy.subtract(data, aMax));
        tmp = Numpy.sum(tmp, axis, false);
        tmp = Numpy.log(tmp);
        tmp = Numpy.add(tmp, Numpy.squeeze(aMax));

        return tmp;
    }

    public static final class NormalDistribution {
        public static double[] cumulativeDistributionFunction(double[] x) {
            throw new RuntimeException();
        }
    }

}
