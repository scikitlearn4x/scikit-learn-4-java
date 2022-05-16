package ai.sklearn4j.core.libraries;

import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;

public class Scipy {
    public static NumpyArray<Double> logSumExponent(NumpyArray<Double> data, int axis) {
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        // Calculates np.log(np.sum(np.exp(a)))
        return Numpy.log(Numpy.sum(Numpy.exp(data), axis));
    }
}
