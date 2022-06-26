package ai.sklearn4j.utils;

import ai.sklearn4j.core.ScikitLearnFeatureNotImplementedException;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

/**
 * Helper class that implements scikit-learn functionalities in utils/extmath.py.
 */
public class ExtMath {
    /**
     * Dot product of the NumpyArray.
     *
     * @param v1 Left-hand side of the expression.
     * @param v2 Right-hand side of the expression.
     * @return The dot product of the two numpy array.
     */
    public static NumpyArray<Double> dot(NumpyArray<Double> v1, NumpyArray<Double> v2) {
        if (v1.numberOfDimensions() == v2.numberOfDimensions() && v1.numberOfDimensions() == 2) {
            // Just do a regular matrix multiplication
            double[][] result = new double[v1.getShape()[0]][v2.getShape()[1]];

            for (int i = 0; i < v1.getShape()[0]; i++) {
                for (int j = 0; j < v2.getShape()[1]; j++) {
                    for (int k = 0; k < v1.getShape()[1]; k++) {
                        result[i][j] += v1.get(i, k) * v2.get(k, j);
                    }
                }
            }

            return NumpyArrayFactory.from(result);
        }

        throw new ScikitLearnFeatureNotImplementedException();
    }

    /**
     * Row-wise (squared) Euclidean norm of X.
     * Equivalent to np.sqrt((X * X).sum(axis=1))
     *
     * @param x The input array.
     * @return The row-wise (squared) Euclidean norm of x.
     */
    public static NumpyArray<Double> rowNorm(NumpyArray<Double> x) {
        return rowNorm(x, false);
    }

    /**
     * Row-wise (squared) Euclidean norm of X.
     * Equivalent to np.sqrt((X * X).sum(axis=1))
     *
     * @param x The input array.
     * @param squared If True, return squared norms.
     * @return The row-wise (squared) Euclidean norm of x.
     */
    public static NumpyArray<Double> rowNorm(NumpyArray<Double> x, boolean squared) {
        NumpyArray<Double> tmp = Numpy.multiply(x, x);
        tmp = Numpy.sum(tmp, 1, false);

        if (!squared) {
            tmp = Numpy.sqrt(tmp);
        }

        return tmp;
    }
}
