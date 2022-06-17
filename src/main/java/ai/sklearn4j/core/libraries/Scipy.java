package ai.sklearn4j.core.libraries;

import ai.sklearn4j.core.libraries.numpy.INumpyArrayElementOperation;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

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

    /**
     * Reshapes the array for supporting arithmetic.
     *
     * @param np Numpy array to be changed.
     * @return The changed NumpyArray.
     */
    private static NumpyArray<Double> to2DArrayShape(NumpyArray<Double> np) {
        double[] values = (double[]) np.getWrapper().getRawArray();
        double[][] result = new double[values.length][1];

        for (int i = 0; i < result.length; i++) {
            result[i][0] = values[i];
        }

        return NumpyArrayFactory.from(result);
//        int[] targetShape = new int[np.getShape().length + 1];
//        for (int i = 0; i < targetShape.length - 1; i++) {
//            targetShape[i] = np.getShape()[i];
//        }
//        targetShape[targetShape.length - 1] = 1;
//
//        NumpyArray<Double> result = NumpyArrayFactory.arrayOfDoubleWithShape(targetShape);
//
//        int[] counter = new int[targetShape.length+1];
//        counter[0] = -1;
//
//        do {
//            NumpyArray.addCounter(counter, targetShape);
//            int[] indexOnOutput = new int[targetShape.length];
//            int[] indexOnInput = new int[targetShape.length - 1];
//            for (int i = 0; i < indexOnOutput.length - 1; i++) {
//                indexOnOutput[i] = counter[i];
//                indexOnInput[i] = counter[i];
//            }
//
//            result.set(np.get(indexOnInput), indexOnOutput);
//        } while (counter[counter.length - 1] == 0);
//
//        return result;
    }
}
