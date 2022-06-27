package ai.sklearn4j.utils;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

/**
 * Utils method that scikit-learn provide to preprocess the data.
 */
public class Preprocessing {
    /**
     * Binarize a numpy array based on a given threshold.
     *
     * @param x         Array to be binarized.
     * @param threshold The threshold for binarization. If the value in x is greater than threshold,
     *                  the target element is 1.0 otherwise is 0.0.
     * @return The binarized numpy array.
     */
    public static NumpyArray<Double> binarizeInput(NumpyArray<Double> x, double threshold) {
        NumpyArray<Double> result = NumpyArrayFactory.arrayOfDoubleWithShape(x.getShape());

        x.applyToEachElementAnsSaveToTarget(result, value -> {
            if (value > threshold) {
                return 1.0;
            }

            return 0.0;
        });

        return result;
    }
}
