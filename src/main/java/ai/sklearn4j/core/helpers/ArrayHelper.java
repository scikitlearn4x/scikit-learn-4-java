package ai.sklearn4j.core.helpers;

import ai.sklearn4j.core.NumpyArray;

public class ArrayHelper {
    public static int[] getArgmaxFromClassProbabilityDistribution(NumpyArray<Double> array) {
        int[] argmax = new int[array.getShape()[0]];

        for (int i = 0; i < array.getShape()[0]; i++) {
            double max = array.get(i, 0);
            argmax[i] = 0;

            for (int j = 1; j < array.getShape()[1]; j++) {
                double current = array.get(i, j);
                if (current > max) {
                    max = current;
                    argmax[i] = j;
                }
            }
        }

        return argmax;
    }

    public static NumpyArray<Double> exponentOfLogProbabilities(NumpyArray<Double> logProbabilities) {
        NumpyArray<Double> result = NumpyArray.withShape(logProbabilities.getShape());

        result.applyToEachElement(Math::exp);

        return result;
    }
}
