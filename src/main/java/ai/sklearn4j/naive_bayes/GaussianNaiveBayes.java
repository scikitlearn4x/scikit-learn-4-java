package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.core.Constants;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.core.libraries.numpy.wrappers.Dim1DoubleNumpyWrapper;
import ai.sklearn4j.core.libraries.numpy.wrappers.Dim2DoubleNumpyWrapper;

import java.util.ArrayList;
import java.util.List;

public class GaussianNaiveBayes extends BaseNaiveBayes {
    /**
     * The frequency of each class in the training set.
     * Dimension: (class_count)
     */
    private NumpyArray<Double> classCounts = null;

    /**
     * The prior probability of each class.
     * Dimension: (class_count)
     */
    private NumpyArray<Double> classPriors = null;

    /**
     * The list of class IDs.
     * Dimension: (class_count)
     */
    private NumpyArray<Long> classes = null;

    /**
     * The user provided class priors. If specified, the priors are not adjusted according to the data.
     * Dimension: (class_count)
     */
    private NumpyArray<Double> priors = null;

    /**
     * Names of features seen during training. Defined only when `X` has feature names that are all strings.
     */
    private String[] featureNamesIn = null;

    /**
     * Number of features seen during training.
     */
    private int numberOfFeatures = 0;

    /**
     * Variance of each feature per class.
     * Dimension: (n_classes, n_features)
     */
    private NumpyArray<Double> sigma = null;

    /**
     * Mean of each feature per class.
     * Dimension: (n_classes, n_features)
     */
    private NumpyArray<Double> theta = null;
    private double varSmoothing = 1e-9;

    @Override
    protected NumpyArray<Double> jointLogLikelihood(NumpyArray<Double> x) {
        int count = x.getShape()[0];
        int classCount = classCounts.getShape()[0];
        int featureCount = sigma.getShape()[1];
        double[][] jointLogLikelihood = new double[count][classCount];

        double[][] variance = ((Dim2DoubleNumpyWrapper) sigma.getWrapper()).getArray();
        double[][] mean = ((Dim2DoubleNumpyWrapper) theta.getWrapper()).getArray();


        for (int cls = 0; cls < classCount; cls++) {
            double sumOfLogVariance = 0;

            for (int feature = 0; feature < featureCount; feature++) {
                sumOfLogVariance += Math.log(2.0 * Constants.PI * variance[cls][feature]);
            }

            double jointi = Math.log(classPriors.get(cls));

            for (int i = 0; i < count; i++) {
                double value = 0;

                for (int feature = 0; feature < featureCount; feature++) {
                    double diff = x.get(i, feature) - mean[cls][feature];
                    value += (Math.pow(x.get(i, feature) - mean[cls][feature], 2) / variance[cls][feature]);
                }

                value = -0.5 * (sumOfLogVariance + value);
                jointLogLikelihood[i][cls] = value + jointi;
            }
        }

        return NumpyArrayFactory.from(jointLogLikelihood);
    }

    @Override
    protected NumpyArray<Double> checkX(NumpyArray<Double> x) {
        return null;
    }
}
