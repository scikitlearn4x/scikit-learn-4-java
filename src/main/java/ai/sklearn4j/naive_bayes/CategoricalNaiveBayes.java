package ai.sklearn4j.naive_bayes;

import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

import java.util.List;

/**
 * Naive Bayes classifier for categorical features.
 * <p>
 * The categorical Naive Bayes classifier is suitable for classification with discrete features that
 * are categorically distributed. The categories of each feature are drawn from a categorical
 * distribution.
 */
public class CategoricalNaiveBayes extends BaseNaiveBayes {
    /**
     * Empirical log probability of features given a class, P(x_i|y).
     */
    private List<NumpyArray<Double>> featureLogProbabilities = null;

    /**
     * Log probability of each class (smoothed).
     */
    private NumpyArray<Double> classLogPrior = null;

    /**
     * Holds arrays of shape (n_classes, n_categories of respective feature) for each feature.
     * Each array provides the number of samples encountered for each class and category of the
     * specific feature.
     */
    private NumpyArray<Double> categoryCounts = null;

    /**
     * Number of categories for each feature. This value is inferred from the data or set by the
     * minimum number of categories.
     */
    private NumpyArray<Long> numberOfCategories = null;

    /**
     * Compute the unnormalized posterior log probability of X.
     * <p>
     * I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of shape
     * (n_samples, n_classes).
     * <p>
     * predict, predictProbabilities, and predictLogProbabilities pass the input over to
     * jointLogLikelihood.
     *
     * @param x An array-like of shape (n_samples, n_classes).
     */
    @Override
    protected NumpyArray<Double> jointLogLikelihood(NumpyArray<Double> x) {
        NumpyArray<Double> jll = NumpyArrayFactory.arrayOfDoubleWithShape(new int[]{x.getShape()[0], classCounts.getShape()[0]});
        for (int i = 0; i < getNumberOfFeatures(); i++) {
            int[] indices = getArrayFirstDimension(x, i);
            NumpyArray<Double> logProb = this.featureLogProbabilities.get(i);
            int classCount = classes.getShape()[0];
            double[][] temp = new double[classCount][indices.length];

            for (int cls = 0; cls < classCount; cls++) {
                for (int j = 0; j < indices.length; j++) {
                    temp[cls][j] = logProb.get(cls, indices[j]);
                }
            }

            NumpyArray<Double> t = NumpyArrayFactory.from(temp).transpose();
            jll = Numpy.add(jll, t);
        }

        return Numpy.add(jll, classLogPrior);
    }

    /**
     * Gets the values of the first dimension. Equivalent to numpy data[:, secondDimensionIndex]
     *
     * @param x                    Array to be sliced.
     * @param secondDimensionIndex The value of the second dimension.
     * @return The sliced first dimension.
     */
    private int[] getArrayFirstDimension(NumpyArray<Double> x, int secondDimensionIndex) {
        int[] indices = new int[x.getShape()[0]];

        for (int j = 0; j < indices.length; j++) {
            double value = x.get(j, secondDimensionIndex);
            indices[j] = (int) value;
        }

        return indices;
    }

    /**
     * Gets the log probability of each class (smoothed).
     *
     * @return Log probability of each class (smoothed).
     */
    public NumpyArray<Double> getClassLogPrior() {
        return classLogPrior;
    }

    /**
     * Sets the log probability of each class (smoothed).
     *
     * @param classLogPrior The log probability of each class (smoothed).
     */
    public void setClassLogPrior(NumpyArray<Double> classLogPrior) {
        this.classLogPrior = classLogPrior;
    }

    /**
     * Gets the empirical log probability of features given a class, P(x_i|y).
     *
     * @return Empirical log probability of features given a class, P(x_i|y).
     */
    public List<NumpyArray<Double>> getFeatureLogProbabilities() {
        return featureLogProbabilities;
    }

    /**
     * Sets the empirical log probability of features given a class, P(x_i|y).
     *
     * @param featureLogProbabilities The empirical log probability of features given a class, P(x_i|y).
     */
    public void setFeatureLogProbabilities(List<NumpyArray<Double>> featureLogProbabilities) {
        this.featureLogProbabilities = featureLogProbabilities;
    }

}
