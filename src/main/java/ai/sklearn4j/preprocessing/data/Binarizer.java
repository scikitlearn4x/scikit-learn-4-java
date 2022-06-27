// ==================================================================
// Inference for Binarizer
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
// ==================================================================
package ai.sklearn4j.preprocessing.data;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

/**
 * Binarize data (set feature values to 0 or 1) according to a
 * threshold.
 * Values greater than the threshold map to 1, while values less than or
 * equal to the threshold map to 0. With the default threshold of 0, only
 * positive values map to 1.
 * Binarization is a common operation on text count data where the
 * analyst can decide to only consider the presence or absence of a
 * feature rather than a quantified number of occurrences for instance.
 * It can also be used as a pre-processing step for estimators that
 * consider boolean random variables (e.g. modelled using the Bernoulli
 * distribution in a Bayesian setting).
 */
public class Binarizer extends TransformerMixin<NumpyArray<Double>, NumpyArray<Double>> {
    /**
     * Instantiate a new object of Binarizer.
     */
    public Binarizer() {

    }

    /**
     * Number of features seen during `fit`.
     */
    private long nFeaturesIn = 0;

    /**
     * Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     */
    private String[] featureNamesIn = null;

    private double threshold = 0.0;

    /**
     * Sets the Number of features seen during `fit`.
     *
     * @param value The new value for nFeaturesIn.
     */
    public void setNFeaturesIn(long value) {
        this.nFeaturesIn = value;
    }


    /**
     * Gets the Number of features seen during `fit`.
     */
    public long getNFeaturesIn() {
        return this.nFeaturesIn;
    }


    /**
     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     *
     * @param value The new value for featureNamesIn.
     */
    public void setFeatureNamesIn(String[] value) {
        this.featureNamesIn = value;
    }


    /**
     * Gets the Names of features seen during `fit`. Defined only when `X` has feature
     * names that are all strings.
     */
    public String[] getFeatureNamesIn() {
        return this.featureNamesIn;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double value) {
        this.threshold = value;
    }

    /**
     * Takes the input array and transforms it.
     *
     * @param array The array to transform.
     * @return The transformed array.
     */
    @Override
    public NumpyArray<Double> transform(NumpyArray<Double> array) {
        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array);

        array.applyToEachElementAnsSaveToTarget(result, value -> {
            if (value > threshold) {
                return 1.0;
            }

            return 0.0;
        });

        return result;
    }

    /**
     * Takes a transformed array and reveres the transformation.
     *
     * @param array The array to apply reveres transform.
     * @return The inversed transform of array.
     */
    @Override
    public NumpyArray<Double> inverseTransform(NumpyArray<Double> array) {
        throw new ScikitLearnCoreException("The inverse transform is not available for the Binarizer preprocessing.");
    }
}