// ==================================================================
// Inference for Normalizer
// 
// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
// ==================================================================
package ai.sklearn4j.preprocessing.data;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.libraries.numpy.Numpy;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.utils.ExtMath;

/**
 * Normalize samples individually to unit norm.
 * Each sample (i.e. each row of the data matrix) with at least one non
 * zero component is rescaled independently of other samples so that its
 * norm (l1, l2 or inf) equals one.
 * This transformer is able to work both with dense numpy arrays and
 * scipy.sparse matrix (use CSR format if you want to avoid the burden of
 * a copy / conversion).
 * Scaling inputs to unit norms is a common operation for text
 * classification or clustering for instance. For instance the dot
 * product of two l2-normalized TF-IDF vectors is the cosine similarity
 * of the vectors and is the base similarity metric for the Vector Space
 * Model commonly used by the Information Retrieval community.
 */
public class Normalizer extends TransformerMixin<NumpyArray<Double>, NumpyArray<Double>> {
    /**
     * Instantiate a new object of Normalizer.
     */
    public Normalizer() {

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

    private String norm = null;

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

    public String getNorm() {
        return norm;
    }

    public void setNorm(String norm) {
        this.norm = norm;
    }

    @Override
    public NumpyArray<Double> transform(NumpyArray<Double> array) {
//        if norm == "l1":
//            norms = np.abs(X).sum(axis=1)
//        elif norm == "l2":
//            norms = row_norms(X)
//        elif norm == "max":
//            norms = np.max(abs(X), axis=1)
//        norms = _handle_zeros_in_scale(norms, copy=False)
//        X /= norms[:, np.newaxis]
        NumpyArray<Double> result = null;
        NumpyArray<Double> norms = null;

        if ("l1".equals(norm)) {
            norms = Numpy.sum(Numpy.abs(array), 1, false);
        } else if ("l2".equals(norm)) {
            norms = ExtMath.rowNorm(array);
        } else if ("max".equals(norm)) {
            norms = Numpy.arrayMax(Numpy.abs(array), 1, false);
        }

        handleZerosInScale(norms);
        result = Numpy.divide(array, addTrailingOneDimension(norms));
        return result;
    }

    private NumpyArray<Double> addTrailingOneDimension(NumpyArray<Double> norms) {
        double[][] result = new double[norms.getShape()[0]][1];

        for (int i = 0; i < result.length; i++) {
            result[i][0] = norms.get(i);
        }

        return NumpyArrayFactory.from(result);
    }

    /**
     * Set scales of near constant features to 1.
     *
     * The goal is to avoid division by very small or zero values.
     *
     * Near constant features are detected automatically by identifying scales close to machine
     * precision unless they are precomputed by the caller and passed with the `constant_mask` kwarg.
     *
     * Typically for standard scaling, the scales are the standard deviation while near constant
     * features are better detected on the computed variances which are closer to machine precision
     * by construction.
     *
     * @param array The array to normalize the zeros.
     */
    private void handleZerosInScale(NumpyArray<Double> array) {
        double epsilon = 2.220446049250313e-16; // np.finfo(np.float64).eps
        final double threshold = 10 * epsilon;

        array.applyToEachElement(value -> {
            if (value < threshold) {
                return 1.0;
            }

            return value;
        });
    }

    @Override
    public NumpyArray<Double> inverseTransform(NumpyArray<Double> array) {
        throw new ScikitLearnCoreException("The inverse transform is not available for the Normalizer preprocessing.");
    }
}