package ai.sklearn4j.core.packaging.loaders.classifiers.naive_bayes;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.naive_bayes.GaussianNaiveBayes;

/**
 * GaussianNaiveBayes object loader.
 */
public class GaussianNaiveBayesContentLoader extends BaseScikitLearnContentLoader<GaussianNaiveBayes> {
    /**
     * Instantiate a new object of GaussianNaiveBayesContentLoader.
     */
    public GaussianNaiveBayesContentLoader() {
        super("nb_gaussian_serializer");
    }

    /**
     * Instantiate an unloaded GaussianNaiveBayes classifier.
     *
     * @return The unloaded classifier.
     */
    @Override
    protected GaussianNaiveBayes createResultObject() {
        return new GaussianNaiveBayes();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new GaussianNaiveBayesContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained classifier.
     */
    @Override
    protected void registerSetters() {
        registerNumpyArrayField("class_count_", this::setClassCount);
        registerNumpyArrayField("classes_", this::setClasses);
        registerNumpyArrayField("class_prior_", this::setClassPriors);
        registerNumpyArrayField("theta_", this::setMeanValues);
        registerNumpyArrayField("var_", this::setVarianceValues);
        registerLongField("n_features_in_", this::setNumberOfFeatureIn);
        registerStringArrayField("feature_names_in_", this::setFeaturesIn);
    }

    /**
     * Sets the list of features names' of the dataset the model was trained on.
     *
     * @param result The classifier to be loaded.
     * @param value  The list of feature names.
     */
    private void setFeaturesIn(GaussianNaiveBayes result, String[] value) {
        result.setFeatureNamesIn(value);
    }

    /**
     * Sets the number of features of the dataset the model was trained on.
     *
     * @param result The classifier to be loaded.
     * @param value  The number of features.
     */
    private void setNumberOfFeatureIn(GaussianNaiveBayes result, long value) {
        result.setNumberOfFeatures((int) value);
    }

    /**
     * Sets the variance of each feature per class.
     *
     * @param result The classifier to be loaded.
     * @param value  The variance of each feature per class.
     */
    private void setVarianceValues(GaussianNaiveBayes result, NumpyArray value) {
        result.setSigma(value);
    }

    /**
     * Sets the mean of each feature per class.
     *
     * @param result The classifier to be loaded.
     * @param value  The mean of each feature per class.
     */
    private void setMeanValues(GaussianNaiveBayes result, NumpyArray value) {
        result.setTheta(value);
    }

    /**
     * Sets the probability of each class.
     *
     * @param result The classifier to be loaded.
     * @param value  The probability of each class.
     */
    private void setClassPriors(GaussianNaiveBayes result, NumpyArray value) {
        result.setClassPriors(value);
    }

    /**
     * Sets the class labels known to the classifier.
     *
     * @param result The classifier to be loaded.
     * @param value  The class labels known to the classifier.
     */
    private void setClasses(GaussianNaiveBayes result, NumpyArray value) {
        result.setClasses(value);
    }

    /**
     * Sets the number of training samples observed in each class.
     *
     * @param result The classifier to be loaded.
     * @param value  The number of training samples observed in each class.
     */
    private void setClassCount(GaussianNaiveBayes result, NumpyArray value) {
        result.setClassCounts(value);
    }
}
