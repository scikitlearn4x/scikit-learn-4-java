package ai.sklearn4j.core.packaging.loaders.classifiers.naive_bayes;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.naive_bayes.ComplementNaiveBayes;

/**
 * ComplementNaiveBayes object loader.
 */
public class ComplementNaiveBayesContentLoader extends BaseScikitLearnContentLoader<ComplementNaiveBayes> {
    /**
     * Instantiate a new object of ComplementNaiveBayesContentLoader.
     */
    public ComplementNaiveBayesContentLoader() {
        super("nb_complement_serializer");
    }

    /**
     * Instantiate an unloaded ComplementNaiveBayes classifier.
     *
     * @return The unloaded classifier.
     */
    @Override
    protected ComplementNaiveBayes createResultObject() {
        return new ComplementNaiveBayes();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new ComplementNaiveBayesContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained classifier.
     */
    @Override
    protected void registerSetters() {
        registerNumpyArrayField("classes_", this::setClasses);
        registerNumpyArrayField("class_count_", this::setClassCount);
        registerNumpyArrayField("class_log_prior_", this::setClassLogPriors);
        registerNumpyArrayField("feature_log_prob_", this::setFeatureLogProbabilities);
        registerNumpyArrayField("feature_count_", this::setFeatureCount);
        registerNumpyArrayField("feature_all_", this::setFeatureAll);
    }

    /**
     * Sets the feature_all_ field.
     *
     * @param result The classifier to be loaded.
     * @param numpyArray The feature_all_ field.
     */
    private void setFeatureAll(ComplementNaiveBayes result, NumpyArray numpyArray) {

    }

    /**
     * Sets the feature's log probability in the training data.
     *
     * @param result The classifier to be loaded.
     * @param numpyArray The feature's log probability in the training data.
     */
    private void setFeatureLogProbabilities(ComplementNaiveBayes result, NumpyArray numpyArray) {
        result.setFeatureLogProbabilities(numpyArray);
    }

    /**
     * Sets the frequency of the features in the training data.
     *
     * @param result The classifier to be loaded.
     * @param numpyArray The frequency of the features in the training data.
     */
    private void setFeatureCount(ComplementNaiveBayes result, NumpyArray numpyArray) {
        result.setFeatureCount(numpyArray);
    }

    /**
     * Sets the probability of each class.
     *
     * @param result The classifier to be loaded.
     * @param value  The probability of each class.
     */
    private void setClassLogPriors(ComplementNaiveBayes result, NumpyArray value) {
        result.setClassLogPrior(value);
    }

    /**
     * Sets the class labels known to the classifier.
     *
     * @param result The classifier to be loaded.
     * @param value  The class labels known to the classifier.
     */
    private void setClasses(ComplementNaiveBayes result, NumpyArray value) {
        result.setClasses(value);
    }

    /**
     * Sets the number of training samples observed in each class.
     *
     * @param result The classifier to be loaded.
     * @param value  The number of training samples observed in each class.
     */
    private void setClassCount(ComplementNaiveBayes result, NumpyArray value) {
        result.setClassCounts(value);
    }
}