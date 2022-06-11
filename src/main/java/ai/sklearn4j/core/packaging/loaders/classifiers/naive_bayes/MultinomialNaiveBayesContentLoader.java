package ai.sklearn4j.core.packaging.loaders.classifiers.naive_bayes;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.naive_bayes.MultinomialNaiveBayes;

/**
 * MultinomialNaiveBayes object loader.
 */
public class MultinomialNaiveBayesContentLoader extends BaseScikitLearnContentLoader<MultinomialNaiveBayes> {
    /**
     * Instantiate a new object of MultinomialNaiveBayesContentLoader.
     */
    public MultinomialNaiveBayesContentLoader() {
        super("nb_multinomial_serializer");
    }

    /**
     * Instantiate an unloaded MultinomialNaiveBayes classifier.
     *
     * @return The unloaded classifier.
     */
    @Override
    protected MultinomialNaiveBayes createResultObject() {
        return new MultinomialNaiveBayes();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new MultinomialNaiveBayesContentLoader();
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
    }

    /**
     * Sets the feature's log probability in the training data.
     *
     * @param result The classifier to be loaded.
     * @param numpyArray The feature's log probability in the training data.
     */
    private void setFeatureLogProbabilities(MultinomialNaiveBayes result, NumpyArray numpyArray) {
        result.setFeatureLogProbabilities(numpyArray);
    }

    /**
     * Sets the frequency of the features in the training data.
     *
     * @param result The classifier to be loaded.
     * @param numpyArray The frequency of the features in the training data.
     */
    private void setFeatureCount(MultinomialNaiveBayes result, NumpyArray numpyArray) {
        result.setFeatureCount(numpyArray);
    }

    /**
     * Sets the probability of each class.
     *
     * @param result The classifier to be loaded.
     * @param value  The probability of each class.
     */
    private void setClassLogPriors(MultinomialNaiveBayes result, NumpyArray value) {
        result.setClassLogPrior(value);
    }

    /**
     * Sets the class labels known to the classifier.
     *
     * @param result The classifier to be loaded.
     * @param value  The class labels known to the classifier.
     */
    private void setClasses(MultinomialNaiveBayes result, NumpyArray value) {
        result.setClasses(value);
    }

    /**
     * Sets the number of training samples observed in each class.
     *
     * @param result The classifier to be loaded.
     * @param value  The number of training samples observed in each class.
     */
    private void setClassCount(MultinomialNaiveBayes result, NumpyArray value) {
        result.setClassCounts(value);
    }
}