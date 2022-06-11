package ai.sklearn4j.core.packaging.loaders.classifiers.naive_bayes;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.naive_bayes.CategoricalNaiveBayes;

import java.util.List;

/**
 * CategoricalNaiveBayes object loader.
 */
public class CategoricalNaiveBayesContentLoader extends BaseScikitLearnContentLoader<CategoricalNaiveBayes> {
    /**
     * Instantiate a new object of CategoricalNaiveBayesContentLoader.
     */
    public CategoricalNaiveBayesContentLoader() {
        super("nb_categorical_serializer");
    }

    /**
     * Instantiate an unloaded CategoricalNaiveBayes classifier.
     *
     * @return The unloaded classifier.
     */
    @Override
    protected CategoricalNaiveBayes createResultObject() {
        return new CategoricalNaiveBayes();
    }

    /**
     * Create a clean instance of the loader.
     *
     * @return A clean instance of the loader.
     */
    @Override
    public IScikitLearnContentLoader duplicate() {
        return new CategoricalNaiveBayesContentLoader();
    }

    /**
     * Defines the fields that are required to initialize a trained classifier.
     */
    @Override
    protected void registerSetters() {
        registerNumpyArrayField("classes_", this::setClasses);
        registerNumpyArrayField("class_count_", this::setClassCount);
        registerNumpyArrayField("class_log_prior_", this::setClassLogPriors);
        registerListOfNumpyArrayField("feature_log_prob_", this::setFeatureLogProbabilities);
        registerListOfNumpyArrayField("category_count_", this::setCategoryCounts);
        registerNumpyArrayField("n_categories_", this::setNumberInCategories);
    }

    /**
     * Sets the field n_categories_ in the classifier.
     *
     * @param result The classifier to be loaded.
     * @param numpyArray The n in each category.
     */
    private void setNumberInCategories(CategoricalNaiveBayes result, NumpyArray numpyArray) {

    }

    /**
     * Sets the category_count_ in the classifier.
     * @param result The classifier to be loaded.
     * @param numpyArrays The categories count in the training data.
     */
    private void setCategoryCounts(CategoricalNaiveBayes result, List<NumpyArray<Double>> numpyArrays) {

    }

    /**
     * Sets the feature's log probability in the training data.
     *
     * @param result The classifier to be loaded.
     * @param numpyArray The feature's log probability in the training data.
     */
    private void setFeatureLogProbabilities(CategoricalNaiveBayes result, List<NumpyArray<Double>> numpyArray) {
        result.setFeatureLogProbabilities(numpyArray);
    }

    /**
     * Sets the probability of each class.
     *
     * @param result The classifier to be loaded.
     * @param value  The probability of each class.
     */
    private void setClassLogPriors(CategoricalNaiveBayes result, NumpyArray value) {
        result.setClassLogPrior(value);
    }

    /**
     * Sets the class labels known to the classifier.
     *
     * @param result The classifier to be loaded.
     * @param value  The class labels known to the classifier.
     */
    private void setClasses(CategoricalNaiveBayes result, NumpyArray value) {
        result.setClasses(value);
    }

    /**
     * Sets the number of training samples observed in each class.
     *
     * @param result The classifier to be loaded.
     * @param value  The number of training samples observed in each class.
     */
    private void setClassCount(CategoricalNaiveBayes result, NumpyArray value) {
        result.setClassCounts(value);
    }
}