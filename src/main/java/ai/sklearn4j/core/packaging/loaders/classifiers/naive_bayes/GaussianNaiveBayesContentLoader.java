package ai.sklearn4j.core.packaging.loaders.classifiers.naive_bayes;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.naive_bayes.GaussianNaiveBayes;

public class GaussianNaiveBayesContentLoader extends BaseScikitLearnContentLoader<GaussianNaiveBayes> {
    public GaussianNaiveBayesContentLoader() {
        super("nb_gaussian_serializer");
    }

    @Override
    protected GaussianNaiveBayes createResultObject() {
        return new GaussianNaiveBayes();
    }

    @Override
    public IScikitLearnContentLoader duplicate() {
        return new GaussianNaiveBayesContentLoader();
    }

    @Override
    protected void registerSetters() {
        registerDoubleField("epsilon_", this::setEpsilon);
        registerNumpyArrayField("class_count_", this::setClassCount);
        registerNumpyArrayField("classes_", this::setClasses);
        registerNumpyArrayField("class_prior_", this::setClassPriors);
        registerNumpyArrayField("theta_", this::setMeanValues);
        registerNumpyArrayField("var_", this::setVarianceValues);
        registerLongField("n_features_in_", this::setNumberOfFeatureIn);
        registerLongField("feature_names_in_", this::setFeaturesIn);
    }

    private void setFeaturesIn(GaussianNaiveBayes result, long value) {
        throw new RuntimeException();
    }

    private void setNumberOfFeatureIn(GaussianNaiveBayes result, long value) {
        result.setNumberOfFeatures((int)value);
    }

    private void setVarianceValues(GaussianNaiveBayes result, NumpyArray value) {
        result.setSigma(value);
    }

    private void setMeanValues(GaussianNaiveBayes result, NumpyArray value) {
        result.setTheta(value);
    }

    private void setClassPriors(GaussianNaiveBayes result, NumpyArray value) {
        result.setClassPriors(value);
    }

    private void setClasses(GaussianNaiveBayes result, NumpyArray value) {
        result.setClasses(value);
    }

    private void setClassCount(GaussianNaiveBayes result, NumpyArray value) {
        result.setClassCounts(value);
    }

    private void setEpsilon(GaussianNaiveBayes result, double value) {

    }
}
