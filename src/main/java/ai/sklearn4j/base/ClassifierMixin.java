package ai.sklearn4j.base;

/**
 * Mixin class for all classifiers in scikit-learn.
 */
public abstract class ClassifierMixin extends BaseEstimator {
    public String getEstimatorType() {
        return "classifier";
    }
}
