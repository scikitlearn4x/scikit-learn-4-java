//// ==================================================================
//// Deserialize QuantileTransformer
////
//// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer
//// ==================================================================
//package ai.sklearn4j.core.packaging.loaders.preprocessing.data;
//
//import ai.sklearn4j.core.libraries.numpy.NumpyArray;
//import ai.sklearn4j.core.packaging.loaders.BaseScikitLearnContentLoader;
//import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
//import ai.sklearn4j.preprocessing.data.QuantileTransformer;
//
//
///**
// * QuantileTransformer object loader.
// */
//
//public class QuantileTransformerContentLoader extends BaseScikitLearnContentLoader<QuantileTransformer> {
//    /**
//     * Instantiate a new object of QuantileTransformerContentLoader.
//     */
//    public QuantileTransformerContentLoader() {
//        super("pp_quantile_transformer");
//    }
//
//    /**
//     * Instantiate an unloaded QuantileTransformer scikit-learn object.
//     *
//     * @return The unloaded scikit-learn object.
//     */
//    @Override
//    protected QuantileTransformer createResultObject() {
//        return new QuantileTransformer();
//    }
//    /**
//     * Create a clean instance of the loader.
//     *
//     * @return A clean instance of the loader.
//     */
//    @Override
//    public IScikitLearnContentLoader duplicate() {
//        return new QuantileTransformerContentLoader();
//    }
//    /**
//     * Defines the fields that are required to initialize a trained scikit-learn object.
//     */
//    @Override
//    protected void registerSetters() {
//        // Fields from the documentation
//        registerLongField("n_quantiles_", this::setNQuantiles);
//        registerNumpyArrayField("quantiles_", this::setQuantiles);
//        registerNumpyArrayField("references_", this::setReferences);
//        registerLongField("n_features", this::setNFeaturesIn);
//        registerStringArrayField("feature_names", this::setFeatureNamesIn);
//
//        // Fields from the dir() method
//        registerLongField("ignore_implicit_zeros", this::setIgnoreImplicitZeros);
//        registerStringField("output_distribution", this::setOutputDistribution);
//        registerLongField("subsample", this::setSubsample);
//    }
//
//    /**
//     * Sets the The values corresponding the quantiles of reference.
//     *
//     * @param result The scikit-learn object to be loaded.
//     * @param value  The loaded value from stream.
//     */
//    private void setQuantiles(QuantileTransformer result, NumpyArray value) {
//        result.setQuantiles(value);
//    }
//
//    /**
//     * Sets the Quantiles of references.
//     *
//     * @param result The scikit-learn object to be loaded.
//     * @param value  The loaded value from stream.
//     */
//    private void setReferences(QuantileTransformer result, NumpyArray value) {
//        result.setReferences(value);
//    }
//
//    /**
//     * Sets the Number of features seen during `fit`.
//     *
//     * @param result The scikit-learn object to be loaded.
//     * @param value  The loaded value from stream.
//     */
//    private void setNFeaturesIn(QuantileTransformer result, long value) {
//        result.setNFeaturesIn(value);
//    }
//
//    /**
//     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
//     * names that are all strings.
//     *
//     * @param result The scikit-learn object to be loaded.
//     * @param value  The loaded value from stream.
//     */
//    private void setFeatureNamesIn(QuantileTransformer result, String[] value) {
//        result.setFeatureNamesIn(value);
//    }
//
//    /**
//     * Sets the ignore_implicit_zeros field.
//     *
//     * @param result The scikit-learn object to be loaded.
//     * @param value  The loaded value from stream.
//     */
//    private void setIgnoreImplicitZeros(QuantileTransformer result, long value) {
//        result.setIgnoreImplicitZeros(value == 1);
//    }
//
//    /**
//     * Sets the n_quantiles field.
//     *
//     * @param result The scikit-learn object to be loaded.
//     * @param value  The loaded value from stream.
//     */
//    private void setNQuantiles(QuantileTransformer result, long value) {
//        result.setNQuantiles(value);
//    }
//
//    /**
//     * Sets the output_distribution field.
//     *
//     * @param result The scikit-learn object to be loaded.
//     * @param value  The loaded value from stream.
//     */
//    private void setOutputDistribution(QuantileTransformer result, String value) {
//        result.setOutputDistribution(value);
//    }
//
//    /**
//     * Sets the subsample field.
//     *
//     * @param result The scikit-learn object to be loaded.
//     * @param value  The loaded value from stream.
//     */
//    private void setSubsample(QuantileTransformer result, long value) {
//        result.setSubsample(value);
//    }
//
//}
