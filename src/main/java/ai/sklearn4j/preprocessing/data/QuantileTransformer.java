//// ==================================================================
//// Inference for QuantileTransformer
////
//// Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer
//// ==================================================================
//package ai.sklearn4j.preprocessing.data;
//
//import ai.sklearn4j.base.TransformerMixin;
//import ai.sklearn4j.core.libraries.Scipy;
//import ai.sklearn4j.core.libraries.numpy.NumpyArray;
//import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
//
///**
// * Transform features using quantiles information.
// * This method transforms the features to follow a uniform or a normal
// * distribution. Therefore, for a given feature, this transformation
// * tends to spread out the most frequent values. It also reduces the
// * impact of (marginal) outliers: this is therefore a robust
// * preprocessing scheme.
// * The transformation is applied on each feature independently. First an
// * estimate of the cumulative distribution function of a feature is used
// * to map the original values to a uniform distribution. The obtained
// * values are then mapped to the desired output distribution using the
// * associated quantile function. Features values of new/unseen data that
// * fall below or above the fitted range will be mapped to the bounds of
// * the output distribution. Note that this transform is non-linear. It
// * may distort linear correlations between variables measured at the same
// * scale but renders variables measured at different scales more directly
// * comparable.
// */
//
//public class QuantileTransformer extends TransformerMixin<NumpyArray<Double>, NumpyArray<Double>> {
//    /**
//     * Instantiate a new object of QuantileTransformer.
//     */
//    public QuantileTransformer() {
//
//    }
//
//    /**
//     * The actual number of quantiles used to discretize the cumulative
//     * distribution function.
//     */
//    private long nQuantiles = 0;
//
//    /**
//     * The values corresponding the quantiles of reference.
//     */
//    private NumpyArray<Double> quantiles = null;
//
//    /**
//     * Quantiles of references.
//     */
//    private NumpyArray references = null;
//
//    /**
//     * Number of features seen during `fit`.
//     */
//    private long nFeaturesIn = 0;
//
//    /**
//     * Names of features seen during `fit`. Defined only when `X` has feature
//     * names that are all strings.
//     */
//    private String[] featureNamesIn = null;
//
//    /**
//     * Internal field of scikit-learn object.
//     */
//    private boolean ignoreImplicitZeros = false;
//
//    /**
//     * Internal field of scikit-learn object.
//     */
//    private String outputDistribution = null;
//
//    /**
//     * Internal field of scikit-learn object.
//     */
//    private long subsample = 0;
//
//    /**
//     * Sets the The actual number of quantiles used to discretize the cumulative
//     * distribution function.
//     *
//     * @param value The new value for nQuantiles.
//     */
//    public void setNQuantiles(long value) {
//        this.nQuantiles = value;
//    }
//
//
//    /**
//     * Gets the The actual number of quantiles used to discretize the cumulative
//     * distribution function.
//     */
//    public long getNQuantiles() {
//        return this.nQuantiles;
//    }
//
//
//    /**
//     * Sets the The values corresponding the quantiles of reference.
//     *
//     * @param value The new value for quantiles.
//     */
//    public void setQuantiles(NumpyArray value) {
//        this.quantiles = value;
//    }
//
//
//    /**
//     * Gets the The values corresponding the quantiles of reference.
//     */
//    public NumpyArray getQuantiles() {
//        return this.quantiles;
//    }
//
//
//    /**
//     * Sets the Quantiles of references.
//     *
//     * @param value The new value for references.
//     */
//    public void setReferences(NumpyArray value) {
//        this.references = value;
//    }
//
//
//    /**
//     * Gets the Quantiles of references.
//     */
//    public NumpyArray getReferences() {
//        return this.references;
//    }
//
//
//    /**
//     * Sets the Number of features seen during `fit`.
//     *
//     * @param value The new value for nFeaturesIn.
//     */
//    public void setNFeaturesIn(long value) {
//        this.nFeaturesIn = value;
//    }
//
//
//    /**
//     * Gets the Number of features seen during `fit`.
//     */
//    public long getNFeaturesIn() {
//        return this.nFeaturesIn;
//    }
//
//
//    /**
//     * Sets the Names of features seen during `fit`. Defined only when `X` has feature
//     * names that are all strings.
//     *
//     * @param value The new value for featureNamesIn.
//     */
//    public void setFeatureNamesIn(String[] value) {
//        this.featureNamesIn = value;
//    }
//
//
//    /**
//     * Gets the Names of features seen during `fit`. Defined only when `X` has feature
//     * names that are all strings.
//     */
//    public String[] getFeatureNamesIn() {
//        return this.featureNamesIn;
//    }
//
//
//    /**
//     * Sets the value of IgnoreImplicitZeros
//     *
//     * @param value The new value for IgnoreImplicitZeros.
//     */
//    public void setIgnoreImplicitZeros(boolean value) {
//        this.ignoreImplicitZeros = value;
//    }
//
//
//    /**
//     * Gets the value of IgnoreImplicitZeros
//     */
//    public boolean getIgnoreImplicitZeros() {
//        return this.ignoreImplicitZeros;
//    }
//
//    /**
//     * Sets the value of OutputDistribution
//     *
//     * @param value The new value for OutputDistribution.
//     */
//    public void setOutputDistribution(String value) {
//        this.outputDistribution = value;
//    }
//
//
//    /**
//     * Gets the value of OutputDistribution
//     */
//    public String getOutputDistribution() {
//        return this.outputDistribution;
//    }
//
//
//    /**
//     * Sets the value of Subsample
//     *
//     * @param value The new value for Subsample.
//     */
//    public void setSubsample(long value) {
//        this.subsample = value;
//    }
//
//
//    /**
//     * Gets the value of Subsample
//     */
//    public long getSubsample() {
//        return this.subsample;
//    }
//
//
//    @Override
//    public NumpyArray<Double> transform(NumpyArray<Double> array) {
//        return innerTransform(array, false);
//    }
//
//    @Override
//    public NumpyArray<Double> inverseTransform(NumpyArray<Double> array) {
//        return innerTransform(array, true);
//    }
//
//    private NumpyArray<Double> innerTransform(NumpyArray<Double> array, boolean inverse) {
//        NumpyArray<Double> result = NumpyArrayFactory.createArrayOfShapeAndTypeInfo(array);
//
//        for (int columnIndex = 0; columnIndex < array.getShape()[1]; columnIndex++) {
//            transformColumn(result, array, columnIndex, inverse);
//        }
//
//        return result;
//    }
//
//    /**
//     * Private function to transform a single feature.
//     *
//     * @param result      The array to store the transformation into.
//     * @param array       The input array to transform.
//     * @param columnIndex The index of the column to apply transform.
//     * @param inverse     Specify if it is a transform or inverseTransform method.
//     */
//    private void transformColumn(NumpyArray<Double> result, NumpyArray<Double> array, int columnIndex, boolean inverse) {
//        String output_distribution = this.outputDistribution;
//        double lower_bound_x = 0;
//        double upper_bound_x = 0;
//        double lower_bound_y = 0;
//        double upper_bound_y = 0;
//        double[] X_col = extractColumn(array, columnIndex);
//
//        if (!inverse) {
//            lower_bound_x = quantiles.get(0);
//            upper_bound_x = quantiles.get(quantiles.getShape()[0] - 1);
//            lower_bound_y = 0;
//            upper_bound_y = 1;
//        } else {
//            lower_bound_x = 0;
//            upper_bound_x = 1;
//            lower_bound_y = quantiles.get(0);
//            upper_bound_y = quantiles.get(quantiles.getShape()[0] - 1);
//
//            // for inverse transform, match a uniform distribution
//            if ("normal".equals(outputDistribution)) {
//                X_col = Scipy.NormalDistribution.cumulativeDistributionFunction(X_col);
//                // else output distribution is already a uniform distribution
//            }
//        }
//
///*
//
//        # find index for lower and higher bounds
//        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
//            if output_distribution == "normal":
//                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
//                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
//            if output_distribution == "uniform":
//                lower_bounds_idx = X_col == lower_bound_x
//                upper_bounds_idx = X_col == upper_bound_x
//
//        isfinite_mask = ~np.isnan(X_col)
//        X_col_finite = X_col[isfinite_mask]
//        if not inverse:
//            # Interpolate in one direction and in the other and take the
//            # mean. This is in case of repeated values in the features
//            # and hence repeated quantiles
//            #
//            # If we don't do this, only one extreme of the duplicated is
//            # used (the upper when we do ascending, and the
//            # lower for descending). We take the mean of these two
//            X_col[isfinite_mask] = 0.5 * (
//                np.interp(X_col_finite, quantiles, self.references_)
//                - np.interp(-X_col_finite, -quantiles[::-1], -self.references_[::-1])
//            )
//        else:
//            X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)
//
//        X_col[upper_bounds_idx] = upper_bound_y
//        X_col[lower_bounds_idx] = lower_bound_y
//        # for forward transform, match the output distribution
//        if not inverse:
//            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
//                if output_distribution == "normal":
//                    X_col = stats.norm.ppf(X_col)
//                    # find the value to clip the data to avoid mapping to
//                    # infinity. Clip such that the inverse transform will be
//                    # consistent
//                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
//                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
//                    X_col = np.clip(X_col, clip_min, clip_max)
//                # else output distribution is uniform and the ppf is the
//                # identity function so we let X_col unchanged
//
//        return X_col
// */
//    }
//
//    private double[] extractColumn(NumpyArray<Double> array, int columnIndex) {
//        double[] X_col = new double[array.getShape()[0]]; // to get from the result
//        for (int i = 0; i < array.getShape()[1]; i++) {
//            X_col[i] = array.get(i, columnIndex);
//        }
//        return X_col;
//    }
//
//}