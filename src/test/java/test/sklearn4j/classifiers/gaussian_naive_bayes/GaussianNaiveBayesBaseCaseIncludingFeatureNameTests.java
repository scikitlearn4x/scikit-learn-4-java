package test.sklearn4j.classifiers.gaussian_naive_bayes;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.core.packaging.ScikitLearnPackageFactory;
import ai.sklearn4j.naive_bayes.GaussianNaiveBayes;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import test.sklearn4j.TestHelper;

public class GaussianNaiveBayesBaseCaseIncludingFeatureNameTests {
    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.0 on python 3.8.13
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.0 on python 3.9.12
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.0.1 on python 3.8.13
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0_1OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0_1OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0_1OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.0.1 on python 3.9.12
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0_1OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0_1OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0_1OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.0.1 on python 3.10.4
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_0_1OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_0_1OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_0_1OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.0.2 on python 3.8.13
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0_2OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0_2OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_0_2OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.0.2 on python 3.9.12
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0_2OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0_2OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_0_2OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.0.2 on python 3.10.4
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_0_2OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_0_2OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_0_2OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.1.0 on python 3.8.13
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_1_0OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_1_0OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_1_0OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.1.0 on python 3.9.12
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_1_0OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_1_0OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_1_0OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.1.0 on python 3.10.4
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_1_0OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_1_0OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_1_0OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.0/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.1.1 on python 3.8.13
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_1_1OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_1_1OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_8_13WithSkLearn1_1_1OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.8/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.1.1 on python 3.9.12
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_1_1OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_1_1OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_9_12WithSkLearn1_1_1OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.9/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    // ------------------------------------------------------------------------
    // Test for scikit-learn 1.1.1 on python 3.10.4
    // ------------------------------------------------------------------------

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_1_1OnIris() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_iris.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_1_1OnWine() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_wine.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"Class Label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "diluted wines"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    @Test
    public void testBaseCaseIncludingFeatureNameOnPython3_10_4WithSkLearn1_1_1OnBreastCancer() {
        String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.10/gaussian_naive_bayes_base_case_including_feature_name_on_breast_cancer.skx");
        IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

        // Check header values
        Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

        // Check extra values
        Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));
        TestHelper.assertCorrectFeatureNames(new String[]{"f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_19", "f_20", "f_21", "f_22", "f_23", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30"}, (String[]) binaryPackage.getExtraValues().get("feature_names"));

        // Check actual computed values
        GaussianNaiveBayes classifier = (GaussianNaiveBayes) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");
        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

}