package test.sklearn4j.classifiers.gaussian_naive_bayes;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.core.packaging.ScikitLearnPackageFactory;
import ai.sklearn4j.naive_bayes.GaussianNaiveBayes;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import test.sklearn4j.TestHelper;

public class GaussianNaiveBayesSimplestBaseCaseWithoutCustomizationTests {
	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.0 on python 3.5.6
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_5_6WithSkLearn0_20_0OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.5/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_5_6WithSkLearn0_20_0OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.5/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_5_6WithSkLearn0_20_0OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.5/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.0 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_0OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_0OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_0OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.0 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_0OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_0OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_0OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.0/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.0", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.1 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.1 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.2 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.2 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.3 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_3OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.3/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_3OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.3/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_20_3OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.3/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.20.3 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_3OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.3/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_3OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.3/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_20_3OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.20.3/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.20.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.21.1 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.21.1 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.21.2 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.21.2 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.21.3 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_3OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.3/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_3OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.3/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_21_3OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.3/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.21.3 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_3OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.3/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_3OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.3/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_21_3OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.21.3/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.21.3", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.22 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_22OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_22OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_22OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.22 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_22OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_22OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_22OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.22.1 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_22_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_22_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_22_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.22.1 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_22_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_22_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_22_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.22.1 on python 3.8.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_22_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_22_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_22_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.22.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.22.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.23.1 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_23_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_23_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_23_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.23.1 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_23_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_23_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_23_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.23.1 on python 3.8.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_23_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_23_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_23_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.23.2 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_23_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_23_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_23_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.23.2 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_23_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_23_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_23_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.23.2 on python 3.8.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_23_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_23_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_23_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.23.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.23.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.1.1 on python 3.9.12
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_1_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_1_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_1_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.1.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.1.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.1 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_24_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_24_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_24_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.1 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_24_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_24_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_24_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.1 on python 3.8.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_24_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_24_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_24_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.1 on python 3.9.12
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn0_24_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn0_24_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn0_24_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.2 on python 3.6.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_24_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_24_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_6_13WithSkLearn0_24_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.6/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.2 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_24_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_24_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn0_24_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.2 on python 3.8.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_24_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_24_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn0_24_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.2 on python 3.9.12
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn0_24_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn0_24_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn0_24_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("0.24.2/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("0.24.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.1 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn1_0_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn1_0_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn1_0_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.1 on python 3.8.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn1_0_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn1_0_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn1_0_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.1 on python 3.9.12
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_0_1OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_0_1OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_0_1OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.1/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.1", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.2 on python 3.7.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn1_0_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn1_0_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_7_13WithSkLearn1_0_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.7/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.2 on python 3.8.13
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn1_0_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn1_0_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_8_13WithSkLearn1_0_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.8/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.2 on python 3.9.12
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_0_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_0_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_9_12WithSkLearn1_0_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.2 on python 3.10.4
	// ------------------------------------------------------------------------

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_10_4WithSkLearn1_0_2OnIris() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.10/gaussian_naive_bayes_simplest_base_case_without_customization_on_iris.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("iris", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_10_4WithSkLearn1_0_2OnWine() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.10/gaussian_naive_bayes_simplest_base_case_without_customization_on_wine.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("wine", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

	@Test
	public void testSimplestBaseCaseWithoutCustomizationOnPython3_10_4WithSkLearn1_0_2OnBreastCancer() {
		String path = TestHelper.getAbsolutePathOfBinaryPackage("1.0.2/3.10/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx");
		IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

		// Check header values
		Assertions.assertEquals(1, binaryPackage.getPackageHeader().getFileFormatVersion());
		Assertions.assertEquals("1.0.2", binaryPackage.getPackageHeader().getScikitLearnVersion());

		// Check extra values
		Assertions.assertEquals("breast_cancer", binaryPackage.getExtraValues().get("dataset_name"));

		// Check actual computed values
		GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel("classifier_to_test");

		NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");
		NumpyArray<Long> gtPredictions = (NumpyArray<Long>)binaryPackage.getExtraValues().get("predictions");
		NumpyArray<Long> predictions = classifier.predict(x);
		TestHelper.assertEqualPredictions(predictions, (long[])gtPredictions.getWrapper().getRawArray());

		NumpyArray<Double> gtProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_probabilities");
		NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
		TestHelper.assertEqualData(probabilities, (double[][])gtProbabilities.getWrapper().getRawArray());

		NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>)binaryPackage.getExtraValues().get("prediction_log_probabilities");
		NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
		TestHelper.assertEqualData(logProbabilities, (double[][])gtLogProbabilities.getWrapper().getRawArray());
	}

}