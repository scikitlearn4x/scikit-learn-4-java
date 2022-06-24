package test.sklearn4j.classifiers.categorical_naive_baye;

import org.junit.jupiter.api.Test;
import test.sklearn4j.core.test_template.ClassifierTestingTemplateV1;

public class CategoricalNaiveBayesWithExplicitPriorTests {
	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.1.1 on python 3.9
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_1_1OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.1.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_1_1OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.1.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_1_1OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.1.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.1 on python 3.6
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_6WithSkLearn0_24_1OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.6";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_6WithSkLearn0_24_1OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.6";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_6WithSkLearn0_24_1OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.6";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.1 on python 3.7
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn0_24_1OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn0_24_1OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn0_24_1OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.1 on python 3.8
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn0_24_1OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn0_24_1OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn0_24_1OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.1 on python 3.9
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn0_24_1OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn0_24_1OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn0_24_1OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.2 on python 3.6
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_6WithSkLearn0_24_2OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.6";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_6WithSkLearn0_24_2OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.6";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_6WithSkLearn0_24_2OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.6";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.2 on python 3.7
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn0_24_2OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn0_24_2OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn0_24_2OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.2 on python 3.8
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn0_24_2OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn0_24_2OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn0_24_2OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 0.24.2 on python 3.9
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn0_24_2OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn0_24_2OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn0_24_2OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "0.24.2";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.1 on python 3.7
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn1_0_1OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn1_0_1OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn1_0_1OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.1 on python 3.8
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn1_0_1OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn1_0_1OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn1_0_1OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.1 on python 3.9
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_0_1OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_0_1OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_0_1OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.1";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.2 on python 3.7
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn1_0_2OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn1_0_2OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_7WithSkLearn1_0_2OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.7";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.2 on python 3.8
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn1_0_2OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn1_0_2OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_8WithSkLearn1_0_2OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.8";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.2 on python 3.9
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_0_2OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_0_2OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_9WithSkLearn1_0_2OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.9";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	// ------------------------------------------------------------------------
	// Test for scikit-learn 1.0.2 on python 3.10
	// ------------------------------------------------------------------------

	@Test
	public void testWithExplicitPriorOnPython3_10WithSkLearn1_0_2OnIris() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.10";
		testingTemplate.dataSetName = "iris";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_10WithSkLearn1_0_2OnWine() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.10";
		testingTemplate.dataSetName = "wine";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

	@Test
	public void testWithExplicitPriorOnPython3_10WithSkLearn1_0_2OnBreastCancer() {
		ClassifierTestingTemplateV1 testingTemplate = new ClassifierTestingTemplateV1();
		
		testingTemplate.scikitLearnVersion = "1.0.2";
		testingTemplate.pythonVersion = "3.10";
		testingTemplate.dataSetName = "breast_cancer";
		testingTemplate.objectName = "Categorical Naive Bayes";
		testingTemplate.configurationName = "with explicit prior";
		testingTemplate.supportProbability = true;
		testingTemplate.featureNames = null;
		
		testingTemplate.test();
	}

}