package test.sklearn4j.core.test_template;

import ai.sklearn4j.base.ClassifierMixin;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import org.junit.jupiter.api.Assertions;
import test.sklearn4j.TestHelper;
import test.sklearn4j.core.test_template.bases.BaseTesterV1;

import java.util.HashMap;
import java.util.Map;

public class ClassifierTestingTemplateV1 extends BaseTesterV1 {
    public String dataSetName = null;
    public String[] featureNames = null;
    public boolean supportProbability = false;

    private static final Map<String, NumpyArray<Double>> datasetCache = new HashMap<>();

    @Override
    protected void performUseCaseSpecificTest(IScikitLearnPackage binaryPackage) {
        validateClassifierData(binaryPackage);
    }

    private void validateClassifierData(IScikitLearnPackage binaryPackage) {
        ClassifierMixin classifier = (ClassifierMixin) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = loadDataSet((String) binaryPackage.getExtraValues().get("dataset_name"));
        NumpyArray<Double> gtProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_probabilities");
        NumpyArray<Long> gtPredictions = (NumpyArray<Long>) binaryPackage.getExtraValues().get("predictions");
        NumpyArray<Double> gtLogProbabilities = (NumpyArray<Double>) binaryPackage.getExtraValues().get("prediction_log_probabilities");

        NumpyArray<Long> predictions = classifier.predict(x);
        TestHelper.assertEqualPredictions(predictions, (long[]) gtPredictions.getWrapper().getRawArray());

        NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
        TestHelper.assertEqualData(probabilities, (double[][]) gtProbabilities.getWrapper().getRawArray());

        NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);
        TestHelper.assertEqualData(logProbabilities, (double[][]) gtLogProbabilities.getWrapper().getRawArray());
    }

    private NumpyArray<Double> loadDataSet(String datasetName) {
        synchronized (datasetCache) {
            if (datasetCache.containsKey(datasetName)) {
                return datasetCache.get(datasetName);
            }

            String[] lines = readAllText(System.getProperty("user.dir") + "/test_data/" + datasetName + "_x.txt").split("\n");
            int featureCount = lines[0].split(" ").length;

            double[][] data = new double[lines.length][featureCount];

            for (int i = 0; i < data.length; i++) {
                String[] items = lines[i].split(" ");

                for (int j = 0; j < items.length; j++) {
                    data[i][j] = Double.parseDouble(items[j]);
                }
            }

            NumpyArray<Double> result = NumpyArrayFactory.from(data);
            datasetCache.put(datasetName, result);
            return result;
        }
    }

    @Override
    protected String getBinaryFilePath(String version) {
        return String.format("%s/%s/%s/%s/%s_%s_on_%s.skx", TestHelper.TEST_FILES_HOME, version, scikitLearnVersion, pythonVersion, objectName.toLowerCase().replace(" ", "_"), configurationName.replace(" ", "_"), dataSetName);
    }

    @Override
    protected void validateExtraValues(IScikitLearnPackage binaryPackage) {
        Assertions.assertEquals(dataSetName, binaryPackage.getExtraValues().get("dataset_name"));
        String[] actualFeatures = (String[]) binaryPackage.getExtraValues().get("feature_names");

        if (actualFeatures != null && featureNames == null) {
            Assertions.fail("The binary package provides feature names but the test doesn't specify them.");
        } else if (actualFeatures == null && featureNames != null) {
            Assertions.fail("The binary package is missing feature names while the test specifies them.");
        } else if (actualFeatures != null && featureNames != null) {
            TestHelper.assertCorrectFeatureNames(featureNames, actualFeatures);
        }
    }
}