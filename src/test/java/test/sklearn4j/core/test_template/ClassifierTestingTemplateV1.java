package test.sklearn4j.core.test_template;

import ai.sklearn4j.base.ClassifierMixin;
import ai.sklearn4j.core.Constants;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.core.packaging.ScikitLearnPackageFactory;
import ai.sklearn4j.naive_bayes.GaussianNaiveBayes;
import org.junit.jupiter.api.Assertions;
import test.sklearn4j.TestHelper;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.List;

public class ClassifierTestingTemplateV1 {
    public static final int BINARY_PACKAGE_FILE_FORMAT = 1;

    public String scikitLearnVersion = null;
    public String pythonVersion = null;
    public String classifierName = null;
    public String classifierConfigurationName = null;
    public String dataSetName = null;
    public String[] featureNames = null;
    public boolean supportProbability = false;

    public void test() {
        adjustPythonVersion();
        List<String> versions = getDirectoriesIn(TestHelper.TEST_FILES_HOME);

        for (String version : versions) {
            String path = String.format("%s%s/%s/%s_%s_on_%s.skx", version, scikitLearnVersion, pythonVersion, classifierName.toLowerCase().replace(" ", "_"), classifierConfigurationName.replace(" ", "_"), dataSetName);
            if (!(new File(path)).exists()) {
                continue;
            }
            IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

            validateHeaderValues(binaryPackage);
            validateExtraValues(binaryPackage);
            validateClassifierData(binaryPackage);
        }
    }

    private void adjustPythonVersion() {
        int count = 0;

        for (int i = 0; i < pythonVersion.length(); i++) {
            char ch = pythonVersion.charAt(i);
            if (ch == '.') {
                count++;
            }
        }

        if (count == 2) {
            pythonVersion = pythonVersion.substring(0, pythonVersion.lastIndexOf("."));
        }
    }

    public static ArrayList<String> getDirectoriesIn(String path) {
        if (!path.endsWith("/")) {
            path += "/";
        }

        File file = new File(path);
        String[] directories = file.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });

        ArrayList<String> result = new ArrayList<>();

        for (String directory : directories) {
            String folder = directory;
            if (!folder.endsWith("/")) {
                folder += "/";
            }

            result.add(path + folder);
        }
        return result;
    }



    private void validateClassifierData(IScikitLearnPackage binaryPackage) {
        ClassifierMixin classifier = (ClassifierMixin) binaryPackage.getModel("classifier_to_test");

        NumpyArray<Double> x = (NumpyArray<Double>) binaryPackage.getExtraValues().get("training_data");
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

    private void validateExtraValues(IScikitLearnPackage binaryPackage) {
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

    private void validateHeaderValues(IScikitLearnPackage binaryPackage) {
        Assertions.assertEquals(BINARY_PACKAGE_FILE_FORMAT, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals(scikitLearnVersion, binaryPackage.getPackageHeader().getScikitLearnVersion());
    }
}
