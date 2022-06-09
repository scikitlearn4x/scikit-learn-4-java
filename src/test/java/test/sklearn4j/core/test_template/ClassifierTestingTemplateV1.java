package test.sklearn4j.core.test_template;

import ai.sklearn4j.base.ClassifierMixin;
import ai.sklearn4j.core.Constants;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.core.packaging.ScikitLearnPackageFactory;
import ai.sklearn4j.naive_bayes.GaussianNaiveBayes;
import org.junit.jupiter.api.Assertions;
import test.sklearn4j.TestHelper;

import java.io.File;
import java.io.FileInputStream;
import java.io.FilenameFilter;
import java.util.*;

public class ClassifierTestingTemplateV1 {
    public static final int BINARY_PACKAGE_FILE_FORMAT = 1;

    public String scikitLearnVersion = null;
    public String pythonVersion = null;
    public String classifierName = null;
    public String classifierConfigurationName = null;
    public String dataSetName = null;
    public String[] featureNames = null;
    public boolean supportProbability = false;

    private static final Map<String, NumpyArray<Double>> datasetCache = new HashMap<>();
    private static final Map<String, List<String>> versionContent = new HashMap<>();

    static {
        List<String> versions = getDirectoriesIn(TestHelper.TEST_FILES_HOME);

        for (String folder : versions) {
            String version = folder.substring(0, folder.length() - 1);
            version = version.substring(version.lastIndexOf("/") + 1);

            List<String> classifiers = Arrays.asList(readAllText(folder + "classifiers.txt").split("\n"));

            versionContent.put(version, classifiers);
        }
    }

    public void test() {
        String classifierId = this.classifierName.replace(" ", "_").toLowerCase();

        for (String version : versionContent.keySet()) {
            List<String> classifiersSupported = versionContent.get(version);
            if (classifiersSupported.contains(classifierId)) {
                String path = String.format("%s/%s/%s/%s/%s_%s_on_%s.skx", TestHelper.TEST_FILES_HOME, version, scikitLearnVersion, pythonVersion, classifierName.toLowerCase().replace(" ", "_"), classifierConfigurationName.replace(" ", "_"), dataSetName);
                IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

                validateHeaderValues(binaryPackage);
                validateExtraValues(binaryPackage);
                validateClassifierData(binaryPackage);
            }
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

    public static String readAllText(String path) {
        try {
            FileInputStream fs = new FileInputStream(path);
            byte[] data = new byte[fs.available()];

            fs.read(data);
            fs.close();

            return new String(data);
        } catch (Exception e) {
//            e.printStackTrace();
        }

        return null;
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