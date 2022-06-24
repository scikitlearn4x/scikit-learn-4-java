package test.sklearn4j.core.test_template.bases;

import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.core.packaging.ScikitLearnPackageFactory;
import org.junit.jupiter.api.Assertions;
import test.sklearn4j.TestHelper;

import java.io.File;
import java.io.FileInputStream;
import java.io.FilenameFilter;
import java.util.*;

public abstract class BaseTesterV1 {
    public static final int BINARY_PACKAGE_FILE_FORMAT = 1;

    public String objectName = null;
    public String scikitLearnVersion = null;
    public String pythonVersion = null;
    public String configurationName = null;

    protected static final Map<String, List<String>> versionContent = new HashMap<>();

    static {
        List<String> versions = getDirectoriesIn(TestHelper.TEST_FILES_HOME);

        for (String folder : versions) {
            String version = folder.substring(0, folder.length() - 1);
            version = version.substring(version.lastIndexOf("/") + 1);

            List<String> files = getFilesIn(folder);
            List<String> items = new ArrayList<>();

            for (String file : files) {
                if (file.endsWith(".txt")) {
                    List<String> objects = Arrays.asList(readAllText(file).split("\n"));
                    items.addAll(objects);
                }
            }

            versionContent.put(version, items);
        }
    }

    private static ArrayList<String> getDirectoriesIn(String path) {
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

    private static ArrayList<String> getFilesIn(String path) {
        if (!path.endsWith("/")) {
            path += "/";
        }

        File file = new File(path);
        String[] directories = file.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isFile();
            }
        });

        ArrayList<String> result = new ArrayList<>();

        for (String directory : directories) {
            String folder = directory;

            result.add(path + folder);
        }
        return result;
    }

    protected static String readAllText(String path) {
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

    public void test() {
        String classifierId = this.objectName.replace(" ", "_").toLowerCase();

        for (String version : versionContent.keySet()) {
            List<String> classifiersSupported = versionContent.get(version);
            if (classifiersSupported.contains(classifierId)) {
                String path = getBinaryFilePath(version);
                IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

                validateHeaderValues(binaryPackage);
                validateExtraValues(binaryPackage);
                performUseCaseSpecificTest(binaryPackage);
            }
        }
    }

    protected abstract String getBinaryFilePath(String version);

    protected abstract void performUseCaseSpecificTest(IScikitLearnPackage binaryPackage);

    protected abstract void validateExtraValues(IScikitLearnPackage binaryPackage);

    private void validateHeaderValues(IScikitLearnPackage binaryPackage) {
        Assertions.assertEquals(BINARY_PACKAGE_FILE_FORMAT, binaryPackage.getPackageHeader().getFileFormatVersion());
        Assertions.assertEquals(scikitLearnVersion, binaryPackage.getPackageHeader().getScikitLearnVersion());
    }
}
