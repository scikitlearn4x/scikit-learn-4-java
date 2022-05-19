package ai.sklearn4j.core.packaging;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.version_1.ScikitLearnPackageV1;

import java.io.FileInputStream;
import java.util.Map;

public class ScikitLearnPackage {
    public static IScikitLearnPackage loadFromFile(String path) {
        int version = readFileVersion(path);
        IScikitLearnPackage pkg = null;

        if (version == 1) {
            pkg = new ScikitLearnPackageV1();
            pkg.loadFromFile(path);
        } else {
            throw new RuntimeException("This version of the file format is not supported.");
        }

        return pkg;
    }

    private static int readFileVersion(String path) {
        try {
            FileInputStream fs = new FileInputStream(path);
            BinaryModelPackage buffer = BinaryModelPackage.fromStream(fs);

            int result = buffer.readInteger();
            fs.close();
            return result;
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

}
