package ai.sklearn4j.core.packaging;

import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.packaging.version_1.ScikitLearnPackageV1;

import java.io.FileInputStream;

/**
 * ScikitLearnPackageFactory is a factory that takes in the physical file (or stream) of a binary package
 * and parse it into a ready to use object. It also supports handling different versions of the binary
 * files.
 */
public class ScikitLearnPackageFactory {
    /**
     * Load a binary package file into a ready to use object. It reads the version of the file and
     * load the appropriate loader to deserialize the file.
     *
     * @param path Path of the file to be loaded.
     *
     * @return An IScikitLearnPackage object that represent the models stored in the binary package file.
     */
    public static IScikitLearnPackage loadFromFile(String path) {
        int version = readFileVersion(path);
        IScikitLearnPackage pkg = null;

        if (version == 1) {
            pkg = new ScikitLearnPackageV1();
            pkg.loadFromFile(path);
        } else {
            throw new ScikitLearnCoreException("This version of the file format is not supported.");
        }

        return pkg;
    }

    /**
     * Reads the version of the binary package from a file.
     *
     * @param path Path of the file to be loaded.
     *
     * @return An integer representing the file format version.
     */
    private static int readFileVersion(String path) {
        try {
            FileInputStream fs = new FileInputStream(path);
            BinaryModelPackage buffer = BinaryModelPackage.fromStream(fs);

            int result = buffer.readInteger();
            fs.close();
            return result;
        } catch (Exception ex) {
            throw new ScikitLearnCoreException("An error occurred while determining the version of binary package file."  + ex.getMessage());
        }
    }

}
