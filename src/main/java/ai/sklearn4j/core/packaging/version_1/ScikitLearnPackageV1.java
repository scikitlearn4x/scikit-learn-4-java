package ai.sklearn4j.core.packaging.version_1;

import ai.sklearn4j.core.packaging.BinaryModelPackage;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.core.packaging.IScikitLearnPackageHeader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.ScikitLearnContentLoaderFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * A data class that contains the values stored in the binary package files. This is the default
 * implementation for version 1 of the binary file formats.
 */
public class ScikitLearnPackageV1 implements IScikitLearnPackage {
    /**
     * The object storing information in the file header.
     */
    private ScikitLearnPackageHeaderV1 header = null;

    /**
     * Map of the scikit-learn objects of the binary package file.
     */
    private Map<String, Object> primaryContent = null;

    /**
     * Extra information that the developer added to the package file.
     */
    private Map<String, Object> extras = null;

    /**
     * Gets the object that stores the information provided in the binary package file header. The header
     * include at the minimum the version of the file and the version of scikit-learn used to train the
     * content of the file.
     *
     * @return An instance of IScikitLearnPackageHeader containing the parsed information of the file header.
     */
    public IScikitLearnPackageHeader getPackageHeader() {
        return header;
    }

    /**
     * Gets a Map[String: Object] of the extra values stored by the developer when saving the binary package.
     *
     * @return A dictionary that contains the extra values stored along with the binary package file.
     */
    public Map<String, Object> getExtraValues() {
        return extras;
    }

    /**
     * Get the primary content stored in binary package file.
     *
     * @param name Name of the content to retrieve.
     * @return A scikit-learn object that can now be used in Java.
     */
    public Object getModel(String name) {
        return primaryContent.get(name);
    }

    /**
     * Loads the binary package from a file.
     *
     * @param path Path of file to be loaded.
     */
    @Override
    public void loadFromFile(String path) {
        BinaryModelPackage buffer = BinaryModelPackage.fromFile(path);

        loadFileHeader(buffer);
        loadFilePrimaryContent(buffer);
        loadFileExtraContent(buffer);

    }

    /**
     * Loads the extra information that the developer added to the package file.
     *
     * @param buffer The wrapper over the input file/stream.
     */
    private void loadFileExtraContent(BinaryModelPackage buffer) {
        if (buffer.canRead()) {
            extras = buffer.readDictionary();
        } else {
            extras = new HashMap<>();
        }
    }

    /**
     * Loads the primary content stored in binary package file.
     *
     * @param buffer The wrapper over the input file/stream.
     */
    private void loadFilePrimaryContent(BinaryModelPackage buffer) {
        primaryContent = new HashMap<>();
        for (String serializerType : header.serializerTypes) {
            IScikitLearnContentLoader loader = ScikitLearnContentLoaderFactory.loaderForType(serializerType);
            String name = buffer.readString();
            primaryContent.put(name, loader.loadContent(buffer));
        }
    }

    /**
     * Loads the header into an ScikitLearnPackageHeaderV1 object.
     *
     * @param buffer The wrapper over the input file/stream.
     */
    private void loadFileHeader(BinaryModelPackage buffer) {
        header = new ScikitLearnPackageHeaderV1();
        header.fileFormatVersion = buffer.readInteger();
        Map<String, Object> headerValues = buffer.readDictionary();
        header.sklearn4xVersion = (String) headerValues.get("sklearn4x_version");
        header.scikitLearnVersion = (String) headerValues.get("scikit_learn_version");
        header.numpyVersion = (String) headerValues.get("numpy_version");
        header.scipyVersion = (String) headerValues.get("scipy_version");
        header.pythonInfo = (String) headerValues.get("python_info");
        header.platformInfo = (String) headerValues.get("platform_info");
        header.serializerTypes = (String[]) headerValues.get("serializer_types");
    }
}
