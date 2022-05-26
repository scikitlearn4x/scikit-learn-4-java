package ai.sklearn4j.core.packaging.version_1;

import ai.sklearn4j.core.packaging.IScikitLearnPackageHeader;

/**
 * A data class that contains the values stored in the header of the binary package files. This is the
 * default implementation for version 1 of the binary file formats.
 */
public class ScikitLearnPackageHeaderV1 implements IScikitLearnPackageHeader {
    /**
     * The version of the binary package file. This value is the first 4 bytes stored in the file.
     */
    public int fileFormatVersion = 0;

    /**
     * The sklearn4x version. This is the library used to serialize the models into file. For more
     * information, please see:
     *
     * https://pypi.org/project/sklearn4x/
     */
    public String sklearn4xVersion = null;

    /**
     * The version of the scikit-learn library used to train/prepare the objects contained in the
     * current file.
     */
    public String scikitLearnVersion = null;

    /**
     * The version of the numpy library used to train/prepare the objects contained in the
     * current file.
     */
    public String numpyVersion = null;

    /**
     * The version of the scipy library used to train/prepare the objects contained in the
     * current file.
     */
    public String scipyVersion = null;

    /**
     * Information about the python version installed that was used to create the binary package file.
     */
    public String pythonInfo = null;

    /**
     * Information about the platform that was used to create the binary package file.
     */
    public String platformInfo = null;

    /**
     * List of the serializers used to serialize the primary contents of the binary package. These values
     * are to be used for internal purposes only and should not be modified by the developers.
     */
    public String[] serializerTypes = null;

    /**
     * Gets the version of the binary package file. This value is the first 4 bytes stored in the file.
     *
     * @return Binary package file version.
     */
    @Override
    public int getFileFormatVersion() {
        return fileFormatVersion;
    }

    /**
     * Gets the version of the scikit-learn library used to train/prepare the objects contained in the
     * current file.
     *
     * @return scikit-learn library used to train/prepare the objects
     */
    @Override
    public String getScikitLearnVersion() {
        return scikitLearnVersion;
    }
}
