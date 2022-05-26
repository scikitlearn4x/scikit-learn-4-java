package ai.sklearn4j.core.packaging;

/**
 * Abstracts the version of the binary package from its physical layout and its version.
 */
public interface IScikitLearnPackageHeader {
    /**
     * Gets the version of the binary package file.
     *
     * @return An int value specifying the version.
     */
    int getFileFormatVersion();

    /**
     * Gets the version that was used to train the scikit-learn object.
     *
     * @return A string version of scikit-learn library.
     */
    String getScikitLearnVersion();
}
