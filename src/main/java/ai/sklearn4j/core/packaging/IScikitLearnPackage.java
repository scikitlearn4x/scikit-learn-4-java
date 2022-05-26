package ai.sklearn4j.core.packaging;

import java.util.Map;

/**
 * This interface provides an abstraction over the physical file format to decouple the deserialization
 * logic from the file format. This will ease the modifications to the file format and versioning for
 * backward compatibility.
 */
public interface IScikitLearnPackage {
    /**
     * Gets the object that stores the information provided in the binary package file header. The header
     * include at the minimum the version of the file and the version of scikit-learn used to train the
     * content of the file.
     *
     * @return An instance of IScikitLearnPackageHeader containing the parsed information of the file header.
     */
    IScikitLearnPackageHeader getPackageHeader();

    /**
     * Gets a Map[String -> Object] of the extra values stored by the developer when saving the binary package.
     * @return A dictionary that contains the extra values stored along with the binary package file.
     */
    Map<String, Object> getExtraValues();

    /**
     * Get the primary content stored in binary package file.
     *
     * @param index Index of the content to retrieve.
     *
     * @return A scikit-learn object that can now be used in Java.
     */
    Object getModel(int index);

    /**
     * Loads the binary package from a file.
     * @param path Path of file to be loaded.
     */
    void loadFromFile(String path);
}
