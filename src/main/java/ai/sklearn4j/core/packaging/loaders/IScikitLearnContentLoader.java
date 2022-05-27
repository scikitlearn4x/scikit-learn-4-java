package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.packaging.BinaryModelPackage;

/**
 * IScikitLearnContentLoader abstracts the format of individual objects from the file formatting. A class
 * should be derived from this interface for each classifier in scikit-learn.
 */
public interface IScikitLearnContentLoader {
    /**
     * Name of the loader. The name is stored in the header of the binary package file to be used during
     * deserialization.
     *
     * @return The name/id of the loader type.
     */
    String getTypeName();

    /**
     * Loads a scikit-learn object from an input stream.
     *
     * @param buffer The buffer to load the object from.
     * @return A deserialized ready to use object.
     */
    Object loadContent(BinaryModelPackage buffer);

    /**
     * Creates a clone of the instance.
     *
     * @return An empty clean instance of the loader.
     */
    IScikitLearnContentLoader duplicate();
}
