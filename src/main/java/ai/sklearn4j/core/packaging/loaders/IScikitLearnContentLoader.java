package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.packaging.BinaryModelPackage;

public interface IScikitLearnContentLoader {
    String getTypeName();
    Object loadContent(BinaryModelPackage buffer);

    IScikitLearnContentLoader duplicate();
}
