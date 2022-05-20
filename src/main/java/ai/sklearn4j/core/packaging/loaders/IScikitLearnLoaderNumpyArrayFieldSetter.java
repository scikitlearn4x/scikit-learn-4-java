package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;

public interface IScikitLearnLoaderNumpyArrayFieldSetter<ObjectType> {
    void setNumpyArrayField(ObjectType obj, NumpyArray value);
}
