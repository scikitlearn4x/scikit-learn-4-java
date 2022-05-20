package ai.sklearn4j.core.packaging.version_1;

import ai.sklearn4j.core.packaging.IScikitLearnPackageHeader;

import java.util.List;

public class ScikitLearnPackageHeaderV1 implements IScikitLearnPackageHeader {
    public int fileFormatVersion = 0;
    public String sklearn4xVersion = null;
    public String scikitLearnVersion = null;
    public String numpyVersion = null;
    public String scipyVersion = null;
    public String pythonInfo = null;
    public String platformInfo = null;
    public String[] serializerTypes = null;

    @Override
    public int getFileFormatVersion() {
        return fileFormatVersion;
    }

    @Override
    public String getScikitLearnVersion() {
        return scikitLearnVersion;
    }
}
