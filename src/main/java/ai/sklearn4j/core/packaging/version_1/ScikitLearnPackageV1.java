package ai.sklearn4j.core.packaging.version_1;

import ai.sklearn4j.core.packaging.BinaryModelPackage;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.core.packaging.IScikitLearnPackageHeader;

import java.util.List;
import java.util.Map;

public class ScikitLearnPackageV1 implements IScikitLearnPackage {
    private ScikitLearnPackageHeaderV1 header = null;
    public IScikitLearnPackageHeader getPackageHeader() {
        return header;
    }

    public Map<String, Object> getExtraValues() {
        return null;
    }

    public Object getModel(int index) {
        return null;
    }

    @Override
    public void loadFromFile(String path) {
        BinaryModelPackage buffer = BinaryModelPackage.fromFile(path);

        header = new ScikitLearnPackageHeaderV1();
        header.fileFormatVersion = buffer.readInteger();
        Map<String, Object> headerValues = buffer.readDictionary();
        header.sklearn4xVersion = (String) headerValues.get("sklearn4x_version");
        header.scikitLearnVersion = (String) headerValues.get("scikit_learn_version");
        header.numpyVersion = (String) headerValues.get("numpy_version");
        header.scipyVersion = (String) headerValues.get("scipy_version");
        header.pythonInfo = (String) headerValues.get("python_info");
        header.platformInfo = (String) headerValues.get("platform_info");
        header.serializerTypes = (List<String>)headerValues.get("serializer_types");
    }
}
