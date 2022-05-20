package ai.sklearn4j.core.packaging.version_1;

import ai.sklearn4j.core.packaging.BinaryModelPackage;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.core.packaging.IScikitLearnPackageHeader;
import ai.sklearn4j.core.packaging.loaders.IScikitLearnContentLoader;
import ai.sklearn4j.core.packaging.loaders.ScikitLearnContentLoader;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ScikitLearnPackageV1 implements IScikitLearnPackage {
    private ScikitLearnPackageHeaderV1 header = null;
    private List<Object> primaryContent = null;
    private Map<String, Object> extras = null;

    public IScikitLearnPackageHeader getPackageHeader() {
        return header;
    }

    public Map<String, Object> getExtraValues() {
        return extras;
    }

    public Object getModel(int index) {
        return primaryContent.get(index);
    }

    @Override
    public void loadFromFile(String path) {
        BinaryModelPackage buffer = BinaryModelPackage.fromFile(path);

        loadFileHeader(buffer);
        loadFilePrimaryContent(buffer);
        loadFileExtraContent(buffer);

    }

    private void loadFileExtraContent(BinaryModelPackage buffer) {
        if (buffer.canRead()) {
            extras = buffer.readDictionary();
        } else {
            extras = new HashMap<>();
        }
    }

    private void loadFilePrimaryContent(BinaryModelPackage buffer) {
        primaryContent = new ArrayList<>();
        for (String serializerType : header.serializerTypes) {
            IScikitLearnContentLoader loader = ScikitLearnContentLoader.loaderForType(serializerType);
            primaryContent.add(loader.loadContent(buffer));
        }
    }

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
        header.serializerTypes = (String[])headerValues.get("serializer_types");
    }
}
