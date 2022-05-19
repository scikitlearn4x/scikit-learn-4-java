package ai.sklearn4j.core.packaging;

import java.util.Map;

public interface IScikitLearnPackage {
    IScikitLearnPackageHeader getPackageHeader();

    Map<String, Object> getExtraValues();

    Object getModel(int index);

    void loadFromFile(String path);
}
