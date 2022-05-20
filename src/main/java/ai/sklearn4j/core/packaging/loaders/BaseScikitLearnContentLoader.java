package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.BinaryModelPackage;

import java.util.HashMap;
import java.util.Map;

public abstract class BaseScikitLearnContentLoader<ObjectType> implements IScikitLearnContentLoader {
    private final String typeName;
    private final Map<String, LoaderFieldInfo> fields = new HashMap<>();

    protected BaseScikitLearnContentLoader(String typeName) {
        this.typeName = typeName;
        registerSetters();
    }

    protected abstract ObjectType createResultObject();

    protected abstract void registerSetters();

    @Override
    public String getTypeName() {
        return this.typeName;
    }

    @Override
    public Object loadContent(BinaryModelPackage buffer) {
        ObjectType result = createResultObject();
        int fieldCount = buffer.readInteger();

        for (int i = 0; i < fieldCount; i++) {
            String name = buffer.readString();
            if (!fields.containsKey(name)) {
                throw new RuntimeException("Package contains an unregistered field name: " + name);
            }

            LoaderFieldInfo info = fields.get(name);

            if (info.fieldType == LoaderFieldInfo.FIELD_TYPE_DOUBLE) {
                double value = buffer.readDouble();
                ((IScikitLearnLoaderDoubleFieldSetter<ObjectType>)info.setter).setDoubleField(result, value);
            } else if (info.fieldType == LoaderFieldInfo.FIELD_TYPE_LONG) {
                long value = buffer.readLongInteger();
                ((IScikitLearnLoaderLongFieldSetter<ObjectType>)info.setter).setLongField(result, value);
            } else if (info.fieldType == LoaderFieldInfo.FIELD_TYPE_NUMPY) {
                NumpyArray value = buffer.readNumpyArray();
                ((IScikitLearnLoaderNumpyArrayFieldSetter<ObjectType>)info.setter).setNumpyArrayField(result, value);
            }
        }

        return result;
    }

    protected void registerDoubleField(String name, IScikitLearnLoaderDoubleFieldSetter<ObjectType> setter) {
        if (fields.containsKey(name)) {
            throw new RuntimeException("Field is already added");
        }

        LoaderFieldInfo field = new LoaderFieldInfo();
        field.name = name;
        field.setter = setter;
        field.fieldType = LoaderFieldInfo.FIELD_TYPE_DOUBLE;

        fields.put(name, field);
    }

    protected void registerLongField(String name, IScikitLearnLoaderLongFieldSetter<ObjectType> setter) {
        if (fields.containsKey(name)) {
            throw new RuntimeException("Field is already added");
        }

        LoaderFieldInfo field = new LoaderFieldInfo();
        field.name = name;
        field.setter = setter;
        field.fieldType = LoaderFieldInfo.FIELD_TYPE_LONG;

        fields.put(name, field);
    }

    protected void registerNumpyArrayField(String name, IScikitLearnLoaderNumpyArrayFieldSetter<ObjectType> setter) {
        if (fields.containsKey(name)) {
            throw new RuntimeException("Field is already added");
        }

        LoaderFieldInfo field = new LoaderFieldInfo();
        field.name = name;
        field.setter = setter;
        field.fieldType = LoaderFieldInfo.FIELD_TYPE_NUMPY;

        fields.put(name, field);
    }
}

class LoaderFieldInfo {
    public static final int FIELD_TYPE_DOUBLE = 1;
    public static final int FIELD_TYPE_LONG = 2;
    public static final int FIELD_TYPE_NUMPY = 3;

    public String name = null;
    public int fieldType = 0;
    public Object setter = null;
}
