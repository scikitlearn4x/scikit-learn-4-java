package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.BinaryModelPackage;

import java.util.HashMap;
import java.util.Map;

/**
 * A base class that implements common functionality shred among the scikit-learn object
 * loaders. Each loader will provide a list of supported fields to BaseScikitLearnContentLoader
 * instead of implementing the deserialization manually to simplify the loaders as much as
 * possible.
 *
 * @param <ObjectType> The type of the scikit-learn object that the loader supports.
 */
public abstract class BaseScikitLearnContentLoader<ObjectType> implements IScikitLearnContentLoader {
    /**
     * The type name of the loader.
     */
    private final String typeName;

    /**
     * A map of the fields the loader requires to load an object.
     */
    private final Map<String, LoaderFieldInfo> fields = new HashMap<>();

    /**
     * Instantiate a BaseScikitLearnContentLoader object.
     *
     * @param typeName The type name of the loader.
     */
    protected BaseScikitLearnContentLoader(String typeName) {
        this.typeName = typeName;
        registerSetters();
    }

    /**
     * An abstract method to initialize a new instance of the scikit-learn object supported
     * by the loader.
     *
     * @return The unloaded scikit-learn object supported by the loader.
     */
    protected abstract ObjectType createResultObject();

    /**
     * An abstract method implemented by the derived classes that loads the layout of the
     * binary format. BaseScikitLearnContentLoader uses this layout to load the object.
     */
    protected abstract void registerSetters();

    /**
     * Name of the loader. The name is stored in the header of the binary package file to be used during
     * deserialization.
     *
     * @return The name/id of the loader type.
     */
    @Override
    public String getTypeName() {
        return this.typeName;
    }

    /**
     * Loads the scikit-learn object with the provided layout in registerSetters.
     *
     * @param buffer The buffer to load the object from.
     * @return The fully initialized scikit-learn object.
     */
    @Override
    public Object loadContent(BinaryModelPackage buffer) {
        ObjectType result = createResultObject();
        int fieldCount = buffer.readInteger();

        for (int i = 0; i < fieldCount; i++) {
            String name = buffer.readString();
            if (!fields.containsKey(name)) {
                throw new ScikitLearnCoreException("Package contains an unregistered field name: " + name);
            }

            LoaderFieldInfo info = fields.get(name);

            if (info.fieldType == LoaderFieldInfo.FIELD_TYPE_DOUBLE) {
                double value = buffer.readDouble();
                ((IScikitLearnLoaderDoubleFieldSetter<ObjectType>) info.setter).setDoubleField(result, value);
            } else if (info.fieldType == LoaderFieldInfo.FIELD_TYPE_LONG) {
                long value = buffer.readLongInteger();
                ((IScikitLearnLoaderLongFieldSetter<ObjectType>) info.setter).setLongField(result, value);
            } else if (info.fieldType == LoaderFieldInfo.FIELD_TYPE_NUMPY) {
                NumpyArray value = buffer.readNumpyArray();
                ((IScikitLearnLoaderNumpyArrayFieldSetter<ObjectType>) info.setter).setNumpyArrayField(result, value);
            } else if (info.fieldType == LoaderFieldInfo.FIELD_TYPE_STRING_ARRAY) {
                String[] value = buffer.readStringArray();
                ((IScikitLearnLoaderStringArrayFieldSetter<ObjectType>) info.setter).setStringArrayField(result, value);
            }
        }

        return result;
    }

    /**
     * Registers a double field for the scikit-learn serialized layout.
     *
     * @param name   Name of the field.
     * @param setter The setter callback to load the value of the scikit-learn object.
     */
    protected void registerDoubleField(String name, IScikitLearnLoaderDoubleFieldSetter<ObjectType> setter) {
        if (fields.containsKey(name)) {
            throw new ScikitLearnCoreException("Field is already added");
        }

        LoaderFieldInfo field = new LoaderFieldInfo();
        field.name = name;
        field.setter = setter;
        field.fieldType = LoaderFieldInfo.FIELD_TYPE_DOUBLE;

        fields.put(name, field);
    }

    /**
     * Registers a long integer field for the scikit-learn serialized layout.
     *
     * @param name   Name of the field.
     * @param setter The setter callback to load the value of the scikit-learn object.
     */
    protected void registerLongField(String name, IScikitLearnLoaderLongFieldSetter<ObjectType> setter) {
        if (fields.containsKey(name)) {
            throw new ScikitLearnCoreException("Field is already added");
        }

        LoaderFieldInfo field = new LoaderFieldInfo();
        field.name = name;
        field.setter = setter;
        field.fieldType = LoaderFieldInfo.FIELD_TYPE_LONG;

        fields.put(name, field);
    }

    /**
     * Registers a numpy array field for the scikit-learn serialized layout.
     *
     * @param name   Name of the field.
     * @param setter The setter callback to load the value of the scikit-learn object.
     */
    protected void registerNumpyArrayField(String name, IScikitLearnLoaderNumpyArrayFieldSetter<ObjectType> setter) {
        if (fields.containsKey(name)) {
            throw new ScikitLearnCoreException("Field is already added");
        }

        LoaderFieldInfo field = new LoaderFieldInfo();
        field.name = name;
        field.setter = setter;
        field.fieldType = LoaderFieldInfo.FIELD_TYPE_NUMPY;

        fields.put(name, field);
    }

    /**
     * Registers a String array field for the scikit-learn serialized layout.
     *
     * @param name   Name of the field.
     * @param setter The setter callback to load the value of the scikit-learn object.
     */
    protected void registerStringArrayField(String name, IScikitLearnLoaderStringArrayFieldSetter<ObjectType> setter) {
        if (fields.containsKey(name)) {
            throw new ScikitLearnCoreException("Field is already added");
        }

        LoaderFieldInfo field = new LoaderFieldInfo();
        field.name = name;
        field.setter = setter;
        field.fieldType = LoaderFieldInfo.FIELD_TYPE_STRING_ARRAY;

        fields.put(name, field);
    }
}

/**
 * Data class to store the information on how to load a field.
 */
class LoaderFieldInfo {
    /**
     * Constant to specify the field is of type double.
     */
    public static final int FIELD_TYPE_DOUBLE = 1;

    /**
     * Constant to specify the field is of type long.
     */
    public static final int FIELD_TYPE_LONG = 2;

    /**
     * Constant to specify the field is of type numpy array.
     */
    public static final int FIELD_TYPE_NUMPY = 3;

    /**
     * Constant to specify the field is of type string array.
     */
    public static final int FIELD_TYPE_STRING_ARRAY = 4;

    /**
     * The name of the field.
     */
    public String name = null;

    /**
     * The type of the field.
     */
    public int fieldType = 0;

    /**
     * The setting method that sets the loaded value in the classifier.
     */
    public Object setter = null;
}
