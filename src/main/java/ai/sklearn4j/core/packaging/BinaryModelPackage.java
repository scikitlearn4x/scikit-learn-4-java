package ai.sklearn4j.core.packaging;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

import java.io.*;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The python package (sklearn4x) implements a class named BinaryBuffer that saves the python objects
 * in a binary format. BinaryModelPackage is its counterpart to load these files (or stream) in other
 * languages.
 */
public class BinaryModelPackage {
    /**
     * Constant value to denote the element is of type byte.
     */
    private static final int ELEMENT_TYPE_BYTE = 0x01;

    /**
     * Constant value to denote the element is of type short.
     */
    private static final int ELEMENT_TYPE_SHORT = 0x02;

    /**
     * Constant value to denote the element is of type int.
     */
    private static final int ELEMENT_TYPE_INT = 0x04;

    /**
     * Constant value to denote the element is of type long.
     */
    private static final int ELEMENT_TYPE_LONG = 0x08;

    /**
     * Constant value to denote the element is of type unsigned byte.
     */
    private static final int ELEMENT_TYPE_UNSIGNED_BYTE = 0x11;

    /**
     * Constant value to denote the element is of type unsigned short.
     */
    private static final int ELEMENT_TYPE_UNSIGNED_SHORT = 0x12;

    /**
     * Constant value to denote the element is of type unsigned int.
     */
    private static final int ELEMENT_TYPE_UNSIGNED_INT = 0x14;

    /**
     * Constant value to denote the element is of type unsigned long.
     */
    private static final int ELEMENT_TYPE_UNSIGNED_LONG = 0x18;

    /**
     * Constant value to denote the element is of type float.
     */
    private static final int ELEMENT_TYPE_FLOAT = 0x20;

    /**
     * Constant value to denote the element is of type double.
     */
    private static final int ELEMENT_TYPE_DOUBLE = 0x21;

    /**
     * Constant value to denote the element is of type string.
     */
    private static final int ELEMENT_TYPE_STRING = 0x30;

    /**
     * Constant value to denote the element is of type list.
     */
    private static final int ELEMENT_TYPE_LIST = 0x40;

    /**
     * Constant value to denote the element is of type dictionary.
     */
    private static final int ELEMENT_TYPE_DICTIONARY = 0x41;

    /**
     * Constant value to denote the element is of type numpy array.
     */
    private static final int ELEMENT_TYPE_NUMPY_ARRAY = 0x42;

    /**
     * Constant value to denote the element is of type string array.
     */
    private static final int ELEMENT_TYPE_STRING_ARRAY = 0x43;

    /**
     * Constant value to denote the element is null.
     */
    private static final int ELEMENT_TYPE_NULL = 0x10;

    /**
     * The input stream from which the buffer is loading.
     */
    private final InputStream stream;

    /**
     * Instantiate a new BinaryModelPackage object.
     *
     * @param stream The input stream from which the buffer is loading.
     */
    private BinaryModelPackage(InputStream stream) {
        this.stream = stream;
    }

    /**
     * Creates a new BinaryModelPackage that reads from  a given file.
     *
     * @param path Path to the file to be read.
     * @return A BinaryModelPackage instance to read the package file.
     */
    public static BinaryModelPackage fromFile(String path) {
        try {
            InputStream stream = new BufferedInputStream(new FileInputStream(path));
            byte[] data = new byte[stream.available()];
            stream.read(data);
            BinaryModelPackage result = fromStream(new ByteArrayInputStream(data));

            stream.close();

            return result;
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Creates a new BinaryModelPackage that reads from a given stream. This method is useful to
     * load the files from a custom-made package or an encrypted one.
     *
     * @param stream The input stream that should be loaded. It is recommended to use a buffered stream.
     * @return A BinaryModelPackage instance to read the package stream.
     */
    public static BinaryModelPackage fromStream(InputStream stream) {
        return new BinaryModelPackage(stream);
    }

    /**
     * Reads a byte from the stream.
     *
     * @return A byte value stored in the stream.
     */
    public byte readByte() {
        int size = 1;

        byte[] data = readBuffer(size);
        return data[0];
    }

    /**
     * Reads a 16-bits integer from the stream. The values are stored in little endian formats.
     *
     * @return A short value stored in the stream.
     */
    public short readShort() {
        int size = 2;
        int result = 0;

        byte[] data = readBuffer(size);

        for (int i = 0; i < size; i++) {
            result = result * 256;
            result = (data[size - 1 - i] & 0x000000FF) + result;
        }

        return (short) result;
    }

    /**
     * Reads a 32-bits integer from the stream. The values are stored in little endian formats.
     *
     * @return An integer value stored in the stream.
     */
    public int readInteger() {
        int size = 4;
        int result = 0;

        byte[] data = readBuffer(size);

        for (int i = 0; i < size; i++) {
            result = result * 256;
            result = (data[size - 1 - i] & 0x000000FF) + result;
        }

        return result;
    }

    /**
     * Reads a 64-bits integer from the stream. The values are stored in little endian formats.
     *
     * @return An integer value stored in the stream.
     */
    public long readLongInteger() {
        int size = 8;
        long result = 0;

        byte[] data = readBuffer(size);

        for (int i = 0; i < size; i++) {
            result = result * 256;
            result = (data[size - 1 - i] & 0x000000FF) + result;
        }

        return result;
    }


    /**
     * Reads a 32-bits floating point value from the stream. The values are stored in IEE 754 and
     * little endian formats.
     *
     * @return A float value stored in the stream.
     */
    public float readFloat() {
        float result = Float.NaN;
        int hasValue = readByte();

        if (hasValue == 1) {
            int temp = readInteger();
            result = Float.intBitsToFloat(temp);
        }

        return result;
    }

    /**
     * Reads a 64-bits floating point value from the stream. The values are stored in IEE 754 and
     * little endian formats.
     *
     * @return A double value stored in the stream.
     */
    public double readDouble() {
        double result = Double.NaN;
        int hasValue = readByte();

        if (hasValue == 1) {
            long temp = readLongInteger();
            result = Double.longBitsToDouble(temp);
        }

        return result;
    }

    /**
     * Reads a string with UTF-8 encoding from stream.
     *
     * @return The string stored in the stream, or null if it has no value.
     */
    public String readString() {
        String result = null;
        int hasValue = readByte();

        if (hasValue == 1) {
            int length = readInteger();
            byte[] data = readBuffer(length);

            result = new String(data, StandardCharsets.UTF_8);
        }

        return result;
    }

    /**
     * Read a multidimensional numpy array from the stream.
     *
     * @return The numpy array stored in the stream, or null if it has no value.
     */
    public NumpyArray readNumpyArray() {
        NumpyArray result = null;
        int hasValue = readByte();

        if (hasValue == 1) {
            int[] shape = new int[readInteger()];
            int elementType = readByte();

            for (int i = 0; i < shape.length; i++) {
                shape[i] = readInteger();
            }

            result = createNumpyArray(elementType, shape);
            readNumpyDataFromStream(result.getWrapper().getRawArray(), shape, 0, elementType);
        }

        return result;
    }

    /**
     * The element type of a numpy array is stored as a byte. The value of this byte is defined as
     * constants ELEMENT_TYPE_* in the class. This method converts these element type constants into
     * respective Java types to create an array using reflection.
     *
     * @param elementType A byte value read from buffer that specifies the type of numpy array.
     * @return A Class object used by reflection to create a new array.
     */
    private NumpyArray createNumpyArray(int elementType, int[] shape) {
        NumpyArray result = null;

        if (elementType == ELEMENT_TYPE_BYTE || elementType == ELEMENT_TYPE_UNSIGNED_BYTE) {
            result = NumpyArrayFactory.arrayOfInt8WithShape(shape);
        } else if (elementType == ELEMENT_TYPE_SHORT || elementType == ELEMENT_TYPE_UNSIGNED_SHORT) {
            result = NumpyArrayFactory.arrayOfInt16WithShape(shape);
        } else if (elementType == ELEMENT_TYPE_INT || elementType == ELEMENT_TYPE_UNSIGNED_INT) {
            result = NumpyArrayFactory.arrayOfInt32WithShape(shape);
        } else if (elementType == ELEMENT_TYPE_LONG || elementType == ELEMENT_TYPE_UNSIGNED_LONG) {
            result = NumpyArrayFactory.arrayOfInt64WithShape(shape);
        } else if (elementType == ELEMENT_TYPE_FLOAT) {
            result = NumpyArrayFactory.arrayOfFloatWithShape(shape);
        } else if (elementType == ELEMENT_TYPE_DOUBLE) {
            result = NumpyArrayFactory.arrayOfDoubleWithShape(shape);
        } else {
            throw new RuntimeException(String.format("Numpy array with element type %d is not supported.", elementType));
        }

        return result;
    }

    /**
     * Reads a list from the stream.
     *
     * @return The list stored in the stream, or null if it has no value.
     */
    public List<Object> readList() {
        List<Object> result = null;
        int hasValue = readByte();

        if (hasValue == 1) {
            result = new ArrayList<>();

            int count = readInteger();
            for (int i = 0; i < count; i++) {
                byte elementType = readByte();

                if (elementType == ELEMENT_TYPE_NULL) {
                    result.add(null);
                } else {
                    IBinaryModelPackagePrimitiveValueReader reader = getPrimitiveDataReader(elementType);
                    Object value = reader.readPrimitiveValue();
                    result.add(value);
                }
            }
        }

        return result;
    }

    /**
     * Reads a dictionary from the stream.
     *
     * @return The dictionary stored in the stream, or null if it has no value.
     */
    public Map<String, Object> readDictionary() {
        Map<String, Object> result = null;
        int hasValue = readByte();

        if (hasValue == 1) {
            result = new HashMap<>();
            int count = readInteger();

            for (int i = 0; i < count; i++) {
                String key = readString();
                byte elementType = readByte();
                if (elementType == ELEMENT_TYPE_NULL) {
                    result.put(key, null);
                } else if (elementType == ELEMENT_TYPE_STRING_ARRAY) {
                    result.put(key, readStringArray());
                } else {
                    IBinaryModelPackagePrimitiveValueReader reader = getPrimitiveDataReader(elementType);
                    Object value = reader.readPrimitiveValue();
                    result.put(key, value);
                }
            }
        }

        return result;
    }

    /**
     * Reads an array of string from the stream.
     *
     * @return The String[] stored in the stream, or null if it has no value.
     */
    public String[] readStringArray() {
        String[] result = null;
        int hasValue = readByte();

        if (hasValue == 1) {
            int count = readInteger();
            result = new String[count];

            for (int i = 0; i < count; i++) {
                result[i] = readString();
            }
        }

        return result;
    }


    /**
     * Recursively read the stream to load an encoded numpy array from the stream.
     *
     * @param array       The array to be loaded.
     * @param shape       The shape of the numpy array.
     * @param dimension   The current dimension being loaded.
     * @param elementType The type of the numpy array's elements.
     */
    private void readNumpyDataFromStream(Object array, int[] shape, int dimension, int elementType) {
        if (dimension == shape.length - 1) {
            // This is the last dimension of the tensor, read actual values.
            IBinaryModelPackagePrimitiveValueReader reader = getPrimitiveDataReader(elementType);
            int count = shape[dimension];

            for (int i = 0; i < count; i++) {
                Array.set(array, i, reader.readPrimitiveValue());
            }
        } else {
            // This is an intermediate dimensions, it should read a tensor from the buffer
            for (int i = 0; i < shape[dimension]; i++) {
                readNumpyDataFromStream(Array.get(array, i), shape, dimension + 1, elementType);
            }
        }
    }

    /**
     * Gets an IBinaryModelPackagePrimitiveValueReader to read numerical values from the stream. This
     * method was added to prevent having many if-elses in the code.
     *
     * @param elementType A byte value read from buffer that specifies the type of numerical value.
     * @return An instance of IBinaryModelPackagePrimitiveValueReader to read numerical values.
     */
    private IBinaryModelPackagePrimitiveValueReader getPrimitiveDataReader(int elementType) {
        IBinaryModelPackagePrimitiveValueReader result = null;

        if (elementType == ELEMENT_TYPE_BYTE || elementType == ELEMENT_TYPE_UNSIGNED_BYTE) {
            result = this::readByte;
        } else if (elementType == ELEMENT_TYPE_SHORT || elementType == ELEMENT_TYPE_UNSIGNED_SHORT) {
            result = this::readShort;
        } else if (elementType == ELEMENT_TYPE_INT || elementType == ELEMENT_TYPE_UNSIGNED_INT) {
            result = this::readInteger;
        } else if (elementType == ELEMENT_TYPE_LONG || elementType == ELEMENT_TYPE_UNSIGNED_LONG) {
            result = this::readLongInteger;
        } else if (elementType == ELEMENT_TYPE_FLOAT) {
            result = this::readFloat;
        } else if (elementType == ELEMENT_TYPE_DOUBLE) {
            result = this::readDouble;
        } else if (elementType == ELEMENT_TYPE_STRING) {
            result = this::readString;
        } else if (elementType == ELEMENT_TYPE_DICTIONARY) {
            result = this::readDictionary;
        } else if (elementType == ELEMENT_TYPE_NUMPY_ARRAY) {
            result = this::readNumpyArray;
        } else if (elementType == ELEMENT_TYPE_LIST) {
            result = this::readList;
        } else {
            throw new RuntimeException(String.format("Numpy array with element type %d is not supported.", elementType));
        }

        return result;
    }


    /**
     * Reads a buffer from the stream.
     *
     * @param size Length of the buffer to be read from stream.
     * @return A byte[] buffer.
     */
    private byte[] readBuffer(int size) {
        byte[] buffer = new byte[size];

        int length = 0;
        try {
            length = stream.read(buffer);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        if (length != size) {
            throw new RuntimeException(String.format("Unable to read %d bytes from the stream.", size));
        }

        return buffer;
    }

    /**
     * Checks if there is still available content in the input stream.
     *
     * @return A boolean indicating if the stream still has content or not.
     */
    public boolean canRead() {
        try {
            return stream.available() > 0;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

/**
 * An abstraction on the methods that can read a primitive value from the stream.
 */
interface IBinaryModelPackagePrimitiveValueReader {
    /**
     * Reads a primitive value from the stream.
     *
     * @return The primitive value read from stream.
     */
    Object readPrimitiveValue();
}
