package ai.sklearn4j.core.packaging;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * The python package (sklearn4x) implements a class named BinaryBuffer that saves the python objects
 * in a binary format. BinaryModelPackage is its conterpart to load these files (or stream) in other
 * languages.
 */
public class BinaryModelPackage {
    private static final int ELEMENT_TYPE_BYTE = 0x01;
    private static final int ELEMENT_TYPE_SHORT = 0x02;
    private static final int ELEMENT_TYPE_INT = 0x04;
    private static final int ELEMENT_TYPE_LONG = 0x08;
    private static final int ELEMENT_TYPE_UNSIGNED_BYTE = 0x11;
    private static final int ELEMENT_TYPE_UNSIGNED_SHORT = 0x12;
    private static final int ELEMENT_TYPE_UNSIGNED_INT = 0x14;
    private static final int ELEMENT_TYPE_UNSIGNED_LONG = 0x18;
    private static final int ELEMENT_TYPE_FLOAT = 0x20;
    private static final int ELEMENT_TYPE_DOUBLE = 0x21;
    private static final int ELEMENT_TYPE_STRING = 0x30;
    private static final int ELEMENT_TYPE_LIST = 0x40;
    private static final int ELEMENT_TYPE_DICTIONARY = 0x41;

    private final InputStream stream;

    private BinaryModelPackage(InputStream stream) {
        this.stream = stream;
    }

    /**
     * Creates a new BinaryModelPackage that reads from  a given file.
     *
     * @param path Path to the file to be read.
     * @return A BinaryModelPackage instance to read the package file.
     */
    public BinaryModelPackage fromFile(String path) {
        try {
            InputStream stream = new BufferedInputStream(new FileInputStream(path));
            return fromStream(stream);
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
    public BinaryModelPackage fromStream(InputStream stream) {
        return new BinaryModelPackage(stream);
    }

    /**
     * Reads a byte from the stream.
     *
     * @return A byte value stored in the stream.
     */
    public int readByte() {
        int size = 1;

        byte[] data = readBuffer(size);
        return data[0];
    }

    /**
     * Reads a 16-bits integer from the stream. The values are stored in little endian formats.
     *
     * @return A short value stored in the stream.
     */
    public int readShort() {
        int size = 2;
        int result = 0;

        byte[] data = readBuffer(size);

        for (int i = 0; i < size; i++) {
            result = result * 256;
            result = (data[size - 1 - i] & 0x000000FF) + result;
        }

        return result;
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
        float result = 0;
        int temp = readInteger();
        result = Float.intBitsToFloat(temp);

        return result;
    }

    /**
     * Reads a 64-bits floating point value from the stream. The values are stored in IEE 754 and
     * little endian formats.
     *
     * @return A double value stored in the stream.
     */
    public double readDouble() {
        double result = 0;
        long temp = readLongInteger();
        result = Double.longBitsToDouble(temp);

        return result;
    }

    /**
     * Reads a string with UTF-8 encoding from stream.
     *
     * @return The string stored in the stream, or null if it has not value.
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
     * Reads a buffer from the stream.
     *
     * @param size Length of the buffer to be read from stream.
     *
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
}
