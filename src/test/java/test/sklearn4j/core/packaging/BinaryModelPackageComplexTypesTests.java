package test.sklearn4j.core.packaging;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.BinaryModelPackage;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

public class BinaryModelPackageComplexTypesTests {
    @Test
    public void testReadingNullString() {
        byte[] data = new byte[]{0};

        String actual = readStringFromByteArray(data);
        Assertions.assertNull(actual);
    }

    @Test
    public void testReadingASCIIString() {
        byte[] data = new byte[]{1, 4, 0, 0, 0, 116, 101, 115, 116};
        String expected = "test";

        String actual = readStringFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadingUTF8String() {
        byte[] data = new byte[]{1, 10, 0, 0, 0, (byte) 217, (byte) 134, (byte) 217, (byte) 133, (byte) 217, (byte) 136, (byte) 217, (byte) 134, (byte) 217, (byte) 135};
        String expected = "نمونه";

        String actual = readStringFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadSimpleNullNumpyArray() {
        byte[] data = new byte[]{0};

        Object actual = readNumpyArrayFromByteArray(data);
        Assertions.assertNull(actual);
    }

    @Test
    public void testReadSimpleUint8NumpyArray() {
        byte[] data = new byte[]{1, 1, 0, 0, 0, 17, 2, 0, 0, 0, 6, 7};
        byte[] expected = new byte[]{6, 7};

        byte[] actual = (byte[]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testReadSimpleUint16NumpyArray() {
        byte[] data = new byte[]{1, 1, 0, 0, 0, 18, 2, 0, 0, 0, 6, 0, 7, 0};
        short[] expected = new short[]{6, 7};

        short[] actual = (short[]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testReadSimpleUint32NumpyArray() {
        byte[] data = new byte[]{1, 1, 0, 0, 0, 20, 2, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0};
        int[] expected = new int[]{6, 7};

        int[] actual = (int[]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testReadSimpleUint64NumpyArray() {
        byte[] data = new byte[]{1, 1, 0, 0, 0, 24, 2, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0};
        long[] expected = new long[]{6, 7};

        long[] actual = (long[]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testReadSimpleFloatNumpyArray() {
        byte[] data = new byte[]{1, 1, 0, 0, 0, 32, 3, 0, 0, 0, 1, 0, 0, (byte) 192, 64, 1, 0, 0, (byte) 224, 64, 0};
        float[] expected = new float[]{6, 7, Float.NaN};

        float[] actual = (float[]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testReadSimpleDoubleNumpyArray() {
        byte[] data = new byte[]{1, 1, 0, 0, 0, 33, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 24, 64, 1, 0, 0, 0, 0, 0, 0, 28, 64, 0};
        double[] expected = new double[]{6, 7, Double.NaN};

        double[] actual = (double[]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testReadVerticalUint8NumpyArray() {
        byte[] data = new byte[]{1, 2, 0, 0, 0, 17, 2, 0, 0, 0, 1, 0, 0, 0, 6, 7};
        byte[][] expected = new byte[][]{{6}, {7}};

        byte[][] actual = (byte[][]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testReadVerticalUint32NumpyArray() {
        byte[] data = new byte[]{1, 2, 0, 0, 0, 4, 2, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0};
        int[][] expected = new int[][]{{6}, {7}};

        int[][] actual = (int[][]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testRead3DimensionTensorUint32NumpyArray() {
        byte[] data = new byte[]{1, 3, 0, 0, 0, 4, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0};
        int[][][] expected = new int[][][]{{{6}, {7}}, {{8}, {9}}};

        int[][][] actual = (int[][][]) readNumpyArrayFromByteArray(data).getWrapper().getRawArray();
        checkArraySimilarity(expected, actual);
    }

    @Test
    public void testReadingNullArrayOfString() {
        byte[] data = new byte[]{0};
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        String[] actual = binary.readStringArray();
        Assertions.assertNull(actual);
    }

    @Test
    public void testReadingArrayOfString() {
        byte[] data = new byte[]{1, 3, 0, 0, 0, 1, 1, 0, 0, 0, 97, 1, 1, 0, 0, 0, 98, 0};
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        String[] actual = binary.readStringArray();
        Assertions.assertEquals(3, actual.length);

        Assertions.assertEquals("a", actual[0]);
        Assertions.assertEquals("b", actual[1]);
        Assertions.assertNull(actual[2]);
    }

    @Test
    public void testReadingArrayOfStringInDictionary() {
        byte[] data = new byte[]{1, 1, 0, 0, 0, 1, 9, 0, 0, 0, 115, 116, 114, 95, 97, 114, 114, 97, 121, 67, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0, 97, 1, 1, 0, 0, 0, 98};
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        Map<String, Object> dic = binary.readDictionary();
        Assertions.assertEquals(1, dic.size());

        String[] actual = (String[]) dic.get("str_array");
        Assertions.assertEquals(2, actual.length);

        Assertions.assertEquals("a", actual[0]);
        Assertions.assertEquals("b", actual[1]);
    }

    @Test
    public void testReadingNullDictionary() {
        byte[] data = new byte[]{0};
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        Map<String, Object> actual = binary.readDictionary();
        Assertions.assertNull(actual);
    }

    @Test
    public void testReadingEmptyDictionary() {
        byte[] data = new byte[]{1, 0, 0, 0, 0};
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        Map<String, Object> actual = binary.readDictionary();
        Assertions.assertEquals(0, actual.size());
    }

    @Test
    public void testReadingDictionary() {
        byte[] data = new byte[]{1, 6, 0, 0, 0, 1, 7, 0, 0, 0, 107, 101, 121, 95, 105, 110, 116, 8, 15, 0, 0, 0, 0, 0, 0, 0, 1, 18, 0, 0, 0, 107, 101, 121, 95, 102, 108, 111, 97, 116, 105, 110, 103, 95, 112, 111, 105, 110, 116, 33, 1, 31, (byte) 133, (byte) 235, 81, (byte) 184, 30, 9, 64, 1, 10, 0, 0, 0, 107, 101, 121, 95, 115, 116, 114, 105, 110, 103, 48, 1, 16, 0, 0, 0, 84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 115, 116, 114, 105, 110, 103, 1, 8, 0, 0, 0, 107, 101, 121, 95, 108, 105, 115, 116, 64, 1, 5, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 33, 1, 31, (byte) 133, (byte) 235, 81, (byte) 184, 30, 9, 64, 48, 1, 14, 0, 0, 0, 97, 110, 111, 116, 104, 101, 114, 95, 115, 116, 114, 105, 110, 103, 65, 1, 2, 0, 0, 0, 1, 6, 0, 0, 0, 115, 97, 109, 112, 108, 101, 48, 1, 3, 0, 0, 0, 111, 110, 101, 1, 7, 0, 0, 0, 97, 110, 111, 116, 104, 101, 114, 33, 1, 92, (byte) 143, (byte) 194, (byte) 245, 40, 92, 27, (byte) 192, 16, 1, 14, 0, 0, 0, 107, 101, 121, 95, 100, 105, 99, 116, 105, 111, 110, 97, 114, 121, 65, 1, 2, 0, 0, 0, 1, 8, 0, 0, 0, 111, 112, 116, 105, 111, 110, 95, 49, 48, 1, 4, 0, 0, 0, 74, 97, 118, 97, 1, 8, 0, 0, 0, 111, 112, 116, 105, 111, 110, 95, 50, 48, 1, 2, 0, 0, 0, 67, 35, 1, 8, 0, 0, 0, 110, 117, 108, 108, 95, 107, 101, 121, 16};
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        Map<String, Object> actual = binary.readDictionary();
        Assertions.assertEquals(6, actual.size());

        Assertions.assertEquals(15, (long) actual.get("key_int"));
        Assertions.assertEquals(3.14, (double) actual.get("key_floating_point"));
        Assertions.assertEquals("This is a string", actual.get("key_string"));
        Assertions.assertNull(actual.get("null_key"));

        Map<String, Object> subDictionary = (Map<String, Object>) actual.get("key_dictionary");
        Assertions.assertEquals(2, subDictionary.size());
        Assertions.assertEquals("Java", subDictionary.get("option_1"));
        Assertions.assertEquals("C#", subDictionary.get("option_2"));

        List<Object> subList = (List<Object>) actual.get("key_list");
        Assertions.assertEquals(5, subList.size());
        Assertions.assertEquals(1, (long) subList.get(0));
        Assertions.assertEquals(3.14, (double) subList.get(1));
        Assertions.assertEquals("another_string", subList.get(2));
        Assertions.assertNull(subList.get(4));

        Map<String, Object> listDictionary = (Map<String, Object>) subList.get(3);
        Assertions.assertEquals(2, listDictionary.size());
        Assertions.assertEquals("one", listDictionary.get("sample"));
        Assertions.assertEquals(-6.84, (double) listDictionary.get("another"));
    }

    private void checkArraySimilarity(byte[] expected, byte[] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], actual[i]);
        }
    }

    private void checkArraySimilarity(byte[][] expected, byte[][] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            checkArraySimilarity(expected[i], actual[i]);
        }
    }

    private void checkArraySimilarity(short[] expected, short[] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], actual[i]);
        }
    }

    private void checkArraySimilarity(int[] expected, int[] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], actual[i]);
        }
    }

    private void checkArraySimilarity(int[][] expected, int[][] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            checkArraySimilarity(expected[i], actual[i]);
        }
    }

    private void checkArraySimilarity(int[][][] expected, int[][][] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            checkArraySimilarity(expected[i], actual[i]);
        }
    }

    private void checkArraySimilarity(long[] expected, long[] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], actual[i]);
        }
    }

    private void checkArraySimilarity(float[] expected, float[] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], actual[i]);
        }
    }

    private void checkArraySimilarity(double[] expected, double[] actual) {
        Assertions.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], actual[i]);
        }
    }

    private String readStringFromByteArray(byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        return binary.readString();
    }

    private NumpyArray readNumpyArrayFromByteArray(byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        return binary.readNumpyArray();
    }
}
