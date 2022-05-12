package test.sklearn4j.core.packaging;

import ai.sklearn4j.core.packaging.BinaryModelPackage;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class BinaryModelPackageBaseTypesTests {
    private static final double DOUBLE_COMPARE_EPSILON = 0.001;

    @Test
    public void testReadFloatNaN() {
        byte[] data = new byte[]{0};
        float expected = Float.NaN;

        float actual = readFloatFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadDoubleNaN() {
        byte[] data = new byte[]{0};
        double expected = Float.NaN;

        double actual = readDoubleFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadFloatPiPositive() {
        byte[] data = new byte[]{1, 86, 14, 73, 64};
        float expected = 3.1415f;

        float actual = readFloatFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadFloatPiNegative() {
        byte[] data = new byte[]{1, 86, 14, 73, -64};
        float expected = -3.1415f;

        float actual = readFloatFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadFloatEPositive() {
        byte[] data = new byte[]{1, 77, -8, 45, 64};
        float expected = 2.71828f;

        float actual = readFloatFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadFloatENegative() {
        byte[] data = new byte[]{1, 77, -8, 45, -64};
        float expected = -2.71828f;

        float actual = readFloatFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadDoublePiPositive() {
        byte[] data = new byte[]{1, 111, 18, -125, -64, -54, 33, 9, 64};
        double expected = 3.1415f;

        double actual = readDoubleFromByteArray(data);
        Assertions.assertTrue(Math.abs(expected - actual) < DOUBLE_COMPARE_EPSILON);
    }

    @Test
    public void testReadDoublePiNegative() {
        byte[] data = new byte[]{1, 111, 18, -125, -64, -54, 33, 9, -64};
        double expected = -3.1415f;

        double actual = readDoubleFromByteArray(data);
        Assertions.assertTrue(Math.abs(expected - actual) < DOUBLE_COMPARE_EPSILON);
    }

    @Test
    public void testReadDoubleEPositive() {
        byte[] data = new byte[]{1, -112, -9, -86, -107, 9, -65, 5, 64};
        double expected = 2.71828f;

        double actual = readDoubleFromByteArray(data);
        Assertions.assertTrue(Math.abs(expected - actual) < DOUBLE_COMPARE_EPSILON);
    }

    @Test
    public void testReadDoubleENegative() {
        byte[] data = new byte[]{1, -112, -9, -86, -107, 9, -65, 5, -64};
        double expected = -2.71828f;

        double actual = readDoubleFromByteArray(data);
        Assertions.assertTrue(Math.abs(expected - actual) < DOUBLE_COMPARE_EPSILON);
    }

    @Test
    public void testReadBytePositive() {
        byte[] data = new byte[]{23};
        int expected = 23;

        int actual = readByteFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadByteNegative() {
        byte[] data = new byte[]{-8};
        int expected = -8;

        int actual = readByteFromByteArray(data);
        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void testReadShortIntegers() {
        testReadShortInteger(10, new byte[]{10, 0});
        testReadShortInteger(567, new byte[]{55, 2});
        testReadShortInteger(16000, new byte[]{-128, 62});
        testReadShortInteger(-16000, new byte[]{-128, -63});
    }

    @Test
    public void testReadIntegers() {
        testReadInteger(10, new byte[]{10, 0, 0, 0});
        testReadInteger(567, new byte[]{55, 2, 0, 0});
        testReadInteger(16000, new byte[]{-128, 62, 0, 0});
        testReadInteger(59, new byte[]{59, 0, 0, 0});
        testReadInteger(-59, new byte[]{-59, -1, -1, -1});
        testReadInteger(300, new byte[]{44, 1, 0, 0});
        testReadInteger(-300, new byte[]{-44, -2, -1, -1});
        testReadInteger(2000000000, new byte[]{0, -108, 53, 119});
        testReadInteger(-2000000000, new byte[]{0, 108, -54, -120});
    }

    @Test
    public void testReadLongIntegers() {
        testReadLongInteger(10, new byte[]{10, 0, 0, 0, 0, 0, 0, 0});
        testReadLongInteger(567, new byte[]{55, 2, 0, 0, 0, 0, 0, 0});
        testReadLongInteger(16000, new byte[]{-128, 62, 0, 0, 0, 0, 0, 0});
        testReadLongInteger(59, new byte[]{59, 0, 0, 0, 0, 0, 0, 0});
        testReadLongInteger(-59, new byte[]{-59, -1, -1, -1, -1, -1, -1, -1});
        testReadLongInteger(300, new byte[]{44, 1, 0, 0, 0, 0, 0, 0});
        testReadLongInteger(-300, new byte[]{-44, -2, -1, -1, -1, -1, -1, -1});
        testReadLongInteger(2000000000, new byte[]{0, -108, 53, 119, 0, 0, 0, 0});
        testReadLongInteger(-2000000000, new byte[]{0, 108, -54, -120, -1, -1, -1, -1});
        testReadLongInteger(200000000000000000L, new byte[]{0, 0, 20, -69, -16, -118, -58, 2});
        testReadLongInteger(-200000000000000000L, new byte[]{0, 0, -20, 68, 15, 117, 57, -3});
    }

    private void testReadShortInteger(int expected, byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        int actual = binary.readShort();
        Assertions.assertEquals(expected, actual);
    }

    private void testReadInteger(int expected, byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        int actual = binary.readInteger();
        Assertions.assertEquals(expected, actual);
    }

    private void testReadLongInteger(long expected, byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        long actual = binary.readLongInteger();
        Assertions.assertEquals(expected, actual);
    }

    private float readFloatFromByteArray(byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        return binary.readFloat();
    }

    private double readDoubleFromByteArray(byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        return binary.readDouble();
    }

    private int readByteFromByteArray(byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        return binary.readByte();
    }
}
