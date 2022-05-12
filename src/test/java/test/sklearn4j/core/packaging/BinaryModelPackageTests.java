package test.sklearn4j.core.packaging;

import ai.sklearn4j.core.packaging.BinaryModelPackage;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class BinaryModelPackageTests {
    @Test
    public void testAppendFloatPiPositive() {
        byte[] data = new byte[] {86, 14, 73, 64};
        float expected = 3.1415f;

        float actual = readFloatFromByteArray(data);

        Assertions.assertEquals(expected, actual);
    }

    private float readFloatFromByteArray(byte[] data) {
        InputStream stream = new ByteArrayInputStream(data);
        BinaryModelPackage binary = BinaryModelPackage.fromStream(stream);

        return binary.readFloat();
    }
}
