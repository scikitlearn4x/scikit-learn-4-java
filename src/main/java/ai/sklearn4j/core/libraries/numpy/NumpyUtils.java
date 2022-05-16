package ai.sklearn4j.core.libraries.numpy;

public class NumpyUtils {
    public static final int SIZE_OF_INT_8 = 1;
    public static final int SIZE_OF_INT_16 = 2;
    public static final int SIZE_OF_INT_32 = 4;
    public static final int SIZE_OF_INT_64 = 8;
    public static final int SIZE_OF_FLOAT = 4;
    public static final int SIZE_OF_DOUBLE = 8;

    public static NumpyArray createArrayOfShapeAndTypeInfo(NumpyArray other) {
        return createArrayOfShapeAndTypeInfo(other.isFloatingPoint(), other.numberOfBytes(), other.getShape());
    }

    public static NumpyArray createArrayOfShapeAndTypeInfo(boolean isFloatingPoint, int size, int[] shape) {
        NumpyArray result = null;

        if (isFloatingPoint) {
            if (size == 8) {
                result = NumpyArrayFactory.arrayOfDoubleWithShape(shape);
            } else {
                result = NumpyArrayFactory.arrayOfFloatWithShape(shape);
            }
        } else {
            if (size == 8) {
                result = NumpyArrayFactory.arrayOfInt64WithShape(shape);
            } else if (size == 4) {
                result = NumpyArrayFactory.arrayOfInt32WithShape(shape);
            } else if (size == 2) {
                result = NumpyArrayFactory.arrayOfInt16WithShape(shape);
            } else if (size == 1) {
                result = NumpyArrayFactory.arrayOfInt8WithShape(shape);
            }
        }

        return result;
    }

    public static byte toByte(Object o) {
        byte result = 0;

        if (o instanceof Byte) {
            result = (byte) o;
        } else if (o instanceof Short) {
            result = (byte) ((short) o);
        } else if (o instanceof Integer) {
            result = (byte) ((int) o);
        } else if (o instanceof Long) {
            result = (byte) ((long) o);
        } else {
            throw new NumpyOperationException("Invalid casting value to byte.");
        }

        return result;
    }

    public static double toDouble(Object value) {
        return (double) value;
    }

    public static float toFloat(Object value) {
        return (float) value;
    }

    public static short toShort(Object o) {
        short result = 0;

        if (o instanceof Byte) {
            result = (short) o;
        } else if (o instanceof Short) {
            result = (short) ((short) o);
        } else if (o instanceof Integer) {
            result = (short) ((int) o);
        } else if (o instanceof Long) {
            result = (short) ((long) o);
        } else {
            throw new NumpyOperationException("Invalid casting value to short.");
        }

        return result;

    }

    public static int toInteger(Object o) {
        int result = 0;

        if (o instanceof Byte) {
            result = (int) o;
        } else if (o instanceof Short) {
            result = (int) ((short) o);
        } else if (o instanceof Integer) {
            result = (int) ((int) o);
        } else if (o instanceof Long) {
            result = (int) ((long) o);
        } else {
            throw new NumpyOperationException("Invalid casting value to int.");
        }

        return result;
    }

    public static long toLong(Object o) {
        long result = 0;

        if (o instanceof Byte) {
            result = (long) o;
        } else if (o instanceof Short) {
            result = (long) ((short) o);
        } else if (o instanceof Integer) {
            result = (long) ((int) o);
        } else if (o instanceof Long) {
            result = (long) ((long) o);
        } else {
            throw new NumpyOperationException("Invalid casting value to long.");
        }

        return result;

    }
}
