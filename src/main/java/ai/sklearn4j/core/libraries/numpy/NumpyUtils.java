package ai.sklearn4j.core.libraries.numpy;

public class NumpyUtils {
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
}
