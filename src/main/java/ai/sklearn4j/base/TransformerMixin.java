package ai.sklearn4j.base;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;

/**
 * Mixin class for all transformers in scikit-learn.
 *
 * @param <InputType> Type of the input for the transformation.
 * @param <OutputType> Type of the output for the transformation.
 */
public abstract class TransformerMixin<InputType, OutputType> {
    /**
     * Takes the input array and transforms it.
     * @param array The array to transform.
     * @return The transformed array.
     */
    public abstract OutputType transform(InputType array);

    /**
     * Takes a transformed array and reveres the transformation.
     * @param array The array to apply reveres transform.
     * @return The inversed transform of array.
     */
    public abstract InputType inverseTransform(OutputType array);
}
