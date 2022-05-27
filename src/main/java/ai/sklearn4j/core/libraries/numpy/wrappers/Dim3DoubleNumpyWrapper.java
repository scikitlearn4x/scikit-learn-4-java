/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.libraries.numpy.wrappers;

import ai.sklearn4j.core.libraries.numpy.INumpyArrayWrapper;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.core.libraries.numpy.NumpyOperationException;

public class Dim3DoubleNumpyWrapper implements INumpyArrayWrapper {
    private final double[][][] array;
    private final int[] shape;

    public Dim3DoubleNumpyWrapper(double[][][] array) {
        this.array = array;
        this.shape = new int[]{array.length, array[0].length, array[0][0].length};
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public Object get(int... indices) {
        return array[indices[0]][indices[1]][indices[2]];
    }

    @Override
    public void set(Object value, int... indices) {
        this.array[indices[0]][indices[1]][indices[2]] = NumpyArrayFactory.toDouble(value);
    }

    public double[][][] getArray() {
        return this.array;
    }

    @Override
    public boolean isFloatingPoint() {
        return true;
    }


    @Override
    public int numberOfBits() {
        return 64;
    }


    @Override
    public Object getRawArray() {
        return array;
    }


    @Override
    public NumpyArray wrapInnerSubsetArray(int... indices) {
        NumpyArray result = null;

        if (indices.length == 1) {
            result = NumpyArrayFactory.from(array[indices[0]]);
        } else if (indices.length == 2) {
            result = NumpyArrayFactory.from(array[indices[0]][indices[1]]);
        } else {
            throw new NumpyOperationException("Invalid slice for array specified.");
        }

        return result;
    }


    @Override
    public NumpyArray transpose() {
        double[][][] result = new double[shape[2]][shape[1]][shape[0]];

        for (int d0 = 0; d0 < this.shape[0]; d0++) {
            for (int d1 = 0; d1 < this.shape[1]; d1++) {
                for (int d2 = 0; d2 < this.shape[2]; d2++) {
                    result[d2][d1][d0] = array[d0][d1][d2];

                }
            }
        }

        return NumpyArrayFactory.from(result);
    }
}