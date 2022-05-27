/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.libraries.numpy.wrappers;

import ai.sklearn4j.core.libraries.numpy.INumpyArrayWrapper;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.core.libraries.numpy.NumpyOperationException;


/**
 * A wrapper over a 7 dimensions double array for NumpyArray class.
 */
public class Dim7DoubleNumpyWrapper implements INumpyArrayWrapper {
	private final double[][][][][][][] array;
	private final int[] shape;

	/**
	 * Instantiate a new wrapper for 7 array.
	 * 
	 * @param array The underlying native array object.
	 */
	public Dim7DoubleNumpyWrapper(double[][][][][][][] array) {
		this.array = array;
		this.shape = new int[] {array.length, array[0].length, array[0][0].length, array[0][0][0].length, array[0][0][0][0].length, array[0][0][0][0][0].length, array[0][0][0][0][0][0].length};
	}

	@Override
	public int[] getShape() {
		return shape;
	}

	@Override
	public Object get(int... indices) {
		return array[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]][indices[6]];
	}

	@Override
	public void set(Object value, int... indices) {
		this.array[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]][indices[6]] = NumpyArrayFactory.toDouble(value);
	}

	/**
	 * Gets the underlying native array object.
	 * 
	 * @return The underlying native array object.
	 */
	public double[][][][][][][] getArray() {
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
		} else if (indices.length == 3) {
			result = NumpyArrayFactory.from(array[indices[0]][indices[1]][indices[2]]);
		} else if (indices.length == 4) {
			result = NumpyArrayFactory.from(array[indices[0]][indices[1]][indices[2]][indices[3]]);
		} else if (indices.length == 5) {
			result = NumpyArrayFactory.from(array[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]]);
		} else if (indices.length == 6) {
			result = NumpyArrayFactory.from(array[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]]);
		} else {
			throw new NumpyOperationException("Invalid slice for array specified.");
		}

		return result;
	}


	@Override
	public NumpyArray transpose() {
		double[][][][][][][] result = new double[shape[6]][shape[5]][shape[4]][shape[3]][shape[2]][shape[1]][shape[0]];

		for (int d0 = 0; d0 < this.shape[0]; d0++) {
			for (int d1 = 0; d1 < this.shape[1]; d1++) {
				for (int d2 = 0; d2 < this.shape[2]; d2++) {
					for (int d3 = 0; d3 < this.shape[3]; d3++) {
						for (int d4 = 0; d4 < this.shape[4]; d4++) {
							for (int d5 = 0; d5 < this.shape[5]; d5++) {
								for (int d6 = 0; d6 < this.shape[6]; d6++) {
									result[d6][d5][d4][d3][d2][d1][d0] = array[d0][d1][d2][d3][d4][d5][d6];

								}
							}
						}
					}
				}
			}
		}

		return NumpyArrayFactory.from(result);
	}
}