/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.libraries.numpy.wrappers;

import ai.sklearn4j.core.libraries.numpy.INumpyArrayWrapper;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;

import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;

public class Dim1DoubleNumpyWrapper implements INumpyArrayWrapper {
	private final double[] array;
	private final int[] shape;

	public Dim1DoubleNumpyWrapper(double[] array) {
		this.array = array;
		this.shape = new int[] {array.length};
	}

	@Override
	public int[] getShape() {
		return shape;
	}

	@Override
	public Object get(int... indices) {
		return array[indices[0]];
	}

	@Override
	public void set(Object value, int... indices) {
		this.array[indices[0]] = (double)value;
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
	public NumpyArray transpose() {
		double[] result = new double[shape[0]];

		for (int d0 = 0; d0 < this.shape[0]; d0++) {
			result[d0] = array[d0];

		}

		return NumpyArrayFactory.from(result);
	}
}