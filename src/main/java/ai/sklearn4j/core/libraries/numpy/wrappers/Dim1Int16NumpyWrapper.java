/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.libraries.numpy.wrappers;

import ai.sklearn4j.core.libraries.numpy.INumpyArrayWrapper;

public class Dim1Int16NumpyWrapper implements INumpyArrayWrapper {
	private final short[] array;
	private final int[] shape;

	public Dim1Int16NumpyWrapper(short[] array) {
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
		this.array[indices[0]] = (short)value;
	}

	public boolean isFloatingPoint() {
		return false;
	}


	public int numberOfBits() {
		return 16;
	}
}