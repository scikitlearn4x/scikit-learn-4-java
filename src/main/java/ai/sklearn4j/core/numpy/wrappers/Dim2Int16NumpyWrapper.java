/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.numpy.wrappers;

import ai.sklearn4j.core.numpy.INumpyArrayWrapper;

public class Dim2Int16NumpyWrapper implements INumpyArrayWrapper {
	private final short[][] array;
	private final int[] shape;

	public Dim2Int16NumpyWrapper(short[][] array) {
		this.array = array;
		this.shape = new int[] {array.length, array[0].length};
	}

	@Override
	public int[] getShape() {
		return shape;
	}

	@Override
	public Object get(int... indices) {
		return array[indices[0]][indices[1]];
	}

	@Override
	public void set(Object value, int... indices) {
		this.array[indices[0]][indices[1]] = (short)value;
	}

}