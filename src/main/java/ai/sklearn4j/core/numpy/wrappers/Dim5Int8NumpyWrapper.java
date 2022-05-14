/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.numpy.wrappers;

import ai.sklearn4j.core.numpy.INumpyArrayWrapper;

public class Dim5Int8NumpyWrapper implements INumpyArrayWrapper {
	private final byte[][][][][] array;
	private final int[] shape;

	public Dim5Int8NumpyWrapper(byte[][][][][] array) {
		this.array = array;
		this.shape = new int[] {array.length, array[0].length, array[0][0].length, array[0][0][0].length, array[0][0][0][0].length};
	}

	@Override
	public int[] getShape() {
		return shape;
	}

	@Override
	public Object get(int... indices) {
		return array[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]];
	}

	@Override
	public void set(Object value, int... indices) {
		this.array[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]] = (byte)value;
	}

}