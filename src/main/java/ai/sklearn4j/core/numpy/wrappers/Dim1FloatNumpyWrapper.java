/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.numpy.wrappers;

import ai.sklearn4j.core.numpy.INumpyArrayWrapper;

public class Dim1FloatNumpyWrapper implements INumpyArrayWrapper {
	private final float[] array;
	private final int[] shape;

	public Dim1FloatNumpyWrapper(float[] array) {
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
		this.array[indices[0]] = (float)value;
	}

}