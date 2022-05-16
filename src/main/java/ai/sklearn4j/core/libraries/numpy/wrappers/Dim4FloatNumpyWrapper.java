/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.libraries.numpy.wrappers;

import ai.sklearn4j.core.libraries.numpy.INumpyArrayWrapper;

public class Dim4FloatNumpyWrapper implements INumpyArrayWrapper {
	private final float[][][][] array;
	private final int[] shape;

	public Dim4FloatNumpyWrapper(float[][][][] array) {
		this.array = array;
		this.shape = new int[] {array.length, array[0].length, array[0][0].length, array[0][0][0].length};
	}

	@Override
	public int[] getShape() {
		return shape;
	}

	@Override
	public Object get(int... indices) {
		return array[indices[0]][indices[1]][indices[2]][indices[3]];
	}

	@Override
	public void set(Object value, int... indices) {
		this.array[indices[0]][indices[1]][indices[2]][indices[3]] = (float)value;
	}

	public boolean isFloatingPoint() {

		return true;
	}


	public int numberOfBits() {

		return 32;
	}
}