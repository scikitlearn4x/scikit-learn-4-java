/*
 * This file was automatically generated, don't edit it manually.
 */

package ai.sklearn4j.core.libraries.numpy.wrappers;

import ai.sklearn4j.core.libraries.numpy.INumpyArrayWrapper;

public class Dim10Int16NumpyWrapper implements INumpyArrayWrapper {
	private final short[][][][][][][][][][] array;
	private final int[] shape;

	public Dim10Int16NumpyWrapper(short[][][][][][][][][][] array) {
		this.array = array;
		this.shape = new int[] {array.length, array[0].length, array[0][0].length, array[0][0][0].length, array[0][0][0][0].length, array[0][0][0][0][0].length, array[0][0][0][0][0][0].length, array[0][0][0][0][0][0][0].length, array[0][0][0][0][0][0][0][0].length, array[0][0][0][0][0][0][0][0][0].length};
	}

	@Override
	public int[] getShape() {
		return shape;
	}

	@Override
	public Object get(int... indices) {
		return array[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]][indices[6]][indices[7]][indices[8]][indices[9]];
	}

	@Override
	public void set(Object value, int... indices) {
		this.array[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]][indices[6]][indices[7]][indices[8]][indices[9]] = (short)value;
	}

	public boolean isFloatingPoint() {

		return false;
	}


	public int numberOfBits() {

		return 16;
	}
}