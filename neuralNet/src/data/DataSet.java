package data;

import java.util.ArrayList;

public class DataSet {

	private ArrayList<Float> input = new ArrayList<Float>();
	private ArrayList<Float> output = new ArrayList<Float>();

	public ArrayList<Float> Input() {
		return input;
	}

	public ArrayList<Float> Output() {
		return output;
	}

	public void addInput(float x) {
		input.add(x);
	}

	public void addOutput(float x) {
		output.add(x);
	}

	public int getInputs() {
		return input.size();
	}

	public int getOutputs() {
		return output.size();
	}

	//check if data is consistent
	public void checkInputs(int num) throws InvalidDataset {
		if (input.size() != num) {
			throw new InvalidDataset("Input");
		}
	}

	public void checkOutputs(int num) throws InvalidDataset {
		if (output.size() != num) {
			throw new InvalidDataset("Output");
		}
	}

	@Override
	public String toString() {
		return input.toString() + output.toString();
	}
}
