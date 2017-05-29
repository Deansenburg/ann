package test;
import java.util.ArrayList;
import java.util.Arrays;

import annNetwork.BaseBackPropNet;

public class Test02 {

	public static void main(String[] args) {
		BaseBackPropNet net = new BaseBackPropNet();

		ArrayList<Float> inputData = new ArrayList<Float>(Arrays.asList(1f,
				0.5f, 0f));
		ArrayList<Float> outputData = new ArrayList<Float>(Arrays.asList(0.04f,
				0.1f));

		net.addInputLayer(inputData.size());
		net.addHiddenLayer(4);
		net.addOutputLayer(outputData.size());

		net.addInputData(inputData);
		net.addOutputData(outputData);

		net.createNetwork();

		net.start(0.000001f, 0.5f);
	}
}
