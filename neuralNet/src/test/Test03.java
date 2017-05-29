package test;
import annNetwork.TrainingBackPropNet;
import data.DataSet;
import data.InvalidDataset;


public class Test03 {
	public static void main(String[] args) {
		TrainingBackPropNet net = new TrainingBackPropNet();

		int inputSize = 3;
		int outputSize = 2;
		
		DataSet d1 = new DataSet();
		d1.addInput(0);
		d1.addInput(1);
		d1.addInput(0.3f);
		d1.addOutput(0.8f);
		d1.addOutput(0.02f);
		
		DataSet d2 = new DataSet();
		d2.addInput(1f);
		d2.addInput(0.5f);
		d2.addInput(0f);
		d2.addOutput(0.5f);
		d2.addOutput(0.6f);
		
		DataSet d3 = new DataSet();
		d3.addInput(0.1f);
		d3.addInput(0.05f);
		d3.addInput(0.9f);
		d3.addOutput(0.9f);
		d3.addOutput(0.1f);
		
		net.addInputLayer(inputSize);
		net.addHiddenLayer(4);
		net.addOutputLayer(outputSize);
		
		try {
			net.addTrainingData(d1);
			net.addTrainingData(d2);
			net.addTrainingData(d3);
		} catch (InvalidDataset e) {
			e.printStackTrace();
		}
		net.createNetwork();
		net.start(0.001f, 0.5f);
	}
}
