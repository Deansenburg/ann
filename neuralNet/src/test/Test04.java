package test;

import java.io.IOException;
import java.util.ArrayList;

import annNetwork.TrainingBackPropNet;
import data.DataReader;
import data.DataSet;
import data.InvalidDataset;


public class Test04 {

	//recreates ann and starts training
	public static void main(String[] args) throws IOException, InvalidDataset {
		
		DataReader d = new DataReader();
		d.readData("data/heart.dat");
		
		d.parseData();
		
		ArrayList<DataSet> data = d.Data();
		
		TrainingBackPropNet net = new TrainingBackPropNet();
		
		int in = data.get(0).getInputs();
		int out = data.get(0).getOutputs();
		
		net.addInputLayer(in);
		net.addHiddenLayer(in);
		net.addOutputLayer(out);
		
		for (DataSet ds:data)
		{
			net.addTrainingData(ds);
		}
		
		net.setDebug(true);
		
		net.createNetwork();

		net.start(0.002f, 0.1f);
		net.getError();
		net.saveToFile("data/net03.net");
	}

}