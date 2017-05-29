package test;

import java.io.IOException;
import java.util.ArrayList;

import annNetwork.TrainingBackPropNet;
import data.DataReader;
import data.DataSet;
import data.InvalidDataset;

public class Test06 {

	//loads ann from file and trains
	public static void main(String[] args) throws InvalidDataset, IOException {
		
		DataReader d = new DataReader();
		d.readData("data/heart.dat");
		
		d.parseData();
		
		ArrayList<DataSet> data = d.Data();
		
		TrainingBackPropNet net = new TrainingBackPropNet();
		
		net.loadFromFile("data/net02.dat");
		
		for (DataSet ds:data)
		{
			net.addTrainingData(ds);
		}
		
		//net.createNetwork();

		//net.start(0.01f, 0.1f);
		net.getError();
		//net.saveToFile("data/net02.dat");
	}

}
