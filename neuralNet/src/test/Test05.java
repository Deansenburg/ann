package test;

import java.io.IOException;
import java.util.ArrayList;

import annNetwork.BaseBackPropNet;
import data.DataReader;
import data.DataSet;


public class Test05 {

	//load ann and test specific data item
	public static void main(String[] args) throws IOException {
		
		DataReader d = new DataReader();
		d.readData("data/heart.dat");
		
		d.parseData();
		
		ArrayList<DataSet> data = d.Data();
		
		BaseBackPropNet net = new BaseBackPropNet();
		
		net.loadFromFile("data/net.dat");

		int dataItem = 100;
		
		net.addInputData(data.get(dataItem).Input());
		net.addOutputData(data.get(dataItem).Output());
		
		net.outputNetwork();
		//net.start(1f, 0.5f);
		//net.singleRun(0.5f);
	}

}
