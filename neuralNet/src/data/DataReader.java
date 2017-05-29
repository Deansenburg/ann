package data;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class DataReader {

	private ArrayList<DataSet> data;
	private ArrayList<DataSet> normData;

	public void readData(String path) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(path));
		data = new ArrayList<DataSet>();
		normData = new ArrayList<DataSet>();
		
		try {
			String line = br.readLine();

			while (line != null) {
				String[] items = line.split("\\s++");
				DataSet ds = new DataSet();
				for (int i = 0; i < items.length - 1; i++) {
					ds.addInput(Float.parseFloat(items[i]));
				}
				ds.addOutput(Float.parseFloat(items[items.length - 1]));
				data.add(ds);
				line = br.readLine();
			}
		} finally {
			br.close();
		}
		normaliseData();
	}

	private void normaliseData() {
		if (data.size() == 0) return;
		int numOfElements = data.get(0).getInputs() + data.get(0).getOutputs();
		
		for (int i = 0; i < data.size(); i++) {
			normData.add(new DataSet());
		}
		
		for (int i = 0; i < numOfElements; i++) {
			// find max
			float maxVal = DataNormaliser.max(data, i);
			// System.out.println(maxVal);
			// find min
			float minVal = DataNormaliser.min(data, i);
			// System.out.println(minVal);
			// normalise data
			DataNormaliser.normalise(data, normData, minVal, maxVal, i);
		}
	}
	
	public void parseData() {
		for (DataSet d : normData) {
			System.out.println("In: " + d.Input() + " Out: " + d.Output());
		}
	}

	public ArrayList<DataSet> Data() {
		return normData;
	}
	
	public ArrayList<DataSet> DataSource()
	{
		return data;
	}

}
