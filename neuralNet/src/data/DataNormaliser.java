package data;

import java.util.ArrayList;

public class DataNormaliser {
	
	public static float max(ArrayList<DataSet> data, int pos) {
		float maxVal = -1;
		for (DataSet d : data) {
			float val;
			if (pos < d.getInputs())// in inputs
			{
				val = d.Input().get(pos);
			} else {
				val = d.Output().get(pos - d.getInputs());
			}
			if (val > maxVal) {
				maxVal = val;
			}
		}
		return maxVal;
	}
	
	public static float max(Float[] data)
	{
		float maxVal = -1;
		for (float f:data)
		{
			if (f > maxVal) {
				maxVal = f;
			}
		}
		return maxVal;
	}

	public static float min(ArrayList<DataSet> data, int pos) {
		float minVal = -1;
		for (DataSet d : data) {
			float val;
			if (pos < d.getInputs())// in inputs
			{
				val = d.Input().get(pos);
			} else {
				val = d.Output().get(pos - d.getInputs());
			}
			if (val < minVal || minVal == -1) {
				minVal = val;
			}
		}
		return minVal;
	}
	
	public static float min(Float[] data)
	{
		float minVal = -1;
		for (float f:data)
		{
			if (f < minVal || minVal == -1) {
				minVal = f;
			}
		}
		return minVal;
	}

	public static void normalise(ArrayList<DataSet> data,
			ArrayList<DataSet> normData, float minVal, float maxVal, int pos) {

		int place = 0;
		for (DataSet d : data) {
			float val;
			if (pos < d.getInputs())// in inputs
			{
				val = d.Input().get(pos);
				normData.get(place).addInput(normalise(minVal, maxVal, val));
			} else {
				val = d.Output().get(pos - d.getInputs());
				normData.get(place).addOutput(normalise(minVal, maxVal, val));
			}
			place++;
		}
	}
	
	public static void normalise(Float[] data, float minVal, float maxVal)
	{
		for (int i=0;i<data.length;i++)
		{
			data[i] = normalise(minVal, maxVal, data[i]);
		}
	}
	
	public static float normalise(float min, float max, float val) {
		float norm = (val - min) / (max - min);
		if (Float.isNaN(norm)) {
			return 0;
		}
		return norm;
	}
	
	public static float denormalise(float min, float max, float norm)
	{
		return norm*(max-min)+min;
	}
}
