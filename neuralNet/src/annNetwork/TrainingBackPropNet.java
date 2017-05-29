package annNetwork;

import java.util.ArrayList;

import data.DataSet;
import data.InvalidDataset;



public class TrainingBackPropNet extends BaseBackPropNet {

	private ArrayList<DataSet> trainingData = new ArrayList<DataSet>();

	public void addTrainingData(DataSet ds) throws InvalidDataset {
		//makes sure that training data is of same format as previous ones
		//does not work with different ins/outs
		ds.checkInputs(ins.size());
		ds.checkOutputs(outs.size());
		trainingData.add(ds);
	}

	//debug thing
	public void getError()
	{
		float right = 0;
		float wrong = 0;
		for (DataSet d : trainingData) {
			addInputData(d.Input());
			addOutputData(d.Output());

			Float[] out = outputNetwork();
			if (out[0] < 0.01f) {
				right++;
			} else {
				wrong++;
			}
			System.out.println(out[0]+" "+out[1]);
		}
		
		System.out.println(wrong+" : "+right);
	}
	
	public float getError(float tErr, float lRate) {
		float totalE = 0;
		// System.out.println("--------------------------------");
		for (DataSet d : trainingData) {
			//populate in/out data
			addInputData(d.Input());
			addOutputData(d.Output());

			Float out = singleNoBack(lRate);

			// System.out.println("Out----------------- " + out);
			totalE += out;
		}
		//totalE = wrong / (wrong + right);
		// /System.out.println("--------------------------------");
		totalE /= trainingData.size();
		//System.out.println("E: " + totalE + " W:"+wrong+" R:"+right);
		System.out.println(totalE);//avg error
		return totalE;
	}

	//to train more than one data entry
	@Override
	public void startLoop(float tErr, float lRate) {
		float totalError = 0;
		do {
			for (DataSet d : trainingData) {
				addInputData(d.Input());
				addOutputData(d.Output());
				float curError = singleNoBack(lRate);
				// System.out.println("Single Before "+curError);
				//only needs to train if err is less than threshold
				//will avoid over training
				if (curError > tErr) {
					float e = singleRun(lRate);//includes backprop
					System.out.println("Train "+curError + " : "+e);
				}
			}
			render();
			totalError = getError(tErr, lRate);
			// System.out.println("Total " + totalError);
		} while (totalError > tErr);
	}
}