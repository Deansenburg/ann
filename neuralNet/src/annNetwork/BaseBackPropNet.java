package annNetwork;

import annModel.Layer;
import annModel.NeuralConnection;
import annModel.Neuron;

public class BaseBackPropNet extends BaseNetwork{

	// only one layer of in/out

	@Override
	public float singleRun(float lRate) {
		forwardPass(network);// get output
		backwardPass(network, lRate);// adjust
		resetNodes(network);// updates weights
		forwardPass(network);// get output to get error
		float error = getTotalError(outs);
		return error;
	}
	
	public float singleNoBack(float lRate)
	{
		return super.singleRun(lRate);
	}

	private void backwardPass(Network n, float lRate) {

		boolean first = true;
		for (Layer l : n.LayersBackward()) {
			if (first) {
				// output layer
				backwardPassOutput(l, lRate);
				first = false;
				continue;
			}
			// hidden layer
			// goto next connection
			for (NeuralConnection nC : l.Connections()) {
				if (nC.Backward().isBias())
				{
					continue;
				}
				float sum = 0;
				for (NeuralConnection con : nC.Forward().ForwardConnections()) {
					Neuron out = con.Forward();
					float error = out.Output() - out.Target();
					float der = out.Output() * (1 - out.Output());
					// System.out.println(out+" Error "+error+" Der "+der);
					sum += (error * der * con.Weight());
					// System.out.println(out+" "+error * der * con.Weight());
				}
				// System.out.println(sum);
				Neuron h1 = nC.Forward();
				float der = h1.Output() * (1 - h1.Output());
				// System.out.println("Output "+der);
				float nodeDelta = sum * der * nC.Backward().Output();
				// System.out.println(sum+" "+der+" "+nC.Backward().Output());
				// System.out.println(nodeDelta);
				// System.out.println("Before: "+nC.Weight());
				float w = nC.Weight() - (lRate * nodeDelta);
				nC.setWeight(w);
				float t = nC.Forward().Target() - (lRate * nodeDelta);
				nC.Forward().setTarget(t);
				// System.out.println("After : "+nC.Weight());
				// System.out.println(nC.Weight() - (lRate * nodeDelta));
			}
		}
	}

	// only does output layer
	private void backwardPassOutput(Layer l, float lRate) {
		for (NeuralConnection nC : l.Connections()) {
			if (nC.Backward().isBias())
			{
				continue;
			}
			Neuron out = nC.Forward();
			Neuron prev = nC.Backward();

			float error = out.Output() - out.Target();
			// System.out.println("Error "+out+" "+error);
			float logFuncDer = out.Output() * (1 - out.Output());
			// System.out.println("Derivative "+out+" "+logFuncDer);
			float prevOut = prev.Output();
			// System.out.println("Out "+out+" "+prevOut);

			float nodeDelta = error * logFuncDer * prevOut;
			// System.out.println("Delta "+out+" "+nodeDelta);
			nC.setWeight(nC.Weight() - (nodeDelta * lRate));
			// System.out.println("Weight"+out+" "+nC.Weight());
		}

	}
}
