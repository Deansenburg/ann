package test;
import java.util.ArrayList;

import javax.swing.JFrame;

import annModel.Layer;
import annModel.NeuralConnection;
import annModel.Neuron;
import annNetwork.Network;
import annView.NetworkView;

public class Test01 {

	public ArrayList<Neuron> ins = new ArrayList<Neuron>();

	public ArrayList<Neuron> hiddens = new ArrayList<Neuron>();
	public ArrayList<Neuron> hiddens2 = new ArrayList<Neuron>();

	public ArrayList<Neuron> outs = new ArrayList<Neuron>();

	public static void main(String[] args) {
		Test01 t = new Test01();

		t.setNumberOfNeurons(t.ins, 2, 0);
		t.setNumberOfNeurons(t.hiddens, 2, 10);
		t.setNumberOfNeurons(t.hiddens2, 2, 20);
		t.setNumberOfNeurons(t.outs, 2, 30);

		float in = 0.05f;
		for (Neuron n : t.ins) {
			n.setOutput(in);
			in += 0.05f;
		}
		float target = 0.01f;
		for (Neuron n : t.outs) {
			n.setTarget(target);
			target += 0.98f;
		}

		ArrayList<NeuralConnection> cons = t.getConnections(t.ins, t.hiddens);
		t.addBias(cons, t.hiddens, 1, 0, 10, 0.35f);
		Layer l1 = new Layer(cons);

		t.weight += 0.15f;
		cons = t.getConnections(t.hiddens, t.hiddens2);
		t.addBias(cons, t.hiddens2, 1, 10, 10, 0.35f);
		Layer l2 = new Layer(cons);

		cons = t.getConnections(t.hiddens2, t.outs);
		t.addBias(cons, t.outs, 1, 20, 10, 0.6f);
		Layer l3 = new Layer(cons);

		ArrayList<Layer> networkLayers = new ArrayList<Layer>();
		networkLayers.add(l1);
		networkLayers.add(l2);
		networkLayers.add(l3);

		Network n = new Network(networkLayers);

		// float v1 = 0.3775f;
		// double v = 1d/(Math.exp(-v1)+1);

		JFrame frame = new JFrame();
		NetworkView view = new NetworkView(n); 
		frame.add(view);
		frame.pack();
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		float error = 0;
		do {
			t.forwardPass(n);
			error = t.getTotalError(t.outs);
			t.backwardPass(n, 0.5f);
			t.resetNodes(n);
		} while (error > 0.0000001f);
		t.forwardPass(n);

		view.render(true);
		frame.repaint();
	}

	public void resetNodes(Network n) {
		for (Layer l:n.LayersForward()) {
			for (NeuralConnection nc:l.Connections()) {
				nc.update();
				if (nc.Forward().canReset()) {
					nc.Forward().reset();
				}
			}
		}

	}

	public void backwardPass(Network n, float lRate) {
	
		boolean first = true;
		for (Layer l:n.LayersBackward()) {
			if (first)
			{
				// output layer
				backwardPassOutput(l, lRate);
				first = false;
				continue;
			}
			// hidden layer
			// goto next connection
			for (NeuralConnection nC:l.Connections()) {
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

	// 100% correct -- although biases should not be updated
	public void backwardPassOutput(Layer l, float lRate) {
		for (NeuralConnection nC:l.Connections()) {
			
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

	// 100% correct
	public float getTotalError(ArrayList<Neuron> neurons) {
		float totalError = 0;
		for (int i = 0; i < neurons.size(); i++) {
			/*
			 * System.out.println(neurons.get(i).Target() + " - " +
			 * neurons.get(i).Output());
			 */
			totalError += 0.5 * Math.pow(
					neurons.get(i).Target() - neurons.get(i).Output(), 2);
		}
		//System.out.println("Error " + totalError);
		return totalError;
	}

	// 100% correct
	public void forwardPass(Network n) {
		for (Layer l:n.LayersForward()) {
			for (NeuralConnection nC:l.Connections()) {
				// look for 10 : 0
				float out = nC.Backward().Output();
				nC.Forward().addInput(out * nC.Weight());
				// System.out.println("Net " + nC.Forward() + " " +
				// nC.Forward().Net());
				// System.out.println("Out "+nC.Forward()+
				// " "+nC.Forward().Output());
			}
		}
	}

	public float weight = 0.15f;

	public ArrayList<NeuralConnection> getConnections(ArrayList<Neuron> a,
			ArrayList<Neuron> b) {
		ArrayList<NeuralConnection> cons = new ArrayList<NeuralConnection>();
		for (Neuron n1 : a) {
			for (Neuron n2 : b) {
				NeuralConnection nC;
				// float rValue = (r.nextInt(10) - 4) / 10f;
				// System.out.println(rValue);
				cons.add(nC = new NeuralConnection(n2, n1, weight));
				n1.addConnection(nC);
				weight += 0.1f;
			}
			weight -= 0.15f;
		}

		return cons;
	}

	public void addBias(ArrayList<NeuralConnection> cons,
			ArrayList<Neuron> neurs, float bVal, int x, int y, float weight) {
		for (Neuron n : neurs) {
			Neuron bias = new Neuron(x, y);
			bias.setOutput(bVal);
			cons.add(new NeuralConnection(n, bias, weight));
		}
	}

	public void setNumberOfNeurons(ArrayList<Neuron> n, int size, int x) {
		int y = 0;
		for (int i = 0; i < size; i++) {
			n.add(new Neuron(x, y));
			y += 5;
		}
	}

}
