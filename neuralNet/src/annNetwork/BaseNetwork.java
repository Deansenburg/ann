package annNetwork;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import annModel.Layer;
import annModel.NeuralConnection;
import annModel.Neuron;
import annView.NetworkView;

public class BaseNetwork {

	protected ArrayList<Neuron> ins = new ArrayList<Neuron>();
	protected ArrayList<Neuron> outs = new ArrayList<Neuron>();

	// multiple layers of hidden
	protected ArrayList<ArrayList<Neuron>> hiddens = new ArrayList<ArrayList<Neuron>>();

	protected ArrayList<Neuron> bias = new ArrayList<Neuron>();

	// for visualisation reasons - keeps track of column
	private int x = 0;

	protected Network network;

	private NetworkView view;
	private boolean debug = true;

	private JFrame frame = null;

	public void setDebug(boolean b) {
		debug = b;
	}

	// only to create a network - not to be used with loadFromFile
	public void createNetwork() {
		ArrayList<Layer> netLayers = new ArrayList<Layer>();

		if (hiddens.size() == 0) {
			netLayers.add(getLayer(ins, outs));
		} else {
			// input to first hidden layer
			netLayers.add(getLayer(ins, hiddens.get(0)));
			// last hidden to output layers
			netLayers.add(getLayer(hiddens.get(hiddens.size() - 1), outs));
			for (int i = 0; i < hiddens.size() - 1; i++) {
				// every hidden in pairs moving forward
				netLayers.add(getLayer(hiddens.get(i), hiddens.get(i + 1)));
			}
		}
		network = new Network(netLayers);
		view = new NetworkView(network);
	}

	public void display()
	{
		startView();
		singleRun(-1);
		render();
	}
	
	public void addInputLayer(int num) {
		if (ins.size() == 0) {
			setNumberOfNeurons(ins, num, x);
			x += 10;
		}
	}

	public void addHiddenLayer(int num) {
		ArrayList<Neuron> h = new ArrayList<Neuron>();
		setNumberOfNeurons(h, num, x);
		x += 10;
		hiddens.add(h);
	}

	public void addOutputLayer(int num) {
		if (outs.size() == 0) {
			setNumberOfNeurons(outs, num, x);
			x += 5;
		}
	}

	public void addInputData(ArrayList<Float> data) {
		for (int i = 0; i < data.size(); i++) {
			ins.get(i).setOutput(data.get(i));
		}
	}

	public void addOutputData(ArrayList<Float> data) {
		for (int i = 0; i < data.size(); i++) {
			outs.get(i).setTarget(data.get(i));
		}
	}

	private void setNumberOfNeurons(ArrayList<Neuron> n, int size, int x) {
		int y = 0;
		for (int i = 0; i < size; i++) {
			n.add(new Neuron(x, y));
			y += 3;
		}
	}

	private ArrayList<NeuralConnection> getConnections(ArrayList<Neuron> a,
			ArrayList<Neuron> b) {
		ArrayList<NeuralConnection> cons = new ArrayList<NeuralConnection>();
		Random r = new Random();
		for (Neuron n1 : a) {
			for (Neuron n2 : b) {
				NeuralConnection nC;
				float rValue = (r.nextInt(10) - 4) / 10f;
				// System.out.println(rValue);
				cons.add(nC = new NeuralConnection(n2, n1, rValue));
				n1.addConnection(nC);
			}
		}
		return cons;
	}

	private void addBias(ArrayList<NeuralConnection> cons,
			ArrayList<Neuron> neurs, float bVal, int x, int y, float weight) {
		Neuron bias = new Neuron(x, y);
		bias.setOutput(bVal);
		bias.setBias(true);
		this.bias.add(bias);
		for (Neuron n : neurs) {
			cons.add(new NeuralConnection(n, bias, weight));
		}
	}

	public void saveToFile(String path) throws FileNotFoundException {
		try (PrintStream out = new PrintStream(new FileOutputStream(path))) {
			out.println("in");
			for (Neuron n : ins) {
				out.println(n.X() + " " + n.Y() + " " + n.Id());
			}
			for (ArrayList<Neuron> arr : hiddens) {
				out.println("hidden");
				for (Neuron n : arr) {
					out.println(n.X() + " " + n.Y() + " " + n.Id());
				}
			}
			out.println("out");
			for (Neuron n : outs) {
				out.println(n.X() + " " + n.Y() + " " + n.Id());
			}
			out.println("bias");
			for (Neuron n : bias) {
				out.println(n.X() + " " + n.Y() + " " + n.Id());
			}
			for (Layer l : network.LayersForward()) {
				out.println("layer");
				for (NeuralConnection nc : l.Connections()) {
					out.println(nc.Forward().Id() + " " + nc.Backward().Id()
							+ " " + nc.Weight());
				}
			}
		}
	}

	public void loadFromFile(String path) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(path));
		ArrayList<Layer> netLayers = new ArrayList<Layer>();
		try {
			String line = br.readLine();
			int state = 0;// 0 nothing, 1 ins, 2 hidden, 3 outs, 4 layers, 5
							// bias

			ArrayList<Neuron> hiddenLayer = null;
			ArrayList<NeuralConnection> cons = null;

			while (line != null) {
				switch (line) {
				case "in":
					state = 1;
					break;
				case "hidden":
					state = 2;
					hiddenLayer = new ArrayList<Neuron>();
					hiddens.add(hiddenLayer);
					break;
				case "out":
					state = 3;
					break;
				case "layer":
					state = 4;
					cons = new ArrayList<NeuralConnection>();
					netLayers.add(new Layer(cons));
					break;
				case "bias":
					state = 5;
					break;
				default:// if not title then values
					String[] data = line.split("\\s++");
					switch (state) {
					case 0:
						break;
					case 1:
						ins.add(new Neuron(data[0], data[1], data[2]));
						break;
					case 2:
						hiddenLayer.add(new Neuron(data[0], data[1], data[2]));
						break;
					case 3:
						outs.add(new Neuron(data[0], data[1], data[2]));
						break;
					case 4:
						// loads all nodes first then does connections
						Neuron n1 = findNeuron(data[0]);
						Neuron n2 = findNeuron(data[1]);
						NeuralConnection nc = new NeuralConnection(n1, n2,
								Float.parseFloat(data[2]));
						cons.add(nc);
						n2.addConnection(nc);
						break;
					case 5:
						Neuron b = new Neuron(data[0], data[1], data[2]);
						b.setOutput(1f);
						b.setBias(true);
						bias.add(b);
						break;
					}
					break;
				}
				line = br.readLine();
			}
		} finally {
			br.close();
		}
		network = new Network(netLayers);
		view = new NetworkView(network);
	}

	protected void resetNodes(Network n) {// and connections
		for (Layer l : n.LayersForward()) {
			for (NeuralConnection nc : l.Connections()) {
				nc.update();
				if (nc.Forward().canReset()) {
					nc.Forward().reset();
				}
			}
		}
	}

	public float singleRun(float lRate) {
		forwardPass(network);
		float error = getTotalError(outs);
		updateNodes();
		return error;
	}

	private Layer getLayer(ArrayList<Neuron> a, ArrayList<Neuron> b) {
		ArrayList<NeuralConnection> cons = getConnections(a, b);
		addBias(cons, b, 1, a.get(0).X(), a.get(a.size() - 1).Y() + 5, 0.35f);
		return new Layer(cons);

	}

	private Neuron findNeuron(String id) {
		return findNeuron(Integer.parseInt(id));
	}

	private Neuron findNeuron(int id) {
		// System.out.println(id);
		// look in all arrays for neuron
		for (Neuron n : ins) {
			if (n.Id() == id)
				return n;
		}
		for (ArrayList<Neuron> arr : hiddens) {
			for (Neuron n : arr) {
				if (n.Id() == id)
					return n;
			}
		}
		for (Neuron n : outs) {
			if (n.Id() == id)
				return n;
		}
		for (Neuron n : bias) {
			if (n.Id() == id)
				return n;
		}
		throw new NullPointerException();// something went wrong if this happens
	}

	public void startView() {
		// only create one frame
		if (debug && frame == null) {
			frame = new JFrame();
			frame.add(view);
			frame.pack();
			frame.setVisible(true);
			// frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		}
	}

	// backprop stuff
	public void start(float tError, float learning)// very small, 0.5f
	{
		startView();

		startLoop(tError, learning);
		forwardPass(network);
//		for (Neuron n : outs) {
//			 System.out.println("Out: " + n.Output());
//		}
		render();
	}

	protected void render() {
		if (debug) {
			view.render(true);
			frame.repaint();
		}
	}

	protected float getTotalError(ArrayList<Neuron> neurons) {
		float totalError = 0;
		for (int i = 0; i < neurons.size(); i++) {

			// System.out.println(neurons.get(i).Target() -
			// neurons.get(i).Output());

			totalError += 0.5 * Math.pow(
					neurons.get(i).Target() - neurons.get(i).Output(), 2);
		}
		// System.out.println("Error " + totalError);
		return totalError;
	}

	// get output
	protected void forwardPass(Network n) {
		for (Layer l : n.LayersForward()) {
			for (NeuralConnection nC : l.Connections()) {
				float out = nC.Backward().Output();
				nC.Forward().addInput(out * nC.Weight());
				// System.out.println("Net " + nC.Forward() + " " +
				// nC.Forward().Net());
				// System.out.println("Out "+nC.Forward()+
				// " "+nC.Forward().Output());
			}
		}
	}

	protected void startLoop(float tErr, float lRate) {
		// trains ANN for one data entry- override to add more
		float error = 0;
		do {
			error = singleRun(lRate);
		} while (error > tErr && tErr != -1);

		System.out.println(error);
	}

	public void updateNodes() {// only nodes
		for (Layer l : network.LayersForward()) {
			for (NeuralConnection nc : l.Connections()) {
				if (nc.Forward().canReset()) {
					nc.Forward().reset();
				}
			}
		}
	}

	// outputs net without backprop
	public Float[] outputNetwork()// first is error, rest are outputs
	{
		Float[] output = new Float[1 + outs.size()];
		int i = 0;

		startView();
		forwardPass(network);
		float error = getTotalError(outs);
		output[i] = error;
		i++;
		// resetNodes(network);
		// System.out.println("E: "+error);
		for (Neuron n : outs) {
			// System.out.println("Out: "+n.Output());
			output[i] = n.Output();
			i++;
		}
		render();
		updateNodes();
		return output;
	}
	
	public int countIns()
	{
		return ins.size();
	}
	public int countOuts()
	{
		return outs.size();
	}

}
