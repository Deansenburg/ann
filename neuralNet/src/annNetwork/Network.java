package annNetwork;
import java.util.ArrayList;
import java.util.Collections;

import annModel.Layer;

public class Network {

	private ArrayList<Layer> mNetwork;
	private ArrayList<Layer> mNetRev;

	//is a list of layers
	//holds lists for forward/reverse making traversing easier
	public Network(ArrayList<Layer> net) {
		mNetwork = net;
		ArrayList<Layer> reversed = new ArrayList<Layer>(mNetwork);
		Collections.reverse(reversed);
		mNetRev = reversed;
	}

	public int LayerCount() {
		return mNetwork.size();
	}

	public ArrayList<Layer> LayersForward() {
		return mNetwork;
	}

	public ArrayList<Layer> LayersBackward() {
		return mNetRev;
	}

}
