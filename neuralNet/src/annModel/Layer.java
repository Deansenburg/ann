package annModel;
import java.util.ArrayList;


public class Layer {

	private ArrayList<NeuralConnection> mConnections;
	
	//is a list of connections
	public Layer(ArrayList<NeuralConnection> cons)
	{
		mConnections = cons;
	}
	
	public int ConnectionCount()
	{
		return mConnections.size();
	}
	
	public ArrayList<NeuralConnection> Connections()
	{
		return mConnections;
	}
}
