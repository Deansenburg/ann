package annModel;
import java.util.ArrayList;

public class Neuron {
	private float mNetValue = 0f;

	private static int curId = 0;
	
	//for identifying between neurons
	private int mId;
	//for visualisation
	private int mX, mY;

	//what the neuron should output - helpful for back prop on multiple H layers
	private float mTargetOut;

	private float mOutput;
	//should the output be calculated or just output
	//for difference between input/bias/other neurons
	private boolean mOut = false;
	
	//what is this neuron connected to going forward - back prop
	private ArrayList<NeuralConnection> mForwardConnections;
	
	//easier reading from file
	public Neuron(String x, String y, String id) {
		mX = Integer.parseInt(x);
		mY = Integer.parseInt(y);
		mForwardConnections = new ArrayList<NeuralConnection>();
		mId = Integer.parseInt(id);
	}
	
	public Neuron(int x, int y, int id) {
		mX = x;
		mY = y;
		mForwardConnections = new ArrayList<NeuralConnection>();
		mId = id;
	}
	
	public Neuron(int x, int y) {
		mX = x;
		mY = y;
		mForwardConnections = new ArrayList<NeuralConnection>();
		mId = curId;
		System.out.println(mId);
		curId++;
	}

	public int Id()
	{
		return mId;
	}
	
	public void addConnection(NeuralConnection nc) {
		mForwardConnections.add(nc);
	}

	public ArrayList<NeuralConnection> ForwardConnections() {
		return mForwardConnections;
	}

	public void setTarget(float out) {
		mTargetOut = out;
	}

	public float Target() {
		return mTargetOut;
	}

	public void addInput(float in) {
		mNetValue += in;
	}

	public float Net() {
		return mNetValue;
	}
	public void setOutput(float f)
	{
		mOutput = f;
		mOut = true;
	}

	public float Output() {
		if (mOut)
		{
			return mOutput;
		}
		return (float)(1d/(Math.exp(-mNetValue)+1));
	}

	public void reset() {
		mNetValue = 0;
	}

	public int X() {
		return mX;
	}

	public int Y() {
		return mY;
	}
	@Override
	public String toString() {
		return mX + " : "+mY;
	}
	
	//distinguish between regular neurons and input
	public boolean canReset()
	{
		return !mOut;
	}
	
	private boolean isBias = false;
	public boolean isBias()
	{
		return isBias;
	}
	public void setBias(boolean b)
	{
		isBias = b;
	}
}
