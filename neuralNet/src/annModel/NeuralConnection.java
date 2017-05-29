package annModel;

public class NeuralConnection {
	private float mWeight;
	private Neuron mForward;
	private Neuron mBackward;

	// for synchronising weight updates
	private float mDelayedWeight;

	public NeuralConnection(Neuron f, Neuron b, float weight) {
		mForward = f;
		mBackward = b;
		mWeight = weight;
	}

	public float Weight() {
		return mWeight;
	}
	boolean n = false;
	public void setWeight(float w) {
		mDelayedWeight = w;
		n = true;
	}

	public Neuron Forward() {
		return mForward;
	}

	public Neuron Backward() {
		return mBackward;
	}

	public void update() {
		if (n) {
			mWeight = mDelayedWeight;
		}
	}
}
