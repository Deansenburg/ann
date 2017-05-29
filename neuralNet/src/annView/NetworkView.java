package annView;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.math.RoundingMode;
import java.text.DecimalFormat;

import javax.swing.JComponent;

import annModel.Layer;
import annModel.NeuralConnection;
import annModel.Neuron;
import annNetwork.Network;

//debug tool

@SuppressWarnings("serial")
public class NetworkView extends JComponent {

	private Network mNet;

	private Boolean mRender = false;
	
	public NetworkView(Network n) {
		mNet = n;
	}

	public void render(Boolean b)
	{
		mRender = b;
	}
	
	@Override
	protected void paintComponent(Graphics gx) {
		super.paintComponent(gx);
		if (!mRender) return;
		gx.setColor(Color.white);
		gx.fillRect(0, 0, getWidth(), getHeight());

		int scale = 30;
		int offset = 20;
		FontMetrics metrics = gx.getFontMetrics();
		DecimalFormat df = new DecimalFormat("#.##");
		df.setRoundingMode(RoundingMode.HALF_EVEN);
		
		for (Layer l:mNet.LayersForward()) {

			for (NeuralConnection nC:l.Connections()) {
				
				gx.setColor(Color.black);

				Neuron n1 = nC.Forward();
				Neuron n2 = nC.Backward();

				int x1 = offset + n1.X() * scale;
				int x2 = offset + n2.X() * scale;
				int y1 = offset + n1.Y() * scale;
				int y2 = offset + n2.Y() * scale;

				gx.drawLine(x1, y1, x2, y2);

				int cX = (int) ((x1 * 0.75) + (x2 * 0.25));
				int cY = (int) ((y1 * 0.75) + (y2 * 0.25));

				String s = df.format(nC.Weight());
				int height = metrics.getHeight();
				int width = metrics.stringWidth(s);

				int pad = 5;
				gx.setColor(Color.white);
				gx.fillRect(cX - (width / 2) - pad, cY - (height / 2) - pad,
						width + (pad * 2), height + (pad * 2));
				gx.setColor(Color.black);
				gx.drawRect(cX - (width / 2) - pad, cY - (height / 2) - pad,
						width + (pad * 2), height + (pad * 2));

				gx.drawString(s, cX - (width / 2), cY + (height / 2));
				// gx.fillOval((int)((x1*0.75)+(x2*0.25)),
				// (int)((y1*0.75)+(y2*0.25)), 5, 5);

			}
		}
		int diam = 40;
		for (Layer l:mNet.LayersForward()) {
			
			for (NeuralConnection nC:l.Connections()) {
				gx.setColor(Color.black);

				Neuron n1 = nC.Forward();
				Neuron n2 = nC.Backward();

				int x1 = offset + n1.X() * scale - diam / 2;
				int x2 = offset + n2.X() * scale - diam / 2;

				int y1 = offset + n1.Y() * scale - diam / 2;
				int y2 = offset + n2.Y() * scale - diam / 2;

				gx.setColor(Color.white);
				gx.fillOval(x1, y1, diam, diam);
				gx.fillOval(x2, y2, diam, diam);
				gx.setColor(Color.black);
				gx.drawOval(x1, y1, diam, diam);
				gx.drawOval(x2, y2, diam, diam);

				int height = metrics.getHeight() / 4;
				
				
				String s = df.format(n1.Net());
			
				
				gx.drawLine(x1 + (diam / 2), y1, x1 + (diam / 2), y1 + diam);
				gx.drawString(s, x1 + (diam / 4) - metrics.stringWidth(s) / 2,
						y1 + (diam / 2) + height);
				s = df.format(n1.Output()) + "";
				gx.drawString(s, x1 + (diam / 4 * 3) - metrics.stringWidth(s)
						/ 2, y1 + (diam / 2) + height);

				gx.drawLine(x2 + (diam / 2), y2, x2 + (diam / 2), y2 + diam);
				s = df.format(n2.Net()) + "";

				gx.drawLine(x2 + (diam / 2), y2, x2 + (diam / 2), y2 + diam);
				gx.drawString(s, x2 + (diam / 4) - metrics.stringWidth(s) / 2,
						y2 + (diam / 2) + height);
				s = df.format(n2.Output()) + "";
				gx.drawString(s, x2 + (diam / 4 * 3) - metrics.stringWidth(s)
						/ 2, y2 + (diam / 2) + height);
			}
		}
	}

	public Dimension getPreferredSize() {
		return new Dimension(650, 200);
	}

	public Dimension getMinimumSize() {
		return getPreferredSize();
	}
}
