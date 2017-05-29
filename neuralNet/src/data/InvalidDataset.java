package data;
public class InvalidDataset extends Exception
	{
		private static final long serialVersionUID = 1L;

		public InvalidDataset(String msg) {
			super(msg);
		}
	}