package org.eclipse.epsilon.fairml.generator.test;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.awt.*;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;


/**
 * Click on the terminal first before running this test.
 */
public class GeneratorTest {

	public static Robot robot;
	public static Keyboard keyboard;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		robot = new Robot();
		keyboard = new Keyboard(robot);
		System.out.println("Please click on the terminal first!");
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	/**
	 * WARNING: Click on the terminal first before running this test.
	 */
	@Test
	public void testGeneration() {
		try {
			String filename = "generation";
			String generatedFile = "test-generated/" + filename + ".flexmi";
			String baselineFile = "test-baseline/" + filename + ".flexmi";

			java.util.List<String> lines = new ArrayList<String>();
			lines.add(filename);
			for (int i = 1; i <= 30; i++) {
				lines.add("\n");
			}
			KeyIn k = new KeyIn(lines);
			k.start();
			org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { "-w", generatedFile });

			String baseline = Files.readString(Paths.get(baselineFile));
			String generated = Files.readString(Paths.get(generatedFile));
			assertEquals(baseline, generated);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public class KeyIn extends Thread {

		java.util.List<String> lines;

		boolean stop = false;

		public KeyIn(java.util.List<String> lines) {
			this.lines = lines;
		}

		public void terminate() {
			stop = true;
		}

		@Override
		public void run() {
			try {
				for (String line : lines) {
					if (stop)
						return;
					Thread.sleep(500);
					keyboard.type(line);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
	};

}
