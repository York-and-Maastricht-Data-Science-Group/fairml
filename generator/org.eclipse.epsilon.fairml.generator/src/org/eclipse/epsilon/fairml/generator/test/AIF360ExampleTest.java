package org.eclipse.epsilon.fairml.generator.test;

import java.util.ArrayList;
import java.util.List;

import org.junit.AfterClass;
import org.junit.Test;

public class AIF360ExampleTest {

	public static List<String> results = new ArrayList<>();

	String decimalFormat = "%.3f";
//	int startMeasure = 5;
//	int endMeasure = 14;
	int startMeasure = 1;
	int endMeasure = 1;

	public static String breakString(String text) {
		text = text.replace("test", "");
		int i = 0;
		StringBuilder sb = new StringBuilder();
		while (i < text.length()) {
			Character c = text.charAt(i);
			if (Character.isUpperCase(c) && i > 0) {
				sb.append(" ");
			}
			sb.append(c);
			i++;
		}

		return sb.toString();
	}

	@AfterClass
	public static void showAllResults() {
		System.out.println();
		for (String result : results) {
			System.out.println(result);
		}
	}

	@Test
	public void testTutorialCreditScoring() {
		// https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_credit_scoring.ipynb
		String filename = "tutorial_credit_scoring";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	@Test
	public void testDemoMetaClassifier() {
		// https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
		String filename = "demo_meta_classifier";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	@Test
	public void testDemoOptimPreprocAdult() {
		String filename = "demo_optim_preproc_adult";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));

	}

	@Test
	public void testTutorialMedicalExpenditure() {
		String filename = "tutorial_medical_expenditure";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));

	}

	@Test
	public void testDemoDisparateImpactRemover() {
		String filename = "demo_disparate_impact_remover";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));

	}

	@Test
	public void testDemoShortGerryfairTest() {
		String filename = "demo_short_gerryfair_test";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	@Test
	public void testDemoExponentiatedGradientReduction() {
		String filename = "demo_exponentiated_gradient_reduction";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	@Test
	public void testDemoReweighingPreproc() {
		String filename = "demo_reweighing_preproc";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	@Test
	public void testDemoRejectOptionClassification() {
		String filename = "demo_reject_option_classification";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	@Test
	public void testDemoAdversarialDebiasing() {
		String filename = "demo_adversarial_debiasing";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	@Test
	public void testDemoLFR() {
		String filename = "demo_lfr";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	@Test
	public void testDemoCalibratedEqoddsPostprocessing() {
		String filename = "demo_calibrated_eqodds_postprocessing";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

//	@Test
//	public void testExplainer() {
//		String filename = "explainer";
//		double total = measureGenerationTime(filename);
//		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
//		}).getClass().getEnclosingMethod().getName()), total));
//
//	}

	@Test
	public void testPaperDemo() {
		String filename = "paper_demo";
		double total = measureGenerationTime(filename);
		results.add(String.format("%s\t" + decimalFormat, breakString((new Object() {
		}).getClass().getEnclosingMethod().getName()), total));
	}

	/***
	 * Measure Generation Time
	 * 
	 * @param filename
	 * @return
	 */
	private double measureGenerationTime(String filename) {
		String modelFile = "test-model/" + filename + ".flexmi";

		double total = 0;
		for (int i = 1; i <= endMeasure; i++) {
			long start = System.currentTimeMillis();

			org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });

			long end = System.currentTimeMillis();
			System.out.println(end - start);
			if (i >= startMeasure)
				total = total + ((end - start) / 1000.0);

			try {
				Thread.sleep(500);
			} catch (Exception e) {
			}
		}
		total = total / (endMeasure - startMeasure + 1);
		return total;
	}
}
