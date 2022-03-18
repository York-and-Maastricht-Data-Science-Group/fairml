package org.eclipse.epsilon.fairml.generator.test;

import static org.junit.Assert.*;

import org.junit.Test;

public class AIF360ExampleTest {

	@Test
	public void testTutorialCreditScoring() {
		// https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_credit_scoring.ipynb
		String filename = "tutorial_credit_scoring";
		String modelFile = "test-model/" + filename + ".flexmi";

		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
	@Test
	public void testDemoMetaClassifier() {
		// https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
		String filename = "demo_meta_classifier";
		String modelFile = "test-model/" + filename + ".flexmi";

		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
	@Test
	public void testExplainer() {
		String filename = "explainer";
		String modelFile = "test-model/" + filename + ".flexmi";

		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
	@Test
	public void testDemoOptimisedPreprocessingAdult() {
		String filename = "demo_optimized_preprocessing_adult";
		String modelFile = "test-model/" + filename + ".flexmi";

		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
	@Test
	public void testTutorialMedicalExpenditure() {
		String filename = "tutorial_medical_expenditure";
		String modelFile = "test-model/" + filename + ".flexmi";

		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
	@Test
	public void testDemoDisparateImpactRemover() {
		String filename = "demo_disparate_impact_remover";
		String modelFile = "test-model/" + filename + ".flexmi";

		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
	@Test
	public void testDemoShortGerryfairTest() {
		String filename = "demo_short_gerryfair_test";
		String modelFile = "test-model/" + filename + ".flexmi";
		
		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
	@Test
	public void testDemoExponentiatedGradientReduction() {
		String filename = "demo_exponentiated_gradient_reduction";
		String modelFile = "test-model/" + filename + ".flexmi";
		
		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
	@Test
	public void testDemoReweighingPreproc() {
		String filename = "demo_reweighing_preproc";
		String modelFile = "test-model/" + filename + ".flexmi";
		
		org.eclipse.epsilon.fairml.generator.FairML.main(new String[] { modelFile });
		
		assertEquals(0, 0);
	}
	
}
