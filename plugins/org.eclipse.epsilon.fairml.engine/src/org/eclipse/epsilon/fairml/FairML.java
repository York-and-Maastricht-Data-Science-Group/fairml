package org.eclipse.epsilon.fairml;

import java.util.ArrayList;
import java.util.List;

public class FairML {

	private String name = null;
	private List<String> sources = new ArrayList<>();
	private List<String> protectedAttributes = new ArrayList<>();
	private List<String> predictedAttributes = new ArrayList<>();
	private List<String> algorithms = new ArrayList<>();
	private List<String> checkingMethods = new ArrayList<>();
	private List<String> mitigationMethods = new ArrayList<>();

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public List<String> getSources() {
		return sources;
	}

	public List<String> getProtectedAttributes() {
		return protectedAttributes;
	}

	public List<String> getPredictedAttributes() {
		return predictedAttributes;
	}

	public List<String> getAlgorithms() {
		return algorithms;
	}

	public List<String> getCheckingMethods() {
		return checkingMethods;
	}

	public List<String> getMitigationMethods() {
		return mitigationMethods;
	}
}
