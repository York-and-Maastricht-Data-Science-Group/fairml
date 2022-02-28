package org.eclipse.epsilon.fairml.generator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.Callable;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceImpl;
import org.eclipse.epsilon.common.module.ModuleElement;
import org.eclipse.epsilon.common.parse.problem.ParseProblem;
import org.eclipse.epsilon.egl.EglFileGeneratingTemplateFactory;
import org.eclipse.epsilon.egl.EgxModule;
import org.eclipse.epsilon.egl.dom.GenerationRule;
import org.eclipse.epsilon.emc.emf.InMemoryEmfModel;
import org.eclipse.epsilon.eol.dom.AssignmentStatement;
import org.eclipse.epsilon.eol.dom.StringLiteral;
import org.eclipse.epsilon.eol.dom.VariableDeclaration;
import org.eclipse.epsilon.erl.dom.Pre;
import org.eclipse.epsilon.flexmi.FlexmiResourceFactory;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

import fairml.FairmlPackage;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

@Command(name = "fairml", mixinStandardHelpOptions = true, //
		version = "fairml 0.1", //
		description = "Generate Bias Mitigation Implementation " + "in Python & Jupyter Notebook "
				+ "from a FairML model in a Flexmi file.")
public class FairML implements Callable<Integer> {

	private static CommandLine commandLine;
	private static final String DIR_GENERATOR = "generator";
	private static final String DIR_DATA = "data";

	public static void main(String... args) {
		commandLine = new CommandLine(new FairML());
		int exitCode = commandLine.execute(args);
		System.exit(exitCode);
	}

	@Parameters(index = "0", description = "A FairML Model in a Flexmi file (*.flexmi).")
	private File flexmiFile;

	@Option(names = { "-w", "--wizard" }, description = "Run the wizard to create a flexmi file.")
	private boolean isWizard = false;

	@Option(names = { "-j", "--jupyter" }, description = "Run Jupyter Notebook.")
	private boolean runJupyter = false;

	private Scanner scanner;

	/***
	 * The business logic is here.
	 */
	@Override
	public Integer call() throws Exception {
		try {
			// extracting generator files from the jar to the local generator dan data
			// folders
			extractFilesFromJar(new String[] { "fairml.egx", "fairml.py", "ipynb.egl", //
					"py.egl", "fairml.eol" }, DIR_GENERATOR);
			extractFilesFromJar(new String[] { "adult.data.numeric.csv", //
					"adult.data.numeric.txt" }, DIR_DATA);

			if (isWizard) {
				scanner = new Scanner(System.in);
				flexmiFile = generateFlexmiFile(flexmiFile);
				scanner.close();
			} else if (runJupyter) {
				String command = null;
				if (System.getProperty("os.name").startsWith("Windows")) {
					command = String.format(
							"cmd.exe /C \"start /B jupyter notebook %s --port=8888 --no-browser --ip='*' --allow-root --NotebookApp.password_required=False --NotebookApp.allow_remote_access=True\"",
							flexmiFile.getAbsolutePath());
				} else {
					command = "jupyter notebook " + flexmiFile.getAbsolutePath()
							+ " --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.password_required=False --NotebookApp.allow_remote_access=True";
				}
				System.out.println(command);
				Process p = Runtime.getRuntime().exec(command);

				BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
				String line;
				while ((line = reader.readLine()) != null) {
					System.out.println(line);
				}
				reader.close();
				
//				command = "jupyter notebook list";
//				System.out.println(command);
//				Process p = Runtime.getRuntime().exec(command);
//
//				BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
//				String line;
//				while ((line = reader.readLine()) != null) {
//					System.out.println(line);
//				}
//				reader.close();

				scanner = new Scanner(System.in);
				System.out.print("Type quit and enter to exit: ");
				String input = scanner.next();
				while (!"quit".equals(input)) {
					input = scanner.next();
				}
				return 0;
			}

			// Load FairML Package
			FairmlPackage.eINSTANCE.eClass();

			// the path of the flexmi file's directory
			flexmiFile = new File(flexmiFile.getAbsolutePath().trim());
			String path = flexmiFile.getParentFile().getPath().trim();

			// load the flexmi file
			ResourceSet flexmiResourceSet = new ResourceSetImpl();
			flexmiResourceSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put("*",
					new FlexmiResourceFactory());
			Resource flexmiResource = flexmiResourceSet.createResource(URI.createFileURI(flexmiFile.getAbsolutePath()));
			flexmiResource.load(null);

			// Parse *.egx, read from file first,
			// if not available then read from JAR
			EgxModule module = new EgxModule(new EglFileGeneratingTemplateFactory());
			File egxFile = new File(DIR_GENERATOR + File.separator + "fairml.egx");
			if (egxFile.exists()) {
				module.parse(egxFile);
			}

			// remove IpynbGenFile rule since it's not required in this CLI app
			GenerationRule rule = module.getGenerationRules().stream().filter(r -> r.getName().equals("IpynbGenFile"))
					.findFirst().orElse(null);
			module.getGenerationRules().remove(rule);

			// update genPath of the loaded egx file
			for (Pre pre : module.getPre()) {
				for (ModuleElement e1 : pre.getChildren()) {
					for (ModuleElement e2 : e1.getChildren()) {
						if (e2 instanceof AssignmentStatement) {
							if (e2.getChildren().get(0) instanceof VariableDeclaration) {
								String varName = ((VariableDeclaration) e2.getChildren().get(0)).getName();
								if (varName.equals("genPath"))
									((StringLiteral) e2.getChildren().get(1)).setValue(path);
							}

						}
					}
				}
			}

			// return -1 if there is a parse problem
			if (!module.getParseProblems().isEmpty()) {
				for (ParseProblem parseProblem : module.getParseProblems()) {
					System.out.println(parseProblem.toString());
				}
				return -1;
			}

			// create a in-memory model of the xmi resource
			InMemoryEmfModel model = new InMemoryEmfModel(flexmiResource);
//			InMemoryEmfModel model = new InMemoryEmfModel(xmiResource);
			model.setName("M");
			model.load();

			// Make the document visible to the EGX program
			module.getContext().getModelRepository().addModel(model);

			// Execute the transformation
			module.execute();

//			Thread.sleep(1000);

			fairml.FairML fairml = (fairml.FairML) flexmiResource.getContents().get(0);
//			fairml.FairML fairml = (fairml.FairML) xmiResource.getContents().get(0);
			String filename = fairml.getName().toLowerCase().replace(" ", "_").trim();
//			System.out.println("Python file:");
//			System.out.println(path + File.separator + filename + ".py");

			// Generate the Jupyter notebook
			System.out.println();
			String command = "";
			if (System.getProperty("os.name").startsWith("Windows")) {
				command = "p2j -o \"" + path + //
						File.separator + filename + ".py\"";
			} else {
//				command = "ls";
				command = path + File.separator + filename + ".py";
				command = "p2j -o " + command.replace(" ", "\\ ");

			}
//			System.out.println(command);
			Process p = Runtime.getRuntime().exec(command);
			BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			String line;
			while ((line = reader.readLine()) != null) {
				System.out.println(line);
			}
			reader.close();

			System.out.println("Finished!");
		} catch (

		Exception e) {
			e.printStackTrace();
		}
		return 0;
	}

	private void extractFilesFromJar(String[] fileNames, String dirInJar) throws IOException {
		File dir = new File("." + File.separator + dirInJar);
		dir = new File(dir.getAbsolutePath());
		if (!dir.exists()) {
			System.out.println("Create generator directory ..");
			Files.createDirectories(Paths.get(dir.getAbsolutePath()));
		}

		for (String filename : fileNames) {
			File targetFile = new File(dirInJar + File.separator + filename);
			targetFile = new File(targetFile.getAbsolutePath());
			InputStream is = getClass().getResourceAsStream("/" + dirInJar + "/" + filename);
			Files.copy(is, targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
		}
	}

	@SuppressWarnings("serial")
	private File generateFlexmiFile(File flexmiFile) throws IOException {

		System.out.println("=====================================");
		System.out.println("            FairML Wizard");
		System.out.println("=====================================");
		System.out.println(this.commandLine.getCommandSpec().version()[0]);
		System.out.println("");

		/** root **/
		Map<String, Object> root = new LinkedHashMap<String, Object>();
		root.put("?nsuri", "fairml");

		/** fairml **/
		List<Object> fairml = new ArrayList<>();
		System.out.println("==== FairML ====");
		String projectName = "Demo";
		String temp = getUserInput("FairML project's name (default: " + //
				projectName + "):", projectName);
		fairml.add(Map.of("name", temp));

		String projectDescription = "Predict income <=50K: 0, >50K: 1";
		temp = getUserInput("FairML project's description (default: " + projectDescription + "):", projectDescription);
		fairml.add(Map.of("description", temp));
		root.put("fairml", fairml);
//
		/** dataset **/
		System.out.println("");
		System.out.println("==== Dataset ====");
		List<Object> dataset = new ArrayList<>();

		String datasetPath = "data/adult.data.numeric.csv";
		temp = getUserInput("Path to your dataset (default: " + datasetPath + "):", datasetPath, File.class);
		dataset.add(Map.of("datasetPath", temp));

		String datasetName = "Adult Dataset";
		temp = getUserInput("The name of the dataset: (default: " + datasetName + "):", datasetName);
		dataset.add(Map.of("name", temp));

		String predictedAttribute = "income-per-year";
		temp = getUserInput("Predicted attribute (default: " + predictedAttribute + "):", predictedAttribute);
		dataset.add(Map.of("predictedAttribute", temp));

		String protectedAttributes = "sex, race";
		temp = getUserInput("Protected attributes (default: " + protectedAttributes + "):", protectedAttributes,
				String[].class);
		dataset.add(Map.of("protectedAttributes", temp));

		String categoricalFeatures = "workclass, education, marital-status, "
				+ "occupation, relationship, native-country";
		temp = getUserInput("Categorical features (default: " + categoricalFeatures + ")\n:", categoricalFeatures,
				String[].class);
		dataset.add(Map.of("categoricalFeatures", temp));

		String trainTestValidationSplit = "7, 2, 3";
		temp = getUserInput("Train test validation split (default: " + trainTestValidationSplit + "):",
				trainTestValidationSplit, Double[].class);
		dataset.add(Map.of("trainTestSplit", temp));
		fairml.add(Map.of("dataset", dataset));

		/** bias mitigation **/
		System.out.println("");
		System.out.println("==== Bias Mitigation ====");
		List<Object> biasMitigation = new ArrayList<>();
		String name = "Bias Mitigation 01";
		temp = getUserInput("Bias mitigation's name (default " + //
				name + "):", name);
		biasMitigation.add(Map.of("name", temp));

		String description = "Bias Mitigation 01 Description";
		temp = getUserInput("Bias mitigation's description (default " + //
				description + "):", description);
		biasMitigation.add(Map.of("description", temp));
		biasMitigation.add(Map.of("dataset", datasetName));
		fairml.add(Map.of("biasMitigation", biasMitigation));

		/** training method **/
		System.out.println("");
		System.out.println("---- Training Algorithm ----");
		List<Object> trainingMethod = new ArrayList<>();
		System.out.println("1. DecisionTreeClassifier");
		System.out.println("2. LogisticRegression");
		System.out.println("3. LinearSVC");
		temp = "1";
		temp = getUserInput("Classifier (default 1. DecisionTreeClassifier):", temp, "Classifier");
		trainingMethod.add(Map.of("algorithm", temp));
		if (temp.equals("DecisionTreeClassifier")) {
			trainingMethod.add(Map.of("parameters", "criterion='gini', max_depth=4"));
		}
		biasMitigation.add(Map.of("trainingMethod", trainingMethod));

		/** bias mitigation algorithm **/
		System.out.println("");
		System.out.println("---- Mitigation Algorithm ----");
//		List<Object> mitigationMethod = new ArrayList<>();

		// preprocessing
		System.out.println("");
		System.out.println("# 1. Pre-processing");
		String prepreprocessingMitigation = "true";
		if (!prepreprocessingMitigation.equals(temp)) {
			temp = getUserInput("Apply bias mitigation in preprocessing" + //
					" (default: true):", prepreprocessingMitigation, Boolean.class);
		}
		biasMitigation.add(Map.of("prepreprocessingMitigation", temp));
		if ((Boolean.parseBoolean(temp.toString()))) {
			String modifiableWeight = "true";
			temp = getUserInput("The weights of the dataset are modifiable" + //
					" (default: true):", modifiableWeight, Boolean.class);
			if (!modifiableWeight.equals(temp)) {
				biasMitigation.add(Map.of("modifiableWeight", temp));
			}
			String allowLatentSpace = "false";
			temp = getUserInput("The bias mitigation allows latent space" + //
					" (default: false):", allowLatentSpace, Boolean.class);
			if (!allowLatentSpace.equals(temp)) {
				biasMitigation.add(Map.of("allowLatentSpace", temp));
			}
		}

		// inprocessing
		System.out.println("");
		System.out.println("# 2. In-processing");
		String inpreprocessingMitigation = "false";
		temp = getUserInput("Apply bias mitigation in in-processing" + //
				" (default: false):", inpreprocessingMitigation, Boolean.class);
		if (!inpreprocessingMitigation.equals(temp)) {
			biasMitigation.add(Map.of("inpreprocessingMitigation", temp));
		}
		if ((Boolean.parseBoolean(temp.toString()))) {
			String allowRegularisation = "false";
			temp = getUserInput("Regularisation is allowed" + //
					" (default: false):", allowRegularisation, Boolean.class);
			if (!allowRegularisation.equals(temp)) {
				biasMitigation.add(Map.of("allowRegularisation", temp));
			}
		}

		// postprocessing
		System.out.println("");
		System.out.println("# 3. Post-processing");
		String postpreprocessingMitigation = "false";
		temp = getUserInput("Apply bias mitigation in post-processing" + //
				" (default: false):", postpreprocessingMitigation, Boolean.class);
		if (!postpreprocessingMitigation.equals(temp)) {
			biasMitigation.add(Map.of("postpreprocessingMitigation", temp));
		}
		if ((Boolean.parseBoolean(temp.toString()))) {
			String allowRandomisation = "false";
			temp = getUserInput("Randomisation is allowed" + //
					" (default: false):", allowRandomisation, Boolean.class);
			if (!allowRandomisation.equals(temp)) {
				biasMitigation.add(Map.of("allowRandomisation", temp));
			}
		}

//		mitigationMethod.add(Map.of("algorithm", getUserInput("Mitigation Algorithm (Reweighing):")));
//		biasMitigation.add(Map.of("mitigationMethod", mitigationMethod));

		/** bias metric **/
		System.out.println("");
		System.out.println("---- Bias Metric ----");
//		List<Object> biasMetric = new ArrayList<>();

		String groupFairness = "true";
		temp = getUserInput("Measure group fairness" + //
				" (default: " + groupFairness + "):", groupFairness, Boolean.class);
		if (!groupFairness.equals(temp)) {
			biasMitigation.add(Map.of("groupFairness", temp));
		}

		String individualFairness = "false";
		temp = getUserInput("Measure individual fairness" + //
				" (default: " + individualFairness + "):", individualFairness, Boolean.class);
		if (!individualFairness.equals(temp)) {
			biasMitigation.add(Map.of("individualFairness", temp));
		}

		String groupIndividualSingleMetric = "false";
		temp = getUserInput("Use single metrics for both individuals and groups" + //
				" (default: " + groupIndividualSingleMetric + "):", groupIndividualSingleMetric, Boolean.class);
		if (!groupIndividualSingleMetric.equals(temp)) {
			biasMitigation.add(Map.of("groupIndividualSingleMetric", temp));
		}

		if (groupFairness.equals("true")) {
			String equalFairness = "false";
			temp = getUserInput("Measure equal fairness" + //
					" (default: " + equalFairness + "):", equalFairness, Boolean.class);
			if (!equalFairness.equals(temp)) {
				biasMitigation.add(Map.of("equalFairness", temp));
			}

			String proportionalFairness = "false";
			temp = getUserInput("Measure proportional fairness" + //
					" (default: " + proportionalFairness + "):", proportionalFairness, Boolean.class);
			if (!proportionalFairness.equals(temp)) {
				biasMitigation.add(Map.of("proportionalFairness", temp));
			}

			String checkFalsePositive = "false";
			temp = getUserInput("Measure false positives" + //
					" (default: " + checkFalsePositive + "):", checkFalsePositive, Boolean.class);
			if (!checkFalsePositive.equals(temp)) {
				biasMitigation.add(Map.of("checkFalsePositive", temp));
			}

			String checkFalseNegative = "false";
			temp = getUserInput("Measure false negatives" + //
					" (default: " + checkFalseNegative + "):", checkFalseNegative, Boolean.class);
			if (!checkFalseNegative.equals(temp)) {
				biasMitigation.add(Map.of("checkFalseNegative", temp));
			}

			String checkErrorRate = "false";
			temp = getUserInput("Measure error rates" + //
					" (default: " + checkErrorRate + "):", checkErrorRate, Boolean.class);
			if (!checkErrorRate.equals(temp)) {
				biasMitigation.add(Map.of("checkErrorRate", temp));
			}

			String checkEqualBenefit = "false";
			temp = getUserInput("Measure equal benefit" + //
					" (default: " + checkEqualBenefit + "):", checkEqualBenefit, Boolean.class);
			if (!checkEqualBenefit.equals(temp)) {
				biasMitigation.add(Map.of("checkEqualBenefit", temp));
			}
		}

		DumperOptions options = new DumperOptions();
		options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
		Yaml yaml = new Yaml(options);
		StringWriter writer = new StringWriter();
		yaml.dump(root, writer);
		System.out.println();
		System.out.println("==== Flexmi Output ====");
		System.out.println(writer.toString());

		System.out.println("------Generating-------");
		FileWriter fw = new FileWriter(flexmiFile.getAbsoluteFile());
		fw.write(writer.toString());
		fw.close();

		return flexmiFile;
	}

	private String getUserInput(String question, String defaultValue) {
		return this.getUserInput(question, defaultValue, String.class);
	}

	private String getUserInput(String question, String defaultValue, Object expectedType) {
		String answer = null;
		boolean valid = false;
		while (!valid) {
			System.out.print(question + " ");
			answer = scanner.nextLine().trim();
			if ((answer == null || answer.trim().equals("")) && defaultValue != null && defaultValue.length() > 0) {
				answer = defaultValue;
			}

			if ("Classifier".equals(expectedType)) {
				try {
					int opt = Integer.parseInt(answer);
					if (opt > 0 && opt < 4) {
						switch (answer) {
						case "2":
							answer = "LogisticRegression";
							break;
						case "3":
							answer = "LinearSVC";
							break;
						default:
							answer = "DecisionTreeClassifier";
						}
						valid = true;
					} else {
						System.out.println("Error: the option is not available");
					}
				} catch (Exception e) {
					System.out.println("Error: answer should be in integer");
				}
			} else if (File.class == expectedType) {
				File file = new File(answer);
				if (file.exists()) {
					valid = true;
				} else {
					System.out.println("Error: file does not exist");
				}
			} else if (Boolean.class == expectedType) {
				if ("true".equals(answer.toLowerCase()) || "false".equals(answer.toLowerCase())) {
					valid = true;
				} else {
					System.out.println("Error: answer should be true or false");
				}
			} else if (String[].class == expectedType) {
				String[] obj = answer.split(",");
				if (obj.getClass().isArray()) {
					valid = true;
				} else {
					System.out.println("Error: answer should in Strings separated by commas");
				}
			} else if (Double[].class == expectedType) {
				String[] obj = answer.split(",");
				if (obj.getClass().isArray()) {
					if (obj.length > 0) {
						try {
							Double.parseDouble(obj[0]);
							valid = true;
						} catch (Exception e) {
							System.out.println("Error: answer should be in numbers separated by commas");
						}
					}
				} else {
					System.out.println("Error: answer should be in numbers separated by commas");
				}
			} else {
				valid = true;
			}
		}

		return answer;
	}

}
