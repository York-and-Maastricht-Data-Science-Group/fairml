package org.eclipse.epsilon.fairml.generator;

import java.io.BufferedReader;
import java.io.File;
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

	private static final String DIR_GENERATOR = "generator";

	public static void main(String... args) {
		int exitCode = new CommandLine(new FairML()).execute(args);
		System.exit(exitCode);
	}

	@Parameters(index = "0", description = "A FairML Model in a Flexmi file (*.flexmi).")
	private File flexmiFile;

	@Option(names = { "-w", "--wizard" }, description = "[true, false(default)] Wizard to create a flexmi file.")
	private boolean isWizard = false;

	private Scanner scanner;

	/***
	 * The business logic is here.
	 */
	@Override
	public Integer call() throws Exception {
		try {
			if (isWizard) {
				scanner = new Scanner(System.in);
				flexmiFile = generateFlexmiFile(flexmiFile);
				scanner.close();
			}

			// Load FairML Package
			FairmlPackage.eINSTANCE.eClass();

			// the path of the flexmi file's directory
			flexmiFile = new File(flexmiFile.getAbsolutePath().trim());
			String path = flexmiFile.getParentFile().getPath().trim();

			// initialise the xmiFile from the flexmi file
			File xmiFile = new File(flexmiFile.getAbsolutePath() + ".xmi");

			// load the flexmi file
			ResourceSet flexmiResourceSet = new ResourceSetImpl();
			flexmiResourceSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put("*",
					new FlexmiResourceFactory());
			Resource flexmiResource = flexmiResourceSet.createResource(URI.createFileURI(flexmiFile.getAbsolutePath()));
			flexmiResource.load(null);

			// The EClasses of all model elements
			final Set<EClass> eClasses = new HashSet<>();

			ResourceSet xmiResourceSet = new ResourceSetImpl();
			xmiResourceSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put("*",
					new XMIResourceFactoryImpl() {
						@Override
						public Resource createResource(URI uri) {
							return new XMIResourceImpl(uri) {
								@Override
								protected boolean useUUIDs() {
									for (EClass eClass : eClasses) {
										for (EAttribute eAttribute : eClass.getEAttributes()) {
											if (eAttribute.isID())
												return false;
										}
									}
									return true;
								}
							};
						}
					});

			// Collect all EClasses of all model elements
			// so that we can use them above to decide if the XMI
			// resource will have XMI IDs or not
			for (Iterator<EObject> it = flexmiResource.getAllContents(); it.hasNext(); eClasses.add(it.next().eClass()))
				;

			URI resourceURI = URI.createFileURI(xmiFile.getAbsolutePath());
			Resource xmiResource = xmiResourceSet.createResource(resourceURI);
			xmiResource.getContents().addAll(EcoreUtil.copyAll(flexmiResource.getContents()));
			xmiResource.save(null);

			// extracting generator files from the jar to the generator folder
			File dir = new File("." + File.separator + DIR_GENERATOR);
			dir = new File(dir.getAbsolutePath());
			if (!dir.exists()) {
				System.out.println("Create generator directory ..");
				Files.createDirectories(Paths.get(dir.getAbsolutePath()));
			}

			String[] fileNames = { "fairml.egx", "fairml.py", "ipynb.egl", //
					"py.egl", "Util.eol" };

			for (String filename : fileNames) {
				File targetFile = new File(DIR_GENERATOR + File.separator + filename);
				targetFile = new File(targetFile.getAbsolutePath());
				InputStream is = getClass().getResourceAsStream("/" + DIR_GENERATOR + "/" + filename);
//				System.out.println("Create file " + targetFile.getAbsolutePath());
				Files.copy(is, targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
//				System.out.println("Success");
			}

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
			InMemoryEmfModel model = new InMemoryEmfModel(xmiResource);
			model.setName("M");
			model.load();

			// Make the document visible to the EGX program
			module.getContext().getModelRepository().addModel(model);

			// Execute the transformation
			module.execute();

//			Thread.sleep(1000);

			fairml.FairML fairml = (fairml.FairML) xmiResource.getContents().get(0);
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
				command = "p2j -o \"" + path + File.separator + filename + ".py\"";
				command = new String(command.getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8);

			}
//			System.out.println(command);
			Process p = Runtime.getRuntime().exec(command);
//			BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
//			String line;
//			while ((line = reader.readLine()) != null) {
//				System.out.println(line);
//			}
//			reader.close();

			System.out.println("Finished!");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return 0;
	}

	@SuppressWarnings("serial")
	private File generateFlexmiFile(File flexmiFile) {

		/** root **/
		Map<String, Object> root = new LinkedHashMap<String, Object>();
		root.put("?nsuri", "fairml");

//		/** fairml **/
//		List<Object> fairml = new ArrayList<>();
//		fairml.add(Map.of("name", getUserInput("FairML project's name:")));		
//		fairml.add(Map.of("description", getUserInput("FairML project's description:")));
//		root.put("fairml", fairml);
//
//		/** dataset **/
//		List<Object> dataset = new ArrayList<>();
//		dataset.add(Map.of("datasetPath", getUserInput("Path to your dataset (e.g., data/census.csv):")));		
//		dataset.add(Map.of("name", getUserInput("The name of the dataset:")));
//		dataset.add(Map.of("predictedAttribute", getUserInput("Predicted attribute:")));
//		dataset.add(Map.of("protectedAttributes", getUserInput("Protected attributes [seperated by comma] (sex, race, etc.):")));
//		dataset.add(Map.of("categoricalFeatures", getUserInput("Categorical features [seperated by comma] (education, occupation, etc.):")));		
//		dataset.add(Map.of("trainTestSplit", getUserInput("Train test split [seperated by comma] (e.g., 7, 3 or 6,4):")));		
//		root.put("dataset", dataset);
		
		/** bias mitigation **/
		List<Object> biasMitigation = new ArrayList<>();
		biasMitigation.add(Map.of("name", getUserInput("Bias mitigation's name:")));		
		biasMitigation.add(Map.of("description", getUserInput("Bias mitigation's description:")));				
		root.put("biasMitigation", biasMitigation);
		
		/** training method **/
		List<Object> trainingMethod = new ArrayList<>();
		trainingMethod.add(Map.of("algorithm", getUserInput("Classifier (DecisionTree):")));						
		biasMitigation.add(Map.of("trainingMethod", trainingMethod));
		
		/** training method **/
		List<Object> mitigationMethod = new ArrayList<>();
		mitigationMethod.add(Map.of("algorithm", getUserInput("Mitigation Algorithm (Reweighing):")));						
		biasMitigation.add(Map.of("mitigationMethod", mitigationMethod));
		
		/** training method **/
		List<Object> biasMetric = new ArrayList<>();
		biasMetric.add(Map.of("algorithm", getUserInput("BiasMetric (false_negative_rate_ratio):")));						
		biasMitigation.add(Map.of("biasMetric", biasMetric));
		
		DumperOptions options = new DumperOptions();
		options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
		Yaml yaml = new Yaml(options);
		StringWriter writer = new StringWriter();
		yaml.dump(root, writer);
		System.out.println();
		System.out.println(writer.toString());

		return flexmiFile;
	}

	private String getUserInput(String question) {
		String answer = null;
		boolean valid = false;
		while (!valid) {
			System.out.print(question + " ");
			answer = scanner.nextLine().trim();
			if (answer instanceof String) {
				valid = true;
			}
		}
		return answer;
	}

}
