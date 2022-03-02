/**
 */
package fairml;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It contains accessors for the meta objects to represent
 * <ul>
 *   <li>each class,</li>
 *   <li>each feature of each class,</li>
 *   <li>each enum,</li>
 *   <li>and each data type</li>
 * </ul>
 * <!-- end-user-doc -->
 * @see fairml.FairmlFactory
 * @model kind="package"
 * @generated
 */
public interface FairmlPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "fairml";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "fairml";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	FairmlPackage eINSTANCE = fairml.impl.FairmlPackageImpl.init();

	/**
	 * The meta object id for the '{@link fairml.impl.FairMLImpl <em>Fair ML</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see fairml.impl.FairMLImpl
	 * @see fairml.impl.FairmlPackageImpl#getFairML()
	 * @generated
	 */
	int FAIR_ML = 0;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAIR_ML__NAME = 0;

	/**
	 * The feature id for the '<em><b>Description</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAIR_ML__DESCRIPTION = 1;

	/**
	 * The feature id for the '<em><b>Filename</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAIR_ML__FILENAME = 2;

	/**
	 * The feature id for the '<em><b>Datasets</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAIR_ML__DATASETS = 3;

	/**
	 * The feature id for the '<em><b>Bias Mitigations</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAIR_ML__BIAS_MITIGATIONS = 4;

	/**
	 * The number of structural features of the '<em>Fair ML</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAIR_ML_FEATURE_COUNT = 5;

	/**
	 * The meta object id for the '{@link fairml.impl.OperationImpl <em>Operation</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see fairml.impl.OperationImpl
	 * @see fairml.impl.FairmlPackageImpl#getOperation()
	 * @generated
	 */
	int OPERATION = 1;

	/**
	 * The feature id for the '<em><b>Package Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OPERATION__PACKAGE_NAME = 0;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OPERATION__NAME = 1;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OPERATION__PARAMETERS = 2;

	/**
	 * The feature id for the '<em><b>Functions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OPERATION__FUNCTIONS = 3;

	/**
	 * The number of structural features of the '<em>Operation</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OPERATION_FEATURE_COUNT = 4;

	/**
	 * The meta object id for the '{@link fairml.impl.FunctionImpl <em>Function</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see fairml.impl.FunctionImpl
	 * @see fairml.impl.FairmlPackageImpl#getFunction()
	 * @generated
	 */
	int FUNCTION = 2;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION__NAME = 0;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION__PARAMETERS = 1;

	/**
	 * The number of structural features of the '<em>Function</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link fairml.impl.TrainingMethodImpl <em>Training Method</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see fairml.impl.TrainingMethodImpl
	 * @see fairml.impl.FairmlPackageImpl#getTrainingMethod()
	 * @generated
	 */
	int TRAINING_METHOD = 3;

	/**
	 * The feature id for the '<em><b>Package Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRAINING_METHOD__PACKAGE_NAME = OPERATION__PACKAGE_NAME;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRAINING_METHOD__NAME = OPERATION__NAME;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRAINING_METHOD__PARAMETERS = OPERATION__PARAMETERS;

	/**
	 * The feature id for the '<em><b>Functions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRAINING_METHOD__FUNCTIONS = OPERATION__FUNCTIONS;

	/**
	 * The feature id for the '<em><b>Algorithm</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRAINING_METHOD__ALGORITHM = OPERATION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Training Method</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRAINING_METHOD_FEATURE_COUNT = OPERATION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link fairml.impl.MitigationMethodImpl <em>Mitigation Method</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see fairml.impl.MitigationMethodImpl
	 * @see fairml.impl.FairmlPackageImpl#getMitigationMethod()
	 * @generated
	 */
	int MITIGATION_METHOD = 4;

	/**
	 * The feature id for the '<em><b>Package Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MITIGATION_METHOD__PACKAGE_NAME = OPERATION__PACKAGE_NAME;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MITIGATION_METHOD__NAME = OPERATION__NAME;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MITIGATION_METHOD__PARAMETERS = OPERATION__PARAMETERS;

	/**
	 * The feature id for the '<em><b>Functions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MITIGATION_METHOD__FUNCTIONS = OPERATION__FUNCTIONS;

	/**
	 * The feature id for the '<em><b>Algorithm</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MITIGATION_METHOD__ALGORITHM = OPERATION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Mitigation Method</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MITIGATION_METHOD_FEATURE_COUNT = OPERATION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link fairml.impl.BiasMetricImpl <em>Bias Metric</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see fairml.impl.BiasMetricImpl
	 * @see fairml.impl.FairmlPackageImpl#getBiasMetric()
	 * @generated
	 */
	int BIAS_METRIC = 5;

	/**
	 * The feature id for the '<em><b>Package Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_METRIC__PACKAGE_NAME = OPERATION__PACKAGE_NAME;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_METRIC__NAME = OPERATION__NAME;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_METRIC__PARAMETERS = OPERATION__PARAMETERS;

	/**
	 * The feature id for the '<em><b>Functions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_METRIC__FUNCTIONS = OPERATION__FUNCTIONS;

	/**
	 * The feature id for the '<em><b>Class Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_METRIC__CLASS_NAME = OPERATION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Bias Metric</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_METRIC_FEATURE_COUNT = OPERATION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link fairml.impl.DatasetImpl <em>Dataset</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see fairml.impl.DatasetImpl
	 * @see fairml.impl.FairmlPackageImpl#getDataset()
	 * @generated
	 */
	int DATASET = 6;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__NAME = 0;

	/**
	 * The feature id for the '<em><b>Dataset Path</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__DATASET_PATH = 1;

	/**
	 * The feature id for the '<em><b>Train Dataset Path</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__TRAIN_DATASET_PATH = 2;

	/**
	 * The feature id for the '<em><b>Test Dataset Path</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__TEST_DATASET_PATH = 3;

	/**
	 * The feature id for the '<em><b>Priviledged Group</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__PRIVILEDGED_GROUP = 4;

	/**
	 * The feature id for the '<em><b>Unpriviledged Group</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__UNPRIVILEDGED_GROUP = 5;

	/**
	 * The feature id for the '<em><b>Predicted Attribute</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__PREDICTED_ATTRIBUTE = 6;

	/**
	 * The feature id for the '<em><b>Favorable Classes</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__FAVORABLE_CLASSES = 7;

	/**
	 * The feature id for the '<em><b>Protected Attributes</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__PROTECTED_ATTRIBUTES = 8;

	/**
	 * The feature id for the '<em><b>Privileged Classes</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__PRIVILEGED_CLASSES = 9;

	/**
	 * The feature id for the '<em><b>Unprivileged Classes</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__UNPRIVILEGED_CLASSES = 10;

	/**
	 * The feature id for the '<em><b>Instance Weights</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__INSTANCE_WEIGHTS = 11;

	/**
	 * The feature id for the '<em><b>Categorical Features</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__CATEGORICAL_FEATURES = 12;

	/**
	 * The feature id for the '<em><b>Dropped Attributes</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__DROPPED_ATTRIBUTES = 13;

	/**
	 * The feature id for the '<em><b>Not Available Values</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__NOT_AVAILABLE_VALUES = 14;

	/**
	 * The feature id for the '<em><b>Default Mappings</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__DEFAULT_MAPPINGS = 15;

	/**
	 * The feature id for the '<em><b>Train Test Validation Split</b></em>' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET__TRAIN_TEST_VALIDATION_SPLIT = 16;

	/**
	 * The number of structural features of the '<em>Dataset</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATASET_FEATURE_COUNT = 17;

	/**
	 * The meta object id for the '{@link fairml.impl.BiasMitigationImpl <em>Bias Mitigation</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see fairml.impl.BiasMitigationImpl
	 * @see fairml.impl.FairmlPackageImpl#getBiasMitigation()
	 * @generated
	 */
	int BIAS_MITIGATION = 7;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__NAME = 0;

	/**
	 * The feature id for the '<em><b>Group Fairness</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__GROUP_FAIRNESS = 1;

	/**
	 * The feature id for the '<em><b>Individual Fairness</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__INDIVIDUAL_FAIRNESS = 2;

	/**
	 * The feature id for the '<em><b>Group Individual Single Metric</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__GROUP_INDIVIDUAL_SINGLE_METRIC = 3;

	/**
	 * The feature id for the '<em><b>Equal Fairness</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__EQUAL_FAIRNESS = 4;

	/**
	 * The feature id for the '<em><b>Proportional Fairness</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__PROPORTIONAL_FAIRNESS = 5;

	/**
	 * The feature id for the '<em><b>Check False Positive</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__CHECK_FALSE_POSITIVE = 6;

	/**
	 * The feature id for the '<em><b>Check False Negative</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__CHECK_FALSE_NEGATIVE = 7;

	/**
	 * The feature id for the '<em><b>Check Error Rate</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__CHECK_ERROR_RATE = 8;

	/**
	 * The feature id for the '<em><b>Check Equal Benefit</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__CHECK_EQUAL_BENEFIT = 9;

	/**
	 * The feature id for the '<em><b>Prepreprocessing Mitigation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__PREPREPROCESSING_MITIGATION = 10;

	/**
	 * The feature id for the '<em><b>Modifiable Weight</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__MODIFIABLE_WEIGHT = 11;

	/**
	 * The feature id for the '<em><b>Allow Latent Space</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__ALLOW_LATENT_SPACE = 12;

	/**
	 * The feature id for the '<em><b>Inpreprocessing Mitigation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__INPREPROCESSING_MITIGATION = 13;

	/**
	 * The feature id for the '<em><b>Allow Regularisation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__ALLOW_REGULARISATION = 14;

	/**
	 * The feature id for the '<em><b>Postpreprocessing Mitigation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__POSTPREPROCESSING_MITIGATION = 15;

	/**
	 * The feature id for the '<em><b>Allow Randomisation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__ALLOW_RANDOMISATION = 16;

	/**
	 * The feature id for the '<em><b>Datasets</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__DATASETS = 17;

	/**
	 * The feature id for the '<em><b>Bias Metrics</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__BIAS_METRICS = 18;

	/**
	 * The feature id for the '<em><b>Mitigation Methods</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__MITIGATION_METHODS = 19;

	/**
	 * The feature id for the '<em><b>Training Methods</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION__TRAINING_METHODS = 20;

	/**
	 * The number of structural features of the '<em>Bias Mitigation</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIAS_MITIGATION_FEATURE_COUNT = 21;


	/**
	 * Returns the meta object for class '{@link fairml.FairML <em>Fair ML</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Fair ML</em>'.
	 * @see fairml.FairML
	 * @generated
	 */
	EClass getFairML();

	/**
	 * Returns the meta object for the attribute '{@link fairml.FairML#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see fairml.FairML#getName()
	 * @see #getFairML()
	 * @generated
	 */
	EAttribute getFairML_Name();

	/**
	 * Returns the meta object for the attribute '{@link fairml.FairML#getDescription <em>Description</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Description</em>'.
	 * @see fairml.FairML#getDescription()
	 * @see #getFairML()
	 * @generated
	 */
	EAttribute getFairML_Description();

	/**
	 * Returns the meta object for the attribute '{@link fairml.FairML#getFilename <em>Filename</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Filename</em>'.
	 * @see fairml.FairML#getFilename()
	 * @see #getFairML()
	 * @generated
	 */
	EAttribute getFairML_Filename();

	/**
	 * Returns the meta object for the containment reference list '{@link fairml.FairML#getDatasets <em>Datasets</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Datasets</em>'.
	 * @see fairml.FairML#getDatasets()
	 * @see #getFairML()
	 * @generated
	 */
	EReference getFairML_Datasets();

	/**
	 * Returns the meta object for the containment reference list '{@link fairml.FairML#getBiasMitigations <em>Bias Mitigations</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Bias Mitigations</em>'.
	 * @see fairml.FairML#getBiasMitigations()
	 * @see #getFairML()
	 * @generated
	 */
	EReference getFairML_BiasMitigations();

	/**
	 * Returns the meta object for class '{@link fairml.Operation <em>Operation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Operation</em>'.
	 * @see fairml.Operation
	 * @generated
	 */
	EClass getOperation();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Operation#getPackageName <em>Package Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Package Name</em>'.
	 * @see fairml.Operation#getPackageName()
	 * @see #getOperation()
	 * @generated
	 */
	EAttribute getOperation_PackageName();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Operation#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see fairml.Operation#getName()
	 * @see #getOperation()
	 * @generated
	 */
	EAttribute getOperation_Name();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Operation#getParameters <em>Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Parameters</em>'.
	 * @see fairml.Operation#getParameters()
	 * @see #getOperation()
	 * @generated
	 */
	EAttribute getOperation_Parameters();

	/**
	 * Returns the meta object for the containment reference list '{@link fairml.Operation#getFunctions <em>Functions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Functions</em>'.
	 * @see fairml.Operation#getFunctions()
	 * @see #getOperation()
	 * @generated
	 */
	EReference getOperation_Functions();

	/**
	 * Returns the meta object for class '{@link fairml.Function <em>Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Function</em>'.
	 * @see fairml.Function
	 * @generated
	 */
	EClass getFunction();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Function#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see fairml.Function#getName()
	 * @see #getFunction()
	 * @generated
	 */
	EAttribute getFunction_Name();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Function#getParameters <em>Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Parameters</em>'.
	 * @see fairml.Function#getParameters()
	 * @see #getFunction()
	 * @generated
	 */
	EAttribute getFunction_Parameters();

	/**
	 * Returns the meta object for class '{@link fairml.TrainingMethod <em>Training Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Training Method</em>'.
	 * @see fairml.TrainingMethod
	 * @generated
	 */
	EClass getTrainingMethod();

	/**
	 * Returns the meta object for the attribute '{@link fairml.TrainingMethod#getAlgorithm <em>Algorithm</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Algorithm</em>'.
	 * @see fairml.TrainingMethod#getAlgorithm()
	 * @see #getTrainingMethod()
	 * @generated
	 */
	EAttribute getTrainingMethod_Algorithm();

	/**
	 * Returns the meta object for class '{@link fairml.MitigationMethod <em>Mitigation Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Mitigation Method</em>'.
	 * @see fairml.MitigationMethod
	 * @generated
	 */
	EClass getMitigationMethod();

	/**
	 * Returns the meta object for the attribute '{@link fairml.MitigationMethod#getAlgorithm <em>Algorithm</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Algorithm</em>'.
	 * @see fairml.MitigationMethod#getAlgorithm()
	 * @see #getMitigationMethod()
	 * @generated
	 */
	EAttribute getMitigationMethod_Algorithm();

	/**
	 * Returns the meta object for class '{@link fairml.BiasMetric <em>Bias Metric</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Bias Metric</em>'.
	 * @see fairml.BiasMetric
	 * @generated
	 */
	EClass getBiasMetric();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMetric#getClassName <em>Class Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Class Name</em>'.
	 * @see fairml.BiasMetric#getClassName()
	 * @see #getBiasMetric()
	 * @generated
	 */
	EAttribute getBiasMetric_ClassName();

	/**
	 * Returns the meta object for class '{@link fairml.Dataset <em>Dataset</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Dataset</em>'.
	 * @see fairml.Dataset
	 * @generated
	 */
	EClass getDataset();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Dataset#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see fairml.Dataset#getName()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_Name();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Dataset#getDatasetPath <em>Dataset Path</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Dataset Path</em>'.
	 * @see fairml.Dataset#getDatasetPath()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_DatasetPath();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Dataset#getTrainDatasetPath <em>Train Dataset Path</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Train Dataset Path</em>'.
	 * @see fairml.Dataset#getTrainDatasetPath()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_TrainDatasetPath();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Dataset#getTestDatasetPath <em>Test Dataset Path</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Test Dataset Path</em>'.
	 * @see fairml.Dataset#getTestDatasetPath()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_TestDatasetPath();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Dataset#getPriviledgedGroup <em>Priviledged Group</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Priviledged Group</em>'.
	 * @see fairml.Dataset#getPriviledgedGroup()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_PriviledgedGroup();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Dataset#getUnpriviledgedGroup <em>Unpriviledged Group</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Unpriviledged Group</em>'.
	 * @see fairml.Dataset#getUnpriviledgedGroup()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_UnpriviledgedGroup();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Dataset#getPredictedAttribute <em>Predicted Attribute</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Predicted Attribute</em>'.
	 * @see fairml.Dataset#getPredictedAttribute()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_PredictedAttribute();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getFavorableClasses <em>Favorable Classes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Favorable Classes</em>'.
	 * @see fairml.Dataset#getFavorableClasses()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_FavorableClasses();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getProtectedAttributes <em>Protected Attributes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Protected Attributes</em>'.
	 * @see fairml.Dataset#getProtectedAttributes()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_ProtectedAttributes();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getPrivilegedClasses <em>Privileged Classes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Privileged Classes</em>'.
	 * @see fairml.Dataset#getPrivilegedClasses()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_PrivilegedClasses();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getUnprivilegedClasses <em>Unprivileged Classes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Unprivileged Classes</em>'.
	 * @see fairml.Dataset#getUnprivilegedClasses()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_UnprivilegedClasses();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getInstanceWeights <em>Instance Weights</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Instance Weights</em>'.
	 * @see fairml.Dataset#getInstanceWeights()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_InstanceWeights();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getCategoricalFeatures <em>Categorical Features</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Categorical Features</em>'.
	 * @see fairml.Dataset#getCategoricalFeatures()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_CategoricalFeatures();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getDroppedAttributes <em>Dropped Attributes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Dropped Attributes</em>'.
	 * @see fairml.Dataset#getDroppedAttributes()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_DroppedAttributes();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getNotAvailableValues <em>Not Available Values</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Not Available Values</em>'.
	 * @see fairml.Dataset#getNotAvailableValues()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_NotAvailableValues();

	/**
	 * Returns the meta object for the attribute '{@link fairml.Dataset#getDefaultMappings <em>Default Mappings</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Default Mappings</em>'.
	 * @see fairml.Dataset#getDefaultMappings()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_DefaultMappings();

	/**
	 * Returns the meta object for the attribute list '{@link fairml.Dataset#getTrainTestValidationSplit <em>Train Test Validation Split</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute list '<em>Train Test Validation Split</em>'.
	 * @see fairml.Dataset#getTrainTestValidationSplit()
	 * @see #getDataset()
	 * @generated
	 */
	EAttribute getDataset_TrainTestValidationSplit();

	/**
	 * Returns the meta object for class '{@link fairml.BiasMitigation <em>Bias Mitigation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Bias Mitigation</em>'.
	 * @see fairml.BiasMitigation
	 * @generated
	 */
	EClass getBiasMitigation();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see fairml.BiasMitigation#getName()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_Name();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isGroupFairness <em>Group Fairness</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Group Fairness</em>'.
	 * @see fairml.BiasMitigation#isGroupFairness()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_GroupFairness();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isIndividualFairness <em>Individual Fairness</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Individual Fairness</em>'.
	 * @see fairml.BiasMitigation#isIndividualFairness()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_IndividualFairness();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isGroupIndividualSingleMetric <em>Group Individual Single Metric</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Group Individual Single Metric</em>'.
	 * @see fairml.BiasMitigation#isGroupIndividualSingleMetric()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_GroupIndividualSingleMetric();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isEqualFairness <em>Equal Fairness</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Equal Fairness</em>'.
	 * @see fairml.BiasMitigation#isEqualFairness()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_EqualFairness();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isProportionalFairness <em>Proportional Fairness</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Proportional Fairness</em>'.
	 * @see fairml.BiasMitigation#isProportionalFairness()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_ProportionalFairness();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isCheckFalsePositive <em>Check False Positive</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Check False Positive</em>'.
	 * @see fairml.BiasMitigation#isCheckFalsePositive()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_CheckFalsePositive();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isCheckFalseNegative <em>Check False Negative</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Check False Negative</em>'.
	 * @see fairml.BiasMitigation#isCheckFalseNegative()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_CheckFalseNegative();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isCheckErrorRate <em>Check Error Rate</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Check Error Rate</em>'.
	 * @see fairml.BiasMitigation#isCheckErrorRate()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_CheckErrorRate();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isCheckEqualBenefit <em>Check Equal Benefit</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Check Equal Benefit</em>'.
	 * @see fairml.BiasMitigation#isCheckEqualBenefit()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_CheckEqualBenefit();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isPrepreprocessingMitigation <em>Prepreprocessing Mitigation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Prepreprocessing Mitigation</em>'.
	 * @see fairml.BiasMitigation#isPrepreprocessingMitigation()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_PrepreprocessingMitigation();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isModifiableWeight <em>Modifiable Weight</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Modifiable Weight</em>'.
	 * @see fairml.BiasMitigation#isModifiableWeight()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_ModifiableWeight();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isAllowLatentSpace <em>Allow Latent Space</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Allow Latent Space</em>'.
	 * @see fairml.BiasMitigation#isAllowLatentSpace()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_AllowLatentSpace();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isInpreprocessingMitigation <em>Inpreprocessing Mitigation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Inpreprocessing Mitigation</em>'.
	 * @see fairml.BiasMitigation#isInpreprocessingMitigation()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_InpreprocessingMitigation();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isAllowRegularisation <em>Allow Regularisation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Allow Regularisation</em>'.
	 * @see fairml.BiasMitigation#isAllowRegularisation()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_AllowRegularisation();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isPostpreprocessingMitigation <em>Postpreprocessing Mitigation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Postpreprocessing Mitigation</em>'.
	 * @see fairml.BiasMitigation#isPostpreprocessingMitigation()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_PostpreprocessingMitigation();

	/**
	 * Returns the meta object for the attribute '{@link fairml.BiasMitigation#isAllowRandomisation <em>Allow Randomisation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Allow Randomisation</em>'.
	 * @see fairml.BiasMitigation#isAllowRandomisation()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EAttribute getBiasMitigation_AllowRandomisation();

	/**
	 * Returns the meta object for the reference list '{@link fairml.BiasMitigation#getDatasets <em>Datasets</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Datasets</em>'.
	 * @see fairml.BiasMitigation#getDatasets()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EReference getBiasMitigation_Datasets();

	/**
	 * Returns the meta object for the containment reference list '{@link fairml.BiasMitigation#getBiasMetrics <em>Bias Metrics</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Bias Metrics</em>'.
	 * @see fairml.BiasMitigation#getBiasMetrics()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EReference getBiasMitigation_BiasMetrics();

	/**
	 * Returns the meta object for the containment reference list '{@link fairml.BiasMitigation#getMitigationMethods <em>Mitigation Methods</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Mitigation Methods</em>'.
	 * @see fairml.BiasMitigation#getMitigationMethods()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EReference getBiasMitigation_MitigationMethods();

	/**
	 * Returns the meta object for the containment reference list '{@link fairml.BiasMitigation#getTrainingMethods <em>Training Methods</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Training Methods</em>'.
	 * @see fairml.BiasMitigation#getTrainingMethods()
	 * @see #getBiasMitigation()
	 * @generated
	 */
	EReference getBiasMitigation_TrainingMethods();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	FairmlFactory getFairmlFactory();

	/**
	 * <!-- begin-user-doc -->
	 * Defines literals for the meta objects that represent
	 * <ul>
	 *   <li>each class,</li>
	 *   <li>each feature of each class,</li>
	 *   <li>each enum,</li>
	 *   <li>and each data type</li>
	 * </ul>
	 * <!-- end-user-doc -->
	 * @generated
	 */
	interface Literals {
		/**
		 * The meta object literal for the '{@link fairml.impl.FairMLImpl <em>Fair ML</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see fairml.impl.FairMLImpl
		 * @see fairml.impl.FairmlPackageImpl#getFairML()
		 * @generated
		 */
		EClass FAIR_ML = eINSTANCE.getFairML();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute FAIR_ML__NAME = eINSTANCE.getFairML_Name();

		/**
		 * The meta object literal for the '<em><b>Description</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute FAIR_ML__DESCRIPTION = eINSTANCE.getFairML_Description();

		/**
		 * The meta object literal for the '<em><b>Filename</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute FAIR_ML__FILENAME = eINSTANCE.getFairML_Filename();

		/**
		 * The meta object literal for the '<em><b>Datasets</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference FAIR_ML__DATASETS = eINSTANCE.getFairML_Datasets();

		/**
		 * The meta object literal for the '<em><b>Bias Mitigations</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference FAIR_ML__BIAS_MITIGATIONS = eINSTANCE.getFairML_BiasMitigations();

		/**
		 * The meta object literal for the '{@link fairml.impl.OperationImpl <em>Operation</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see fairml.impl.OperationImpl
		 * @see fairml.impl.FairmlPackageImpl#getOperation()
		 * @generated
		 */
		EClass OPERATION = eINSTANCE.getOperation();

		/**
		 * The meta object literal for the '<em><b>Package Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute OPERATION__PACKAGE_NAME = eINSTANCE.getOperation_PackageName();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute OPERATION__NAME = eINSTANCE.getOperation_Name();

		/**
		 * The meta object literal for the '<em><b>Parameters</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute OPERATION__PARAMETERS = eINSTANCE.getOperation_Parameters();

		/**
		 * The meta object literal for the '<em><b>Functions</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference OPERATION__FUNCTIONS = eINSTANCE.getOperation_Functions();

		/**
		 * The meta object literal for the '{@link fairml.impl.FunctionImpl <em>Function</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see fairml.impl.FunctionImpl
		 * @see fairml.impl.FairmlPackageImpl#getFunction()
		 * @generated
		 */
		EClass FUNCTION = eINSTANCE.getFunction();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute FUNCTION__NAME = eINSTANCE.getFunction_Name();

		/**
		 * The meta object literal for the '<em><b>Parameters</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute FUNCTION__PARAMETERS = eINSTANCE.getFunction_Parameters();

		/**
		 * The meta object literal for the '{@link fairml.impl.TrainingMethodImpl <em>Training Method</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see fairml.impl.TrainingMethodImpl
		 * @see fairml.impl.FairmlPackageImpl#getTrainingMethod()
		 * @generated
		 */
		EClass TRAINING_METHOD = eINSTANCE.getTrainingMethod();

		/**
		 * The meta object literal for the '<em><b>Algorithm</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute TRAINING_METHOD__ALGORITHM = eINSTANCE.getTrainingMethod_Algorithm();

		/**
		 * The meta object literal for the '{@link fairml.impl.MitigationMethodImpl <em>Mitigation Method</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see fairml.impl.MitigationMethodImpl
		 * @see fairml.impl.FairmlPackageImpl#getMitigationMethod()
		 * @generated
		 */
		EClass MITIGATION_METHOD = eINSTANCE.getMitigationMethod();

		/**
		 * The meta object literal for the '<em><b>Algorithm</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MITIGATION_METHOD__ALGORITHM = eINSTANCE.getMitigationMethod_Algorithm();

		/**
		 * The meta object literal for the '{@link fairml.impl.BiasMetricImpl <em>Bias Metric</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see fairml.impl.BiasMetricImpl
		 * @see fairml.impl.FairmlPackageImpl#getBiasMetric()
		 * @generated
		 */
		EClass BIAS_METRIC = eINSTANCE.getBiasMetric();

		/**
		 * The meta object literal for the '<em><b>Class Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_METRIC__CLASS_NAME = eINSTANCE.getBiasMetric_ClassName();

		/**
		 * The meta object literal for the '{@link fairml.impl.DatasetImpl <em>Dataset</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see fairml.impl.DatasetImpl
		 * @see fairml.impl.FairmlPackageImpl#getDataset()
		 * @generated
		 */
		EClass DATASET = eINSTANCE.getDataset();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__NAME = eINSTANCE.getDataset_Name();

		/**
		 * The meta object literal for the '<em><b>Dataset Path</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__DATASET_PATH = eINSTANCE.getDataset_DatasetPath();

		/**
		 * The meta object literal for the '<em><b>Train Dataset Path</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__TRAIN_DATASET_PATH = eINSTANCE.getDataset_TrainDatasetPath();

		/**
		 * The meta object literal for the '<em><b>Test Dataset Path</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__TEST_DATASET_PATH = eINSTANCE.getDataset_TestDatasetPath();

		/**
		 * The meta object literal for the '<em><b>Priviledged Group</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__PRIVILEDGED_GROUP = eINSTANCE.getDataset_PriviledgedGroup();

		/**
		 * The meta object literal for the '<em><b>Unpriviledged Group</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__UNPRIVILEDGED_GROUP = eINSTANCE.getDataset_UnpriviledgedGroup();

		/**
		 * The meta object literal for the '<em><b>Predicted Attribute</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__PREDICTED_ATTRIBUTE = eINSTANCE.getDataset_PredictedAttribute();

		/**
		 * The meta object literal for the '<em><b>Favorable Classes</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__FAVORABLE_CLASSES = eINSTANCE.getDataset_FavorableClasses();

		/**
		 * The meta object literal for the '<em><b>Protected Attributes</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__PROTECTED_ATTRIBUTES = eINSTANCE.getDataset_ProtectedAttributes();

		/**
		 * The meta object literal for the '<em><b>Privileged Classes</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__PRIVILEGED_CLASSES = eINSTANCE.getDataset_PrivilegedClasses();

		/**
		 * The meta object literal for the '<em><b>Unprivileged Classes</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__UNPRIVILEGED_CLASSES = eINSTANCE.getDataset_UnprivilegedClasses();

		/**
		 * The meta object literal for the '<em><b>Instance Weights</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__INSTANCE_WEIGHTS = eINSTANCE.getDataset_InstanceWeights();

		/**
		 * The meta object literal for the '<em><b>Categorical Features</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__CATEGORICAL_FEATURES = eINSTANCE.getDataset_CategoricalFeatures();

		/**
		 * The meta object literal for the '<em><b>Dropped Attributes</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__DROPPED_ATTRIBUTES = eINSTANCE.getDataset_DroppedAttributes();

		/**
		 * The meta object literal for the '<em><b>Not Available Values</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__NOT_AVAILABLE_VALUES = eINSTANCE.getDataset_NotAvailableValues();

		/**
		 * The meta object literal for the '<em><b>Default Mappings</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__DEFAULT_MAPPINGS = eINSTANCE.getDataset_DefaultMappings();

		/**
		 * The meta object literal for the '<em><b>Train Test Validation Split</b></em>' attribute list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATASET__TRAIN_TEST_VALIDATION_SPLIT = eINSTANCE.getDataset_TrainTestValidationSplit();

		/**
		 * The meta object literal for the '{@link fairml.impl.BiasMitigationImpl <em>Bias Mitigation</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see fairml.impl.BiasMitigationImpl
		 * @see fairml.impl.FairmlPackageImpl#getBiasMitigation()
		 * @generated
		 */
		EClass BIAS_MITIGATION = eINSTANCE.getBiasMitigation();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__NAME = eINSTANCE.getBiasMitigation_Name();

		/**
		 * The meta object literal for the '<em><b>Group Fairness</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__GROUP_FAIRNESS = eINSTANCE.getBiasMitigation_GroupFairness();

		/**
		 * The meta object literal for the '<em><b>Individual Fairness</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__INDIVIDUAL_FAIRNESS = eINSTANCE.getBiasMitigation_IndividualFairness();

		/**
		 * The meta object literal for the '<em><b>Group Individual Single Metric</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__GROUP_INDIVIDUAL_SINGLE_METRIC = eINSTANCE.getBiasMitigation_GroupIndividualSingleMetric();

		/**
		 * The meta object literal for the '<em><b>Equal Fairness</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__EQUAL_FAIRNESS = eINSTANCE.getBiasMitigation_EqualFairness();

		/**
		 * The meta object literal for the '<em><b>Proportional Fairness</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__PROPORTIONAL_FAIRNESS = eINSTANCE.getBiasMitigation_ProportionalFairness();

		/**
		 * The meta object literal for the '<em><b>Check False Positive</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__CHECK_FALSE_POSITIVE = eINSTANCE.getBiasMitigation_CheckFalsePositive();

		/**
		 * The meta object literal for the '<em><b>Check False Negative</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__CHECK_FALSE_NEGATIVE = eINSTANCE.getBiasMitigation_CheckFalseNegative();

		/**
		 * The meta object literal for the '<em><b>Check Error Rate</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__CHECK_ERROR_RATE = eINSTANCE.getBiasMitigation_CheckErrorRate();

		/**
		 * The meta object literal for the '<em><b>Check Equal Benefit</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__CHECK_EQUAL_BENEFIT = eINSTANCE.getBiasMitigation_CheckEqualBenefit();

		/**
		 * The meta object literal for the '<em><b>Prepreprocessing Mitigation</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__PREPREPROCESSING_MITIGATION = eINSTANCE.getBiasMitigation_PrepreprocessingMitigation();

		/**
		 * The meta object literal for the '<em><b>Modifiable Weight</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__MODIFIABLE_WEIGHT = eINSTANCE.getBiasMitigation_ModifiableWeight();

		/**
		 * The meta object literal for the '<em><b>Allow Latent Space</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__ALLOW_LATENT_SPACE = eINSTANCE.getBiasMitigation_AllowLatentSpace();

		/**
		 * The meta object literal for the '<em><b>Inpreprocessing Mitigation</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__INPREPROCESSING_MITIGATION = eINSTANCE.getBiasMitigation_InpreprocessingMitigation();

		/**
		 * The meta object literal for the '<em><b>Allow Regularisation</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__ALLOW_REGULARISATION = eINSTANCE.getBiasMitigation_AllowRegularisation();

		/**
		 * The meta object literal for the '<em><b>Postpreprocessing Mitigation</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__POSTPREPROCESSING_MITIGATION = eINSTANCE.getBiasMitigation_PostpreprocessingMitigation();

		/**
		 * The meta object literal for the '<em><b>Allow Randomisation</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIAS_MITIGATION__ALLOW_RANDOMISATION = eINSTANCE.getBiasMitigation_AllowRandomisation();

		/**
		 * The meta object literal for the '<em><b>Datasets</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIAS_MITIGATION__DATASETS = eINSTANCE.getBiasMitigation_Datasets();

		/**
		 * The meta object literal for the '<em><b>Bias Metrics</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIAS_MITIGATION__BIAS_METRICS = eINSTANCE.getBiasMitigation_BiasMetrics();

		/**
		 * The meta object literal for the '<em><b>Mitigation Methods</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIAS_MITIGATION__MITIGATION_METHODS = eINSTANCE.getBiasMitigation_MitigationMethods();

		/**
		 * The meta object literal for the '<em><b>Training Methods</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIAS_MITIGATION__TRAINING_METHODS = eINSTANCE.getBiasMitigation_TrainingMethods();

	}

} //FairmlPackage
