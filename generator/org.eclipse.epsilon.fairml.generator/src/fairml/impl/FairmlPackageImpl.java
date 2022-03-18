/**
 */
package fairml.impl;

import fairml.BiasMetric;
import fairml.BiasMitigation;
import fairml.Dataset;
import fairml.FairML;
import fairml.FairmlFactory;
import fairml.FairmlPackage;
import fairml.Function;
import fairml.MitigationMethod;
import fairml.MitigationType;
import fairml.Operation;
import fairml.TrainingMethod;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class FairmlPackageImpl extends EPackageImpl implements FairmlPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass fairMLEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass operationEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass functionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass trainingMethodEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass mitigationMethodEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass biasMetricEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass datasetEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass biasMitigationEClass = null;

	/**
	 * Creates an instance of the model <b>Package</b>, registered with
	 * {@link org.eclipse.emf.ecore.EPackage.Registry EPackage.Registry} by the package
	 * package URI value.
	 * <p>Note: the correct way to create the package is via the static
	 * factory method {@link #init init()}, which also performs
	 * initialization of the package, or returns the registered package,
	 * if one already exists.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see org.eclipse.emf.ecore.EPackage.Registry
	 * @see fairml.FairmlPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private FairmlPackageImpl() {
		super(eNS_URI, FairmlFactory.eINSTANCE);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static boolean isInited = false;

	/**
	 * Creates, registers, and initializes the <b>Package</b> for this model, and for any others upon which it depends.
	 *
	 * <p>This method is used to initialize {@link FairmlPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @see #createPackageContents()
	 * @see #initializePackageContents()
	 * @generated
	 */
	public static FairmlPackage init() {
		if (isInited) return (FairmlPackage)EPackage.Registry.INSTANCE.getEPackage(FairmlPackage.eNS_URI);

		// Obtain or create and register package
		Object registeredFairmlPackage = EPackage.Registry.INSTANCE.get(eNS_URI);
		FairmlPackageImpl theFairmlPackage = registeredFairmlPackage instanceof FairmlPackageImpl ? (FairmlPackageImpl)registeredFairmlPackage : new FairmlPackageImpl();

		isInited = true;

		// Create package meta-data objects
		theFairmlPackage.createPackageContents();

		// Initialize created meta-data
		theFairmlPackage.initializePackageContents();

		// Mark meta-data to indicate it can't be changed
		theFairmlPackage.freeze();

		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(FairmlPackage.eNS_URI, theFairmlPackage);
		return theFairmlPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EClass getFairML() {
		return fairMLEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getFairML_Name() {
		return (EAttribute)fairMLEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getFairML_Description() {
		return (EAttribute)fairMLEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getFairML_Filename() {
		return (EAttribute)fairMLEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EReference getFairML_Datasets() {
		return (EReference)fairMLEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EReference getFairML_BiasMitigations() {
		return (EReference)fairMLEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getFairML_Modules() {
		return (EAttribute)fairMLEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EClass getOperation() {
		return operationEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getOperation_PackageName() {
		return (EAttribute)operationEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getOperation_Name() {
		return (EAttribute)operationEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getOperation_Parameters() {
		return (EAttribute)operationEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EReference getOperation_Functions() {
		return (EReference)operationEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EClass getFunction() {
		return functionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getFunction_Name() {
		return (EAttribute)functionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getFunction_Parameters() {
		return (EAttribute)functionEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EClass getTrainingMethod() {
		return trainingMethodEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getTrainingMethod_Algorithm() {
		return (EAttribute)trainingMethodEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getTrainingMethod_FitParameters() {
		return (EAttribute)trainingMethodEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getTrainingMethod_PredictParameters() {
		return (EAttribute)trainingMethodEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getTrainingMethod_WithoutWeight() {
		return (EAttribute)trainingMethodEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EClass getMitigationMethod() {
		return mitigationMethodEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getMitigationMethod_Algorithm() {
		return (EAttribute)mitigationMethodEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getMitigationMethod_FitParameters() {
		return (EAttribute)mitigationMethodEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getMitigationMethod_PredictParameters() {
		return (EAttribute)mitigationMethodEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EClass getBiasMetric() {
		return biasMetricEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMetric_ClassName() {
		return (EAttribute)biasMetricEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMetric_DatasetType() {
		return (EAttribute)biasMetricEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMetric_OptimalThreshold() {
		return (EAttribute)biasMetricEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMetric_PlotThreshold() {
		return (EAttribute)biasMetricEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EClass getDataset() {
		return datasetEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_Name() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_DatasetPath() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_DatasetModule() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_TrainDatasetModule() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_TestDatasetModule() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_ValidationDatasetModule() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_DatasetModuleParameters() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_TrainDatasetModuleParameters() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(7);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_TestDatasetModuleParameters() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(8);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_ValidationDatasetModuleParameters() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(9);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_PriviledgedGroup() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(10);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_UnpriviledgedGroup() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(11);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_PredictedAttribute() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(12);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_FavorableClasses() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(13);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_ProtectedAttributes() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(14);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_PrivilegedClasses() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(15);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_UnprivilegedClasses() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(16);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_InstanceWeights() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(17);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_CategoricalFeatures() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(18);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_DroppedAttributes() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(19);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_FeaturesToKeep() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(20);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_NotAvailableValues() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(21);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_DefaultMappings() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(22);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getDataset_TrainTestValidationSplit() {
		return (EAttribute)datasetEClass.getEStructuralFeatures().get(23);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EClass getBiasMitigation() {
		return biasMitigationEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_Name() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_GroupFairness() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_IndividualFairness() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_GroupIndividualSingleMetric() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_EqualFairness() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_ProportionalFairness() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_CheckFalsePositive() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_CheckFalseNegative() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(7);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_CheckErrorRate() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(8);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_CheckEqualBenefit() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(9);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_PrepreprocessingMitigation() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(10);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_ModifiableWeight() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(11);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_AllowLatentSpace() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(12);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_InpreprocessingMitigation() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(13);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_AllowRegularisation() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(14);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_PostpreprocessingMitigation() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(15);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EAttribute getBiasMitigation_AllowRandomisation() {
		return (EAttribute)biasMitigationEClass.getEStructuralFeatures().get(16);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EReference getBiasMitigation_Datasets() {
		return (EReference)biasMitigationEClass.getEStructuralFeatures().get(17);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EReference getBiasMitigation_BiasMetrics() {
		return (EReference)biasMitigationEClass.getEStructuralFeatures().get(18);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EReference getBiasMitigation_MitigationMethods() {
		return (EReference)biasMitigationEClass.getEStructuralFeatures().get(19);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EReference getBiasMitigation_TrainingMethods() {
		return (EReference)biasMitigationEClass.getEStructuralFeatures().get(20);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public FairmlFactory getFairmlFactory() {
		return (FairmlFactory)getEFactoryInstance();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isCreated = false;

	/**
	 * Creates the meta-model objects for the package.  This method is
	 * guarded to have no affect on any invocation but its first.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void createPackageContents() {
		if (isCreated) return;
		isCreated = true;

		// Create classes and their features
		fairMLEClass = createEClass(FAIR_ML);
		createEAttribute(fairMLEClass, FAIR_ML__NAME);
		createEAttribute(fairMLEClass, FAIR_ML__DESCRIPTION);
		createEAttribute(fairMLEClass, FAIR_ML__FILENAME);
		createEReference(fairMLEClass, FAIR_ML__DATASETS);
		createEReference(fairMLEClass, FAIR_ML__BIAS_MITIGATIONS);
		createEAttribute(fairMLEClass, FAIR_ML__MODULES);

		operationEClass = createEClass(OPERATION);
		createEAttribute(operationEClass, OPERATION__PACKAGE_NAME);
		createEAttribute(operationEClass, OPERATION__NAME);
		createEAttribute(operationEClass, OPERATION__PARAMETERS);
		createEReference(operationEClass, OPERATION__FUNCTIONS);

		functionEClass = createEClass(FUNCTION);
		createEAttribute(functionEClass, FUNCTION__NAME);
		createEAttribute(functionEClass, FUNCTION__PARAMETERS);

		trainingMethodEClass = createEClass(TRAINING_METHOD);
		createEAttribute(trainingMethodEClass, TRAINING_METHOD__ALGORITHM);
		createEAttribute(trainingMethodEClass, TRAINING_METHOD__FIT_PARAMETERS);
		createEAttribute(trainingMethodEClass, TRAINING_METHOD__PREDICT_PARAMETERS);
		createEAttribute(trainingMethodEClass, TRAINING_METHOD__WITHOUT_WEIGHT);

		mitigationMethodEClass = createEClass(MITIGATION_METHOD);
		createEAttribute(mitigationMethodEClass, MITIGATION_METHOD__ALGORITHM);
		createEAttribute(mitigationMethodEClass, MITIGATION_METHOD__FIT_PARAMETERS);
		createEAttribute(mitigationMethodEClass, MITIGATION_METHOD__PREDICT_PARAMETERS);

		biasMetricEClass = createEClass(BIAS_METRIC);
		createEAttribute(biasMetricEClass, BIAS_METRIC__CLASS_NAME);
		createEAttribute(biasMetricEClass, BIAS_METRIC__DATASET_TYPE);
		createEAttribute(biasMetricEClass, BIAS_METRIC__OPTIMAL_THRESHOLD);
		createEAttribute(biasMetricEClass, BIAS_METRIC__PLOT_THRESHOLD);

		datasetEClass = createEClass(DATASET);
		createEAttribute(datasetEClass, DATASET__NAME);
		createEAttribute(datasetEClass, DATASET__DATASET_PATH);
		createEAttribute(datasetEClass, DATASET__DATASET_MODULE);
		createEAttribute(datasetEClass, DATASET__TRAIN_DATASET_MODULE);
		createEAttribute(datasetEClass, DATASET__TEST_DATASET_MODULE);
		createEAttribute(datasetEClass, DATASET__VALIDATION_DATASET_MODULE);
		createEAttribute(datasetEClass, DATASET__DATASET_MODULE_PARAMETERS);
		createEAttribute(datasetEClass, DATASET__TRAIN_DATASET_MODULE_PARAMETERS);
		createEAttribute(datasetEClass, DATASET__TEST_DATASET_MODULE_PARAMETERS);
		createEAttribute(datasetEClass, DATASET__VALIDATION_DATASET_MODULE_PARAMETERS);
		createEAttribute(datasetEClass, DATASET__PRIVILEDGED_GROUP);
		createEAttribute(datasetEClass, DATASET__UNPRIVILEDGED_GROUP);
		createEAttribute(datasetEClass, DATASET__PREDICTED_ATTRIBUTE);
		createEAttribute(datasetEClass, DATASET__FAVORABLE_CLASSES);
		createEAttribute(datasetEClass, DATASET__PROTECTED_ATTRIBUTES);
		createEAttribute(datasetEClass, DATASET__PRIVILEGED_CLASSES);
		createEAttribute(datasetEClass, DATASET__UNPRIVILEGED_CLASSES);
		createEAttribute(datasetEClass, DATASET__INSTANCE_WEIGHTS);
		createEAttribute(datasetEClass, DATASET__CATEGORICAL_FEATURES);
		createEAttribute(datasetEClass, DATASET__DROPPED_ATTRIBUTES);
		createEAttribute(datasetEClass, DATASET__FEATURES_TO_KEEP);
		createEAttribute(datasetEClass, DATASET__NOT_AVAILABLE_VALUES);
		createEAttribute(datasetEClass, DATASET__DEFAULT_MAPPINGS);
		createEAttribute(datasetEClass, DATASET__TRAIN_TEST_VALIDATION_SPLIT);

		biasMitigationEClass = createEClass(BIAS_MITIGATION);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__NAME);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__GROUP_FAIRNESS);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__INDIVIDUAL_FAIRNESS);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__GROUP_INDIVIDUAL_SINGLE_METRIC);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__EQUAL_FAIRNESS);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__PROPORTIONAL_FAIRNESS);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__CHECK_FALSE_POSITIVE);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__CHECK_FALSE_NEGATIVE);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__CHECK_ERROR_RATE);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__CHECK_EQUAL_BENEFIT);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__PREPREPROCESSING_MITIGATION);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__MODIFIABLE_WEIGHT);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__ALLOW_LATENT_SPACE);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__INPREPROCESSING_MITIGATION);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__ALLOW_REGULARISATION);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__POSTPREPROCESSING_MITIGATION);
		createEAttribute(biasMitigationEClass, BIAS_MITIGATION__ALLOW_RANDOMISATION);
		createEReference(biasMitigationEClass, BIAS_MITIGATION__DATASETS);
		createEReference(biasMitigationEClass, BIAS_MITIGATION__BIAS_METRICS);
		createEReference(biasMitigationEClass, BIAS_MITIGATION__MITIGATION_METHODS);
		createEReference(biasMitigationEClass, BIAS_MITIGATION__TRAINING_METHODS);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isInitialized = false;

	/**
	 * Complete the initialization of the package and its meta-model.  This
	 * method is guarded to have no affect on any invocation but its first.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void initializePackageContents() {
		if (isInitialized) return;
		isInitialized = true;

		// Initialize package
		setName(eNAME);
		setNsPrefix(eNS_PREFIX);
		setNsURI(eNS_URI);

		// Create type parameters

		// Set bounds for type parameters

		// Add supertypes to classes
		trainingMethodEClass.getESuperTypes().add(this.getOperation());
		mitigationMethodEClass.getESuperTypes().add(this.getOperation());
		biasMetricEClass.getESuperTypes().add(this.getOperation());

		// Initialize classes and features; add operations and parameters
		initEClass(fairMLEClass, FairML.class, "FairML", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getFairML_Name(), ecorePackage.getEString(), "name", null, 0, 1, FairML.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getFairML_Description(), ecorePackage.getEString(), "description", null, 0, 1, FairML.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getFairML_Filename(), ecorePackage.getEString(), "filename", null, 0, 1, FairML.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getFairML_Datasets(), this.getDataset(), null, "datasets", null, 0, -1, FairML.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getFairML_BiasMitigations(), this.getBiasMitigation(), null, "biasMitigations", null, 0, -1, FairML.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getFairML_Modules(), ecorePackage.getEString(), "modules", null, 0, -1, FairML.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(operationEClass, Operation.class, "Operation", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getOperation_PackageName(), ecorePackage.getEString(), "packageName", null, 0, 1, Operation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getOperation_Name(), ecorePackage.getEString(), "name", null, 0, 1, Operation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getOperation_Parameters(), ecorePackage.getEString(), "parameters", null, 0, -1, Operation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getOperation_Functions(), this.getFunction(), null, "functions", null, 0, -1, Operation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(functionEClass, Function.class, "Function", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getFunction_Name(), ecorePackage.getEString(), "name", null, 0, 1, Function.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getFunction_Parameters(), ecorePackage.getEString(), "parameters", null, 0, -1, Function.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(trainingMethodEClass, TrainingMethod.class, "TrainingMethod", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getTrainingMethod_Algorithm(), ecorePackage.getEString(), "algorithm", null, 0, 1, TrainingMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getTrainingMethod_FitParameters(), ecorePackage.getEString(), "fitParameters", null, 0, -1, TrainingMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getTrainingMethod_PredictParameters(), ecorePackage.getEString(), "predictParameters", null, 0, -1, TrainingMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getTrainingMethod_WithoutWeight(), ecorePackage.getEBoolean(), "withoutWeight", "true", 0, 1, TrainingMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(mitigationMethodEClass, MitigationMethod.class, "MitigationMethod", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getMitigationMethod_Algorithm(), ecorePackage.getEString(), "algorithm", null, 0, 1, MitigationMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getMitigationMethod_FitParameters(), ecorePackage.getEString(), "fitParameters", null, 0, -1, MitigationMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getMitigationMethod_PredictParameters(), ecorePackage.getEString(), "predictParameters", null, 0, -1, MitigationMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(biasMetricEClass, BiasMetric.class, "BiasMetric", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getBiasMetric_ClassName(), ecorePackage.getEString(), "className", "FairMLMetric", 0, 1, BiasMetric.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMetric_DatasetType(), ecorePackage.getEString(), "datasetType", "test", 0, 1, BiasMetric.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMetric_OptimalThreshold(), ecorePackage.getEBoolean(), "optimalThreshold", "false", 0, 1, BiasMetric.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMetric_PlotThreshold(), ecorePackage.getEBoolean(), "plotThreshold", "false", 0, 1, BiasMetric.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(datasetEClass, Dataset.class, "Dataset", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getDataset_Name(), ecorePackage.getEString(), "name", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_DatasetPath(), ecorePackage.getEString(), "datasetPath", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_DatasetModule(), ecorePackage.getEString(), "datasetModule", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_TrainDatasetModule(), ecorePackage.getEString(), "trainDatasetModule", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_TestDatasetModule(), ecorePackage.getEString(), "testDatasetModule", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_ValidationDatasetModule(), ecorePackage.getEString(), "validationDatasetModule", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_DatasetModuleParameters(), ecorePackage.getEString(), "datasetModuleParameters", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_TrainDatasetModuleParameters(), ecorePackage.getEString(), "trainDatasetModuleParameters", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_TestDatasetModuleParameters(), ecorePackage.getEString(), "testDatasetModuleParameters", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_ValidationDatasetModuleParameters(), ecorePackage.getEString(), "validationDatasetModuleParameters", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_PriviledgedGroup(), ecorePackage.getEInt(), "priviledgedGroup", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_UnpriviledgedGroup(), ecorePackage.getEInt(), "unpriviledgedGroup", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_PredictedAttribute(), ecorePackage.getEString(), "predictedAttribute", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_FavorableClasses(), ecorePackage.getEInt(), "favorableClasses", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_ProtectedAttributes(), ecorePackage.getEString(), "protectedAttributes", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_PrivilegedClasses(), ecorePackage.getEInt(), "privilegedClasses", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_UnprivilegedClasses(), ecorePackage.getEInt(), "unprivilegedClasses", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_InstanceWeights(), ecorePackage.getEString(), "instanceWeights", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_CategoricalFeatures(), ecorePackage.getEString(), "categoricalFeatures", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_DroppedAttributes(), ecorePackage.getEString(), "droppedAttributes", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_FeaturesToKeep(), ecorePackage.getEString(), "featuresToKeep", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_NotAvailableValues(), ecorePackage.getEString(), "notAvailableValues", null, 0, -1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_DefaultMappings(), ecorePackage.getEString(), "defaultMappings", null, 0, 1, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataset_TrainTestValidationSplit(), ecorePackage.getEFloat(), "trainTestValidationSplit", null, 0, 3, Dataset.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(biasMitigationEClass, BiasMitigation.class, "BiasMitigation", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getBiasMitigation_Name(), ecorePackage.getEString(), "name", null, 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_GroupFairness(), ecorePackage.getEBoolean(), "groupFairness", "true", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_IndividualFairness(), ecorePackage.getEBoolean(), "individualFairness", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_GroupIndividualSingleMetric(), ecorePackage.getEBoolean(), "groupIndividualSingleMetric", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_EqualFairness(), ecorePackage.getEBoolean(), "equalFairness", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_ProportionalFairness(), ecorePackage.getEBoolean(), "proportionalFairness", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_CheckFalsePositive(), ecorePackage.getEBoolean(), "checkFalsePositive", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_CheckFalseNegative(), ecorePackage.getEBoolean(), "checkFalseNegative", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_CheckErrorRate(), ecorePackage.getEBoolean(), "checkErrorRate", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_CheckEqualBenefit(), ecorePackage.getEBoolean(), "checkEqualBenefit", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_PrepreprocessingMitigation(), ecorePackage.getEBoolean(), "prepreprocessingMitigation", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_ModifiableWeight(), ecorePackage.getEBoolean(), "modifiableWeight", "true", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_AllowLatentSpace(), ecorePackage.getEBoolean(), "allowLatentSpace", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_InpreprocessingMitigation(), ecorePackage.getEBoolean(), "inpreprocessingMitigation", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_AllowRegularisation(), ecorePackage.getEBoolean(), "allowRegularisation", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_PostpreprocessingMitigation(), ecorePackage.getEBoolean(), "postpreprocessingMitigation", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getBiasMitigation_AllowRandomisation(), ecorePackage.getEBoolean(), "allowRandomisation", "false", 0, 1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getBiasMitigation_Datasets(), this.getDataset(), null, "datasets", null, 0, -1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getBiasMitigation_BiasMetrics(), this.getBiasMetric(), null, "biasMetrics", null, 0, -1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getBiasMitigation_MitigationMethods(), this.getMitigationMethod(), null, "mitigationMethods", null, 0, -1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getBiasMitigation_TrainingMethods(), this.getTrainingMethod(), null, "trainingMethods", null, 0, -1, BiasMitigation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		// Create resource
		createResource(eNS_URI);
	}

} //FairmlPackageImpl
