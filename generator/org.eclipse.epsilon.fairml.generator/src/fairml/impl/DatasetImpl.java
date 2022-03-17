/**
 */
package fairml.impl;

import fairml.Dataset;
import fairml.FairmlPackage;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EDataTypeEList;
import org.eclipse.emf.ecore.util.EDataTypeUniqueEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Dataset</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link fairml.impl.DatasetImpl#getName <em>Name</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getDatasetPath <em>Dataset Path</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getDatasetModule <em>Dataset Module</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getTrainDatasetModule <em>Train Dataset Module</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getTestDatasetModule <em>Test Dataset Module</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getValidationDatasetModule <em>Validation Dataset Module</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getDatasetModuleParameters <em>Dataset Module Parameters</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getTrainDatasetModuleParameters <em>Train Dataset Module Parameters</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getTestDatasetModuleParameters <em>Test Dataset Module Parameters</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getValidationDatasetModuleParameters <em>Validation Dataset Module Parameters</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getPriviledgedGroup <em>Priviledged Group</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getUnpriviledgedGroup <em>Unpriviledged Group</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getPredictedAttribute <em>Predicted Attribute</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getFavorableClasses <em>Favorable Classes</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getProtectedAttributes <em>Protected Attributes</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getPrivilegedClasses <em>Privileged Classes</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getUnprivilegedClasses <em>Unprivileged Classes</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getInstanceWeights <em>Instance Weights</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getCategoricalFeatures <em>Categorical Features</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getDroppedAttributes <em>Dropped Attributes</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getFeaturesToKeep <em>Features To Keep</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getNotAvailableValues <em>Not Available Values</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getDefaultMappings <em>Default Mappings</em>}</li>
 *   <li>{@link fairml.impl.DatasetImpl#getTrainTestValidationSplit <em>Train Test Validation Split</em>}</li>
 * </ul>
 *
 * @generated
 */
public class DatasetImpl extends EObjectImpl implements Dataset {
	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;

	/**
	 * The default value of the '{@link #getDatasetPath() <em>Dataset Path</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasetPath()
	 * @generated
	 * @ordered
	 */
	protected static final String DATASET_PATH_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getDatasetPath() <em>Dataset Path</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasetPath()
	 * @generated
	 * @ordered
	 */
	protected String datasetPath = DATASET_PATH_EDEFAULT;

	/**
	 * The default value of the '{@link #getDatasetModule() <em>Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasetModule()
	 * @generated
	 * @ordered
	 */
	protected static final String DATASET_MODULE_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getDatasetModule() <em>Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasetModule()
	 * @generated
	 * @ordered
	 */
	protected String datasetModule = DATASET_MODULE_EDEFAULT;

	/**
	 * The default value of the '{@link #getTrainDatasetModule() <em>Train Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTrainDatasetModule()
	 * @generated
	 * @ordered
	 */
	protected static final String TRAIN_DATASET_MODULE_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getTrainDatasetModule() <em>Train Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTrainDatasetModule()
	 * @generated
	 * @ordered
	 */
	protected String trainDatasetModule = TRAIN_DATASET_MODULE_EDEFAULT;

	/**
	 * The default value of the '{@link #getTestDatasetModule() <em>Test Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTestDatasetModule()
	 * @generated
	 * @ordered
	 */
	protected static final String TEST_DATASET_MODULE_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getTestDatasetModule() <em>Test Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTestDatasetModule()
	 * @generated
	 * @ordered
	 */
	protected String testDatasetModule = TEST_DATASET_MODULE_EDEFAULT;

	/**
	 * The default value of the '{@link #getValidationDatasetModule() <em>Validation Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getValidationDatasetModule()
	 * @generated
	 * @ordered
	 */
	protected static final String VALIDATION_DATASET_MODULE_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getValidationDatasetModule() <em>Validation Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getValidationDatasetModule()
	 * @generated
	 * @ordered
	 */
	protected String validationDatasetModule = VALIDATION_DATASET_MODULE_EDEFAULT;

	/**
	 * The cached value of the '{@link #getDatasetModuleParameters() <em>Dataset Module Parameters</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasetModuleParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<String> datasetModuleParameters;

	/**
	 * The cached value of the '{@link #getTrainDatasetModuleParameters() <em>Train Dataset Module Parameters</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTrainDatasetModuleParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<String> trainDatasetModuleParameters;

	/**
	 * The cached value of the '{@link #getTestDatasetModuleParameters() <em>Test Dataset Module Parameters</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTestDatasetModuleParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<String> testDatasetModuleParameters;

	/**
	 * The cached value of the '{@link #getValidationDatasetModuleParameters() <em>Validation Dataset Module Parameters</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getValidationDatasetModuleParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<String> validationDatasetModuleParameters;

	/**
	 * The default value of the '{@link #getPriviledgedGroup() <em>Priviledged Group</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPriviledgedGroup()
	 * @generated
	 * @ordered
	 */
	protected static final int PRIVILEDGED_GROUP_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getPriviledgedGroup() <em>Priviledged Group</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPriviledgedGroup()
	 * @generated
	 * @ordered
	 */
	protected int priviledgedGroup = PRIVILEDGED_GROUP_EDEFAULT;

	/**
	 * The default value of the '{@link #getUnpriviledgedGroup() <em>Unpriviledged Group</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUnpriviledgedGroup()
	 * @generated
	 * @ordered
	 */
	protected static final int UNPRIVILEDGED_GROUP_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getUnpriviledgedGroup() <em>Unpriviledged Group</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUnpriviledgedGroup()
	 * @generated
	 * @ordered
	 */
	protected int unpriviledgedGroup = UNPRIVILEDGED_GROUP_EDEFAULT;

	/**
	 * The default value of the '{@link #getPredictedAttribute() <em>Predicted Attribute</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPredictedAttribute()
	 * @generated
	 * @ordered
	 */
	protected static final String PREDICTED_ATTRIBUTE_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getPredictedAttribute() <em>Predicted Attribute</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPredictedAttribute()
	 * @generated
	 * @ordered
	 */
	protected String predictedAttribute = PREDICTED_ATTRIBUTE_EDEFAULT;

	/**
	 * The cached value of the '{@link #getFavorableClasses() <em>Favorable Classes</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getFavorableClasses()
	 * @generated
	 * @ordered
	 */
	protected EList<Integer> favorableClasses;

	/**
	 * The cached value of the '{@link #getProtectedAttributes() <em>Protected Attributes</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getProtectedAttributes()
	 * @generated
	 * @ordered
	 */
	protected EList<String> protectedAttributes;

	/**
	 * The cached value of the '{@link #getPrivilegedClasses() <em>Privileged Classes</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrivilegedClasses()
	 * @generated
	 * @ordered
	 */
	protected EList<Integer> privilegedClasses;

	/**
	 * The cached value of the '{@link #getUnprivilegedClasses() <em>Unprivileged Classes</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUnprivilegedClasses()
	 * @generated
	 * @ordered
	 */
	protected EList<Integer> unprivilegedClasses;

	/**
	 * The cached value of the '{@link #getInstanceWeights() <em>Instance Weights</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInstanceWeights()
	 * @generated
	 * @ordered
	 */
	protected EList<String> instanceWeights;

	/**
	 * The cached value of the '{@link #getCategoricalFeatures() <em>Categorical Features</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCategoricalFeatures()
	 * @generated
	 * @ordered
	 */
	protected EList<String> categoricalFeatures;

	/**
	 * The cached value of the '{@link #getDroppedAttributes() <em>Dropped Attributes</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDroppedAttributes()
	 * @generated
	 * @ordered
	 */
	protected EList<String> droppedAttributes;

	/**
	 * The cached value of the '{@link #getFeaturesToKeep() <em>Features To Keep</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getFeaturesToKeep()
	 * @generated
	 * @ordered
	 */
	protected EList<String> featuresToKeep;

	/**
	 * The cached value of the '{@link #getNotAvailableValues() <em>Not Available Values</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getNotAvailableValues()
	 * @generated
	 * @ordered
	 */
	protected EList<String> notAvailableValues;

	/**
	 * The default value of the '{@link #getDefaultMappings() <em>Default Mappings</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDefaultMappings()
	 * @generated
	 * @ordered
	 */
	protected static final String DEFAULT_MAPPINGS_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getDefaultMappings() <em>Default Mappings</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDefaultMappings()
	 * @generated
	 * @ordered
	 */
	protected String defaultMappings = DEFAULT_MAPPINGS_EDEFAULT;

	/**
	 * The cached value of the '{@link #getTrainTestValidationSplit() <em>Train Test Validation Split</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTrainTestValidationSplit()
	 * @generated
	 * @ordered
	 */
	protected EList<Float> trainTestValidationSplit;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected DatasetImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return FairmlPackage.Literals.DATASET;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getDatasetPath() {
		return datasetPath;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setDatasetPath(String newDatasetPath) {
		String oldDatasetPath = datasetPath;
		datasetPath = newDatasetPath;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__DATASET_PATH, oldDatasetPath, datasetPath));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getDatasetModule() {
		return datasetModule;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setDatasetModule(String newDatasetModule) {
		String oldDatasetModule = datasetModule;
		datasetModule = newDatasetModule;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__DATASET_MODULE, oldDatasetModule, datasetModule));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getTrainDatasetModule() {
		return trainDatasetModule;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setTrainDatasetModule(String newTrainDatasetModule) {
		String oldTrainDatasetModule = trainDatasetModule;
		trainDatasetModule = newTrainDatasetModule;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__TRAIN_DATASET_MODULE, oldTrainDatasetModule, trainDatasetModule));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getTestDatasetModule() {
		return testDatasetModule;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setTestDatasetModule(String newTestDatasetModule) {
		String oldTestDatasetModule = testDatasetModule;
		testDatasetModule = newTestDatasetModule;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__TEST_DATASET_MODULE, oldTestDatasetModule, testDatasetModule));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getValidationDatasetModule() {
		return validationDatasetModule;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setValidationDatasetModule(String newValidationDatasetModule) {
		String oldValidationDatasetModule = validationDatasetModule;
		validationDatasetModule = newValidationDatasetModule;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__VALIDATION_DATASET_MODULE, oldValidationDatasetModule, validationDatasetModule));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getDatasetModuleParameters() {
		if (datasetModuleParameters == null) {
			datasetModuleParameters = new EDataTypeEList<String>(String.class, this, FairmlPackage.DATASET__DATASET_MODULE_PARAMETERS);
		}
		return datasetModuleParameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getTrainDatasetModuleParameters() {
		if (trainDatasetModuleParameters == null) {
			trainDatasetModuleParameters = new EDataTypeEList<String>(String.class, this, FairmlPackage.DATASET__TRAIN_DATASET_MODULE_PARAMETERS);
		}
		return trainDatasetModuleParameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getTestDatasetModuleParameters() {
		if (testDatasetModuleParameters == null) {
			testDatasetModuleParameters = new EDataTypeEList<String>(String.class, this, FairmlPackage.DATASET__TEST_DATASET_MODULE_PARAMETERS);
		}
		return testDatasetModuleParameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getValidationDatasetModuleParameters() {
		if (validationDatasetModuleParameters == null) {
			validationDatasetModuleParameters = new EDataTypeEList<String>(String.class, this, FairmlPackage.DATASET__VALIDATION_DATASET_MODULE_PARAMETERS);
		}
		return validationDatasetModuleParameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public int getPriviledgedGroup() {
		return priviledgedGroup;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setPriviledgedGroup(int newPriviledgedGroup) {
		int oldPriviledgedGroup = priviledgedGroup;
		priviledgedGroup = newPriviledgedGroup;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__PRIVILEDGED_GROUP, oldPriviledgedGroup, priviledgedGroup));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public int getUnpriviledgedGroup() {
		return unpriviledgedGroup;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setUnpriviledgedGroup(int newUnpriviledgedGroup) {
		int oldUnpriviledgedGroup = unpriviledgedGroup;
		unpriviledgedGroup = newUnpriviledgedGroup;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__UNPRIVILEDGED_GROUP, oldUnpriviledgedGroup, unpriviledgedGroup));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getPredictedAttribute() {
		return predictedAttribute;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setPredictedAttribute(String newPredictedAttribute) {
		String oldPredictedAttribute = predictedAttribute;
		predictedAttribute = newPredictedAttribute;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__PREDICTED_ATTRIBUTE, oldPredictedAttribute, predictedAttribute));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<Integer> getFavorableClasses() {
		if (favorableClasses == null) {
			favorableClasses = new EDataTypeEList<Integer>(Integer.class, this, FairmlPackage.DATASET__FAVORABLE_CLASSES);
		}
		return favorableClasses;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getProtectedAttributes() {
		if (protectedAttributes == null) {
			protectedAttributes = new EDataTypeUniqueEList<String>(String.class, this, FairmlPackage.DATASET__PROTECTED_ATTRIBUTES);
		}
		return protectedAttributes;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<Integer> getPrivilegedClasses() {
		if (privilegedClasses == null) {
			privilegedClasses = new EDataTypeEList<Integer>(Integer.class, this, FairmlPackage.DATASET__PRIVILEGED_CLASSES);
		}
		return privilegedClasses;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<Integer> getUnprivilegedClasses() {
		if (unprivilegedClasses == null) {
			unprivilegedClasses = new EDataTypeEList<Integer>(Integer.class, this, FairmlPackage.DATASET__UNPRIVILEGED_CLASSES);
		}
		return unprivilegedClasses;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getInstanceWeights() {
		if (instanceWeights == null) {
			instanceWeights = new EDataTypeUniqueEList<String>(String.class, this, FairmlPackage.DATASET__INSTANCE_WEIGHTS);
		}
		return instanceWeights;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getCategoricalFeatures() {
		if (categoricalFeatures == null) {
			categoricalFeatures = new EDataTypeUniqueEList<String>(String.class, this, FairmlPackage.DATASET__CATEGORICAL_FEATURES);
		}
		return categoricalFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getDroppedAttributes() {
		if (droppedAttributes == null) {
			droppedAttributes = new EDataTypeUniqueEList<String>(String.class, this, FairmlPackage.DATASET__DROPPED_ATTRIBUTES);
		}
		return droppedAttributes;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getFeaturesToKeep() {
		if (featuresToKeep == null) {
			featuresToKeep = new EDataTypeUniqueEList<String>(String.class, this, FairmlPackage.DATASET__FEATURES_TO_KEEP);
		}
		return featuresToKeep;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getNotAvailableValues() {
		if (notAvailableValues == null) {
			notAvailableValues = new EDataTypeUniqueEList<String>(String.class, this, FairmlPackage.DATASET__NOT_AVAILABLE_VALUES);
		}
		return notAvailableValues;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getDefaultMappings() {
		return defaultMappings;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setDefaultMappings(String newDefaultMappings) {
		String oldDefaultMappings = defaultMappings;
		defaultMappings = newDefaultMappings;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.DATASET__DEFAULT_MAPPINGS, oldDefaultMappings, defaultMappings));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<Float> getTrainTestValidationSplit() {
		if (trainTestValidationSplit == null) {
			trainTestValidationSplit = new EDataTypeEList<Float>(Float.class, this, FairmlPackage.DATASET__TRAIN_TEST_VALIDATION_SPLIT);
		}
		return trainTestValidationSplit;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case FairmlPackage.DATASET__NAME:
				return getName();
			case FairmlPackage.DATASET__DATASET_PATH:
				return getDatasetPath();
			case FairmlPackage.DATASET__DATASET_MODULE:
				return getDatasetModule();
			case FairmlPackage.DATASET__TRAIN_DATASET_MODULE:
				return getTrainDatasetModule();
			case FairmlPackage.DATASET__TEST_DATASET_MODULE:
				return getTestDatasetModule();
			case FairmlPackage.DATASET__VALIDATION_DATASET_MODULE:
				return getValidationDatasetModule();
			case FairmlPackage.DATASET__DATASET_MODULE_PARAMETERS:
				return getDatasetModuleParameters();
			case FairmlPackage.DATASET__TRAIN_DATASET_MODULE_PARAMETERS:
				return getTrainDatasetModuleParameters();
			case FairmlPackage.DATASET__TEST_DATASET_MODULE_PARAMETERS:
				return getTestDatasetModuleParameters();
			case FairmlPackage.DATASET__VALIDATION_DATASET_MODULE_PARAMETERS:
				return getValidationDatasetModuleParameters();
			case FairmlPackage.DATASET__PRIVILEDGED_GROUP:
				return getPriviledgedGroup();
			case FairmlPackage.DATASET__UNPRIVILEDGED_GROUP:
				return getUnpriviledgedGroup();
			case FairmlPackage.DATASET__PREDICTED_ATTRIBUTE:
				return getPredictedAttribute();
			case FairmlPackage.DATASET__FAVORABLE_CLASSES:
				return getFavorableClasses();
			case FairmlPackage.DATASET__PROTECTED_ATTRIBUTES:
				return getProtectedAttributes();
			case FairmlPackage.DATASET__PRIVILEGED_CLASSES:
				return getPrivilegedClasses();
			case FairmlPackage.DATASET__UNPRIVILEGED_CLASSES:
				return getUnprivilegedClasses();
			case FairmlPackage.DATASET__INSTANCE_WEIGHTS:
				return getInstanceWeights();
			case FairmlPackage.DATASET__CATEGORICAL_FEATURES:
				return getCategoricalFeatures();
			case FairmlPackage.DATASET__DROPPED_ATTRIBUTES:
				return getDroppedAttributes();
			case FairmlPackage.DATASET__FEATURES_TO_KEEP:
				return getFeaturesToKeep();
			case FairmlPackage.DATASET__NOT_AVAILABLE_VALUES:
				return getNotAvailableValues();
			case FairmlPackage.DATASET__DEFAULT_MAPPINGS:
				return getDefaultMappings();
			case FairmlPackage.DATASET__TRAIN_TEST_VALIDATION_SPLIT:
				return getTrainTestValidationSplit();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case FairmlPackage.DATASET__NAME:
				setName((String)newValue);
				return;
			case FairmlPackage.DATASET__DATASET_PATH:
				setDatasetPath((String)newValue);
				return;
			case FairmlPackage.DATASET__DATASET_MODULE:
				setDatasetModule((String)newValue);
				return;
			case FairmlPackage.DATASET__TRAIN_DATASET_MODULE:
				setTrainDatasetModule((String)newValue);
				return;
			case FairmlPackage.DATASET__TEST_DATASET_MODULE:
				setTestDatasetModule((String)newValue);
				return;
			case FairmlPackage.DATASET__VALIDATION_DATASET_MODULE:
				setValidationDatasetModule((String)newValue);
				return;
			case FairmlPackage.DATASET__DATASET_MODULE_PARAMETERS:
				getDatasetModuleParameters().clear();
				getDatasetModuleParameters().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__TRAIN_DATASET_MODULE_PARAMETERS:
				getTrainDatasetModuleParameters().clear();
				getTrainDatasetModuleParameters().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__TEST_DATASET_MODULE_PARAMETERS:
				getTestDatasetModuleParameters().clear();
				getTestDatasetModuleParameters().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__VALIDATION_DATASET_MODULE_PARAMETERS:
				getValidationDatasetModuleParameters().clear();
				getValidationDatasetModuleParameters().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__PRIVILEDGED_GROUP:
				setPriviledgedGroup((Integer)newValue);
				return;
			case FairmlPackage.DATASET__UNPRIVILEDGED_GROUP:
				setUnpriviledgedGroup((Integer)newValue);
				return;
			case FairmlPackage.DATASET__PREDICTED_ATTRIBUTE:
				setPredictedAttribute((String)newValue);
				return;
			case FairmlPackage.DATASET__FAVORABLE_CLASSES:
				getFavorableClasses().clear();
				getFavorableClasses().addAll((Collection<? extends Integer>)newValue);
				return;
			case FairmlPackage.DATASET__PROTECTED_ATTRIBUTES:
				getProtectedAttributes().clear();
				getProtectedAttributes().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__PRIVILEGED_CLASSES:
				getPrivilegedClasses().clear();
				getPrivilegedClasses().addAll((Collection<? extends Integer>)newValue);
				return;
			case FairmlPackage.DATASET__UNPRIVILEGED_CLASSES:
				getUnprivilegedClasses().clear();
				getUnprivilegedClasses().addAll((Collection<? extends Integer>)newValue);
				return;
			case FairmlPackage.DATASET__INSTANCE_WEIGHTS:
				getInstanceWeights().clear();
				getInstanceWeights().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__CATEGORICAL_FEATURES:
				getCategoricalFeatures().clear();
				getCategoricalFeatures().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__DROPPED_ATTRIBUTES:
				getDroppedAttributes().clear();
				getDroppedAttributes().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__FEATURES_TO_KEEP:
				getFeaturesToKeep().clear();
				getFeaturesToKeep().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__NOT_AVAILABLE_VALUES:
				getNotAvailableValues().clear();
				getNotAvailableValues().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.DATASET__DEFAULT_MAPPINGS:
				setDefaultMappings((String)newValue);
				return;
			case FairmlPackage.DATASET__TRAIN_TEST_VALIDATION_SPLIT:
				getTrainTestValidationSplit().clear();
				getTrainTestValidationSplit().addAll((Collection<? extends Float>)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case FairmlPackage.DATASET__NAME:
				setName(NAME_EDEFAULT);
				return;
			case FairmlPackage.DATASET__DATASET_PATH:
				setDatasetPath(DATASET_PATH_EDEFAULT);
				return;
			case FairmlPackage.DATASET__DATASET_MODULE:
				setDatasetModule(DATASET_MODULE_EDEFAULT);
				return;
			case FairmlPackage.DATASET__TRAIN_DATASET_MODULE:
				setTrainDatasetModule(TRAIN_DATASET_MODULE_EDEFAULT);
				return;
			case FairmlPackage.DATASET__TEST_DATASET_MODULE:
				setTestDatasetModule(TEST_DATASET_MODULE_EDEFAULT);
				return;
			case FairmlPackage.DATASET__VALIDATION_DATASET_MODULE:
				setValidationDatasetModule(VALIDATION_DATASET_MODULE_EDEFAULT);
				return;
			case FairmlPackage.DATASET__DATASET_MODULE_PARAMETERS:
				getDatasetModuleParameters().clear();
				return;
			case FairmlPackage.DATASET__TRAIN_DATASET_MODULE_PARAMETERS:
				getTrainDatasetModuleParameters().clear();
				return;
			case FairmlPackage.DATASET__TEST_DATASET_MODULE_PARAMETERS:
				getTestDatasetModuleParameters().clear();
				return;
			case FairmlPackage.DATASET__VALIDATION_DATASET_MODULE_PARAMETERS:
				getValidationDatasetModuleParameters().clear();
				return;
			case FairmlPackage.DATASET__PRIVILEDGED_GROUP:
				setPriviledgedGroup(PRIVILEDGED_GROUP_EDEFAULT);
				return;
			case FairmlPackage.DATASET__UNPRIVILEDGED_GROUP:
				setUnpriviledgedGroup(UNPRIVILEDGED_GROUP_EDEFAULT);
				return;
			case FairmlPackage.DATASET__PREDICTED_ATTRIBUTE:
				setPredictedAttribute(PREDICTED_ATTRIBUTE_EDEFAULT);
				return;
			case FairmlPackage.DATASET__FAVORABLE_CLASSES:
				getFavorableClasses().clear();
				return;
			case FairmlPackage.DATASET__PROTECTED_ATTRIBUTES:
				getProtectedAttributes().clear();
				return;
			case FairmlPackage.DATASET__PRIVILEGED_CLASSES:
				getPrivilegedClasses().clear();
				return;
			case FairmlPackage.DATASET__UNPRIVILEGED_CLASSES:
				getUnprivilegedClasses().clear();
				return;
			case FairmlPackage.DATASET__INSTANCE_WEIGHTS:
				getInstanceWeights().clear();
				return;
			case FairmlPackage.DATASET__CATEGORICAL_FEATURES:
				getCategoricalFeatures().clear();
				return;
			case FairmlPackage.DATASET__DROPPED_ATTRIBUTES:
				getDroppedAttributes().clear();
				return;
			case FairmlPackage.DATASET__FEATURES_TO_KEEP:
				getFeaturesToKeep().clear();
				return;
			case FairmlPackage.DATASET__NOT_AVAILABLE_VALUES:
				getNotAvailableValues().clear();
				return;
			case FairmlPackage.DATASET__DEFAULT_MAPPINGS:
				setDefaultMappings(DEFAULT_MAPPINGS_EDEFAULT);
				return;
			case FairmlPackage.DATASET__TRAIN_TEST_VALIDATION_SPLIT:
				getTrainTestValidationSplit().clear();
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case FairmlPackage.DATASET__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case FairmlPackage.DATASET__DATASET_PATH:
				return DATASET_PATH_EDEFAULT == null ? datasetPath != null : !DATASET_PATH_EDEFAULT.equals(datasetPath);
			case FairmlPackage.DATASET__DATASET_MODULE:
				return DATASET_MODULE_EDEFAULT == null ? datasetModule != null : !DATASET_MODULE_EDEFAULT.equals(datasetModule);
			case FairmlPackage.DATASET__TRAIN_DATASET_MODULE:
				return TRAIN_DATASET_MODULE_EDEFAULT == null ? trainDatasetModule != null : !TRAIN_DATASET_MODULE_EDEFAULT.equals(trainDatasetModule);
			case FairmlPackage.DATASET__TEST_DATASET_MODULE:
				return TEST_DATASET_MODULE_EDEFAULT == null ? testDatasetModule != null : !TEST_DATASET_MODULE_EDEFAULT.equals(testDatasetModule);
			case FairmlPackage.DATASET__VALIDATION_DATASET_MODULE:
				return VALIDATION_DATASET_MODULE_EDEFAULT == null ? validationDatasetModule != null : !VALIDATION_DATASET_MODULE_EDEFAULT.equals(validationDatasetModule);
			case FairmlPackage.DATASET__DATASET_MODULE_PARAMETERS:
				return datasetModuleParameters != null && !datasetModuleParameters.isEmpty();
			case FairmlPackage.DATASET__TRAIN_DATASET_MODULE_PARAMETERS:
				return trainDatasetModuleParameters != null && !trainDatasetModuleParameters.isEmpty();
			case FairmlPackage.DATASET__TEST_DATASET_MODULE_PARAMETERS:
				return testDatasetModuleParameters != null && !testDatasetModuleParameters.isEmpty();
			case FairmlPackage.DATASET__VALIDATION_DATASET_MODULE_PARAMETERS:
				return validationDatasetModuleParameters != null && !validationDatasetModuleParameters.isEmpty();
			case FairmlPackage.DATASET__PRIVILEDGED_GROUP:
				return priviledgedGroup != PRIVILEDGED_GROUP_EDEFAULT;
			case FairmlPackage.DATASET__UNPRIVILEDGED_GROUP:
				return unpriviledgedGroup != UNPRIVILEDGED_GROUP_EDEFAULT;
			case FairmlPackage.DATASET__PREDICTED_ATTRIBUTE:
				return PREDICTED_ATTRIBUTE_EDEFAULT == null ? predictedAttribute != null : !PREDICTED_ATTRIBUTE_EDEFAULT.equals(predictedAttribute);
			case FairmlPackage.DATASET__FAVORABLE_CLASSES:
				return favorableClasses != null && !favorableClasses.isEmpty();
			case FairmlPackage.DATASET__PROTECTED_ATTRIBUTES:
				return protectedAttributes != null && !protectedAttributes.isEmpty();
			case FairmlPackage.DATASET__PRIVILEGED_CLASSES:
				return privilegedClasses != null && !privilegedClasses.isEmpty();
			case FairmlPackage.DATASET__UNPRIVILEGED_CLASSES:
				return unprivilegedClasses != null && !unprivilegedClasses.isEmpty();
			case FairmlPackage.DATASET__INSTANCE_WEIGHTS:
				return instanceWeights != null && !instanceWeights.isEmpty();
			case FairmlPackage.DATASET__CATEGORICAL_FEATURES:
				return categoricalFeatures != null && !categoricalFeatures.isEmpty();
			case FairmlPackage.DATASET__DROPPED_ATTRIBUTES:
				return droppedAttributes != null && !droppedAttributes.isEmpty();
			case FairmlPackage.DATASET__FEATURES_TO_KEEP:
				return featuresToKeep != null && !featuresToKeep.isEmpty();
			case FairmlPackage.DATASET__NOT_AVAILABLE_VALUES:
				return notAvailableValues != null && !notAvailableValues.isEmpty();
			case FairmlPackage.DATASET__DEFAULT_MAPPINGS:
				return DEFAULT_MAPPINGS_EDEFAULT == null ? defaultMappings != null : !DEFAULT_MAPPINGS_EDEFAULT.equals(defaultMappings);
			case FairmlPackage.DATASET__TRAIN_TEST_VALIDATION_SPLIT:
				return trainTestValidationSplit != null && !trainTestValidationSplit.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuilder result = new StringBuilder(super.toString());
		result.append(" (name: ");
		result.append(name);
		result.append(", datasetPath: ");
		result.append(datasetPath);
		result.append(", datasetModule: ");
		result.append(datasetModule);
		result.append(", trainDatasetModule: ");
		result.append(trainDatasetModule);
		result.append(", testDatasetModule: ");
		result.append(testDatasetModule);
		result.append(", validationDatasetModule: ");
		result.append(validationDatasetModule);
		result.append(", datasetModuleParameters: ");
		result.append(datasetModuleParameters);
		result.append(", trainDatasetModuleParameters: ");
		result.append(trainDatasetModuleParameters);
		result.append(", testDatasetModuleParameters: ");
		result.append(testDatasetModuleParameters);
		result.append(", validationDatasetModuleParameters: ");
		result.append(validationDatasetModuleParameters);
		result.append(", priviledgedGroup: ");
		result.append(priviledgedGroup);
		result.append(", unpriviledgedGroup: ");
		result.append(unpriviledgedGroup);
		result.append(", predictedAttribute: ");
		result.append(predictedAttribute);
		result.append(", favorableClasses: ");
		result.append(favorableClasses);
		result.append(", protectedAttributes: ");
		result.append(protectedAttributes);
		result.append(", privilegedClasses: ");
		result.append(privilegedClasses);
		result.append(", unprivilegedClasses: ");
		result.append(unprivilegedClasses);
		result.append(", instanceWeights: ");
		result.append(instanceWeights);
		result.append(", categoricalFeatures: ");
		result.append(categoricalFeatures);
		result.append(", droppedAttributes: ");
		result.append(droppedAttributes);
		result.append(", featuresToKeep: ");
		result.append(featuresToKeep);
		result.append(", notAvailableValues: ");
		result.append(notAvailableValues);
		result.append(", defaultMappings: ");
		result.append(defaultMappings);
		result.append(", trainTestValidationSplit: ");
		result.append(trainTestValidationSplit);
		result.append(')');
		return result.toString();
	}

} //DatasetImpl
