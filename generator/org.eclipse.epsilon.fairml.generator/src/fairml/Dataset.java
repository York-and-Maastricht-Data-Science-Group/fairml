/**
 */
package fairml;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Dataset</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link fairml.Dataset#getName <em>Name</em>}</li>
 *   <li>{@link fairml.Dataset#getDatasetPath <em>Dataset Path</em>}</li>
 *   <li>{@link fairml.Dataset#getDatasetModule <em>Dataset Module</em>}</li>
 *   <li>{@link fairml.Dataset#getTrainDatasetModule <em>Train Dataset Module</em>}</li>
 *   <li>{@link fairml.Dataset#getTestDatasetModule <em>Test Dataset Module</em>}</li>
 *   <li>{@link fairml.Dataset#getValidationDatasetModule <em>Validation Dataset Module</em>}</li>
 *   <li>{@link fairml.Dataset#getDatasetModuleParameters <em>Dataset Module Parameters</em>}</li>
 *   <li>{@link fairml.Dataset#getTrainDatasetModuleParameters <em>Train Dataset Module Parameters</em>}</li>
 *   <li>{@link fairml.Dataset#getTestDatasetModuleParameters <em>Test Dataset Module Parameters</em>}</li>
 *   <li>{@link fairml.Dataset#getValidationDatasetModuleParameters <em>Validation Dataset Module Parameters</em>}</li>
 *   <li>{@link fairml.Dataset#getPriviledgedGroup <em>Priviledged Group</em>}</li>
 *   <li>{@link fairml.Dataset#getUnpriviledgedGroup <em>Unpriviledged Group</em>}</li>
 *   <li>{@link fairml.Dataset#getPredictedAttribute <em>Predicted Attribute</em>}</li>
 *   <li>{@link fairml.Dataset#getFavorableClasses <em>Favorable Classes</em>}</li>
 *   <li>{@link fairml.Dataset#getProtectedAttributes <em>Protected Attributes</em>}</li>
 *   <li>{@link fairml.Dataset#getPrivilegedClasses <em>Privileged Classes</em>}</li>
 *   <li>{@link fairml.Dataset#getUnprivilegedClasses <em>Unprivileged Classes</em>}</li>
 *   <li>{@link fairml.Dataset#getInstanceWeights <em>Instance Weights</em>}</li>
 *   <li>{@link fairml.Dataset#getCategoricalFeatures <em>Categorical Features</em>}</li>
 *   <li>{@link fairml.Dataset#getDroppedAttributes <em>Dropped Attributes</em>}</li>
 *   <li>{@link fairml.Dataset#getFeaturesToKeep <em>Features To Keep</em>}</li>
 *   <li>{@link fairml.Dataset#getNotAvailableValues <em>Not Available Values</em>}</li>
 *   <li>{@link fairml.Dataset#getDefaultMappings <em>Default Mappings</em>}</li>
 *   <li>{@link fairml.Dataset#getTrainTestValidationSplit <em>Train Test Validation Split</em>}</li>
 * </ul>
 *
 * @see fairml.FairmlPackage#getDataset()
 * @model
 * @generated
 */
public interface Dataset extends EObject {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see fairml.FairmlPackage#getDataset_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Dataset Path</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Dataset Path</em>' attribute.
	 * @see #setDatasetPath(String)
	 * @see fairml.FairmlPackage#getDataset_DatasetPath()
	 * @model
	 * @generated
	 */
	String getDatasetPath();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getDatasetPath <em>Dataset Path</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Dataset Path</em>' attribute.
	 * @see #getDatasetPath()
	 * @generated
	 */
	void setDatasetPath(String value);

	/**
	 * Returns the value of the '<em><b>Dataset Module</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Dataset Module</em>' attribute.
	 * @see #setDatasetModule(String)
	 * @see fairml.FairmlPackage#getDataset_DatasetModule()
	 * @model
	 * @generated
	 */
	String getDatasetModule();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getDatasetModule <em>Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Dataset Module</em>' attribute.
	 * @see #getDatasetModule()
	 * @generated
	 */
	void setDatasetModule(String value);

	/**
	 * Returns the value of the '<em><b>Train Dataset Module</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Train Dataset Module</em>' attribute.
	 * @see #setTrainDatasetModule(String)
	 * @see fairml.FairmlPackage#getDataset_TrainDatasetModule()
	 * @model
	 * @generated
	 */
	String getTrainDatasetModule();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getTrainDatasetModule <em>Train Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Train Dataset Module</em>' attribute.
	 * @see #getTrainDatasetModule()
	 * @generated
	 */
	void setTrainDatasetModule(String value);

	/**
	 * Returns the value of the '<em><b>Test Dataset Module</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Test Dataset Module</em>' attribute.
	 * @see #setTestDatasetModule(String)
	 * @see fairml.FairmlPackage#getDataset_TestDatasetModule()
	 * @model
	 * @generated
	 */
	String getTestDatasetModule();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getTestDatasetModule <em>Test Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Test Dataset Module</em>' attribute.
	 * @see #getTestDatasetModule()
	 * @generated
	 */
	void setTestDatasetModule(String value);

	/**
	 * Returns the value of the '<em><b>Validation Dataset Module</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Validation Dataset Module</em>' attribute.
	 * @see #setValidationDatasetModule(String)
	 * @see fairml.FairmlPackage#getDataset_ValidationDatasetModule()
	 * @model
	 * @generated
	 */
	String getValidationDatasetModule();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getValidationDatasetModule <em>Validation Dataset Module</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Validation Dataset Module</em>' attribute.
	 * @see #getValidationDatasetModule()
	 * @generated
	 */
	void setValidationDatasetModule(String value);

	/**
	 * Returns the value of the '<em><b>Dataset Module Parameters</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Dataset Module Parameters</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_DatasetModuleParameters()
	 * @model unique="false"
	 * @generated
	 */
	EList<String> getDatasetModuleParameters();

	/**
	 * Returns the value of the '<em><b>Train Dataset Module Parameters</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Train Dataset Module Parameters</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_TrainDatasetModuleParameters()
	 * @model unique="false"
	 * @generated
	 */
	EList<String> getTrainDatasetModuleParameters();

	/**
	 * Returns the value of the '<em><b>Test Dataset Module Parameters</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Test Dataset Module Parameters</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_TestDatasetModuleParameters()
	 * @model unique="false"
	 * @generated
	 */
	EList<String> getTestDatasetModuleParameters();

	/**
	 * Returns the value of the '<em><b>Validation Dataset Module Parameters</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Validation Dataset Module Parameters</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_ValidationDatasetModuleParameters()
	 * @model unique="false"
	 * @generated
	 */
	EList<String> getValidationDatasetModuleParameters();

	/**
	 * Returns the value of the '<em><b>Priviledged Group</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Priviledged Group</em>' attribute.
	 * @see #setPriviledgedGroup(int)
	 * @see fairml.FairmlPackage#getDataset_PriviledgedGroup()
	 * @model
	 * @generated
	 */
	int getPriviledgedGroup();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getPriviledgedGroup <em>Priviledged Group</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Priviledged Group</em>' attribute.
	 * @see #getPriviledgedGroup()
	 * @generated
	 */
	void setPriviledgedGroup(int value);

	/**
	 * Returns the value of the '<em><b>Unpriviledged Group</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Unpriviledged Group</em>' attribute.
	 * @see #setUnpriviledgedGroup(int)
	 * @see fairml.FairmlPackage#getDataset_UnpriviledgedGroup()
	 * @model
	 * @generated
	 */
	int getUnpriviledgedGroup();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getUnpriviledgedGroup <em>Unpriviledged Group</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Unpriviledged Group</em>' attribute.
	 * @see #getUnpriviledgedGroup()
	 * @generated
	 */
	void setUnpriviledgedGroup(int value);

	/**
	 * Returns the value of the '<em><b>Predicted Attribute</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Predicted Attribute</em>' attribute.
	 * @see #setPredictedAttribute(String)
	 * @see fairml.FairmlPackage#getDataset_PredictedAttribute()
	 * @model
	 * @generated
	 */
	String getPredictedAttribute();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getPredictedAttribute <em>Predicted Attribute</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Predicted Attribute</em>' attribute.
	 * @see #getPredictedAttribute()
	 * @generated
	 */
	void setPredictedAttribute(String value);

	/**
	 * Returns the value of the '<em><b>Favorable Classes</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.Integer}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Favorable Classes</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_FavorableClasses()
	 * @model unique="false"
	 * @generated
	 */
	EList<Integer> getFavorableClasses();

	/**
	 * Returns the value of the '<em><b>Protected Attributes</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Protected Attributes</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_ProtectedAttributes()
	 * @model
	 * @generated
	 */
	EList<String> getProtectedAttributes();

	/**
	 * Returns the value of the '<em><b>Privileged Classes</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.Integer}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Privileged Classes</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_PrivilegedClasses()
	 * @model unique="false"
	 * @generated
	 */
	EList<Integer> getPrivilegedClasses();

	/**
	 * Returns the value of the '<em><b>Unprivileged Classes</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.Integer}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Unprivileged Classes</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_UnprivilegedClasses()
	 * @model unique="false"
	 * @generated
	 */
	EList<Integer> getUnprivilegedClasses();

	/**
	 * Returns the value of the '<em><b>Instance Weights</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Instance Weights</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_InstanceWeights()
	 * @model
	 * @generated
	 */
	EList<String> getInstanceWeights();

	/**
	 * Returns the value of the '<em><b>Categorical Features</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Categorical Features</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_CategoricalFeatures()
	 * @model
	 * @generated
	 */
	EList<String> getCategoricalFeatures();

	/**
	 * Returns the value of the '<em><b>Dropped Attributes</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Dropped Attributes</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_DroppedAttributes()
	 * @model
	 * @generated
	 */
	EList<String> getDroppedAttributes();

	/**
	 * Returns the value of the '<em><b>Features To Keep</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Features To Keep</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_FeaturesToKeep()
	 * @model
	 * @generated
	 */
	EList<String> getFeaturesToKeep();

	/**
	 * Returns the value of the '<em><b>Not Available Values</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Not Available Values</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_NotAvailableValues()
	 * @model
	 * @generated
	 */
	EList<String> getNotAvailableValues();

	/**
	 * Returns the value of the '<em><b>Default Mappings</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Default Mappings</em>' attribute.
	 * @see #setDefaultMappings(String)
	 * @see fairml.FairmlPackage#getDataset_DefaultMappings()
	 * @model
	 * @generated
	 */
	String getDefaultMappings();

	/**
	 * Sets the value of the '{@link fairml.Dataset#getDefaultMappings <em>Default Mappings</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Default Mappings</em>' attribute.
	 * @see #getDefaultMappings()
	 * @generated
	 */
	void setDefaultMappings(String value);

	/**
	 * Returns the value of the '<em><b>Train Test Validation Split</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.Float}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Train Test Validation Split</em>' attribute list.
	 * @see fairml.FairmlPackage#getDataset_TrainTestValidationSplit()
	 * @model unique="false" upper="3"
	 * @generated
	 */
	EList<Float> getTrainTestValidationSplit();

} // Dataset
