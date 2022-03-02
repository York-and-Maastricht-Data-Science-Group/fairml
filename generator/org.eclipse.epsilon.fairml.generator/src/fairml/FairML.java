/**
 */
package fairml;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Fair ML</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link fairml.FairML#getName <em>Name</em>}</li>
 *   <li>{@link fairml.FairML#getDescription <em>Description</em>}</li>
 *   <li>{@link fairml.FairML#getFilename <em>Filename</em>}</li>
 *   <li>{@link fairml.FairML#getDatasets <em>Datasets</em>}</li>
 *   <li>{@link fairml.FairML#getBiasMitigations <em>Bias Mitigations</em>}</li>
 * </ul>
 *
 * @see fairml.FairmlPackage#getFairML()
 * @model
 * @generated
 */
public interface FairML extends EObject {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see fairml.FairmlPackage#getFairML_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link fairml.FairML#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Description</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Description</em>' attribute.
	 * @see #setDescription(String)
	 * @see fairml.FairmlPackage#getFairML_Description()
	 * @model
	 * @generated
	 */
	String getDescription();

	/**
	 * Sets the value of the '{@link fairml.FairML#getDescription <em>Description</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Description</em>' attribute.
	 * @see #getDescription()
	 * @generated
	 */
	void setDescription(String value);

	/**
	 * Returns the value of the '<em><b>Filename</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Filename</em>' attribute.
	 * @see #setFilename(String)
	 * @see fairml.FairmlPackage#getFairML_Filename()
	 * @model
	 * @generated
	 */
	String getFilename();

	/**
	 * Sets the value of the '{@link fairml.FairML#getFilename <em>Filename</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Filename</em>' attribute.
	 * @see #getFilename()
	 * @generated
	 */
	void setFilename(String value);

	/**
	 * Returns the value of the '<em><b>Datasets</b></em>' containment reference list.
	 * The list contents are of type {@link fairml.Dataset}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Datasets</em>' containment reference list.
	 * @see fairml.FairmlPackage#getFairML_Datasets()
	 * @model containment="true"
	 * @generated
	 */
	EList<Dataset> getDatasets();

	/**
	 * Returns the value of the '<em><b>Bias Mitigations</b></em>' containment reference list.
	 * The list contents are of type {@link fairml.BiasMitigation}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Bias Mitigations</em>' containment reference list.
	 * @see fairml.FairmlPackage#getFairML_BiasMitigations()
	 * @model containment="true"
	 * @generated
	 */
	EList<BiasMitigation> getBiasMitigations();

} // FairML
