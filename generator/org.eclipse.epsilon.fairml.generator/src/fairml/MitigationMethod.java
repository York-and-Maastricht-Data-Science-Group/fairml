/**
 */
package fairml;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Mitigation Method</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link fairml.MitigationMethod#getAlgorithm <em>Algorithm</em>}</li>
 * </ul>
 *
 * @see fairml.FairmlPackage#getMitigationMethod()
 * @model
 * @generated
 */
public interface MitigationMethod extends Operation {
	/**
	 * Returns the value of the '<em><b>Algorithm</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Algorithm</em>' attribute.
	 * @see #setAlgorithm(String)
	 * @see fairml.FairmlPackage#getMitigationMethod_Algorithm()
	 * @model
	 * @generated
	 */
	String getAlgorithm();

	/**
	 * Sets the value of the '{@link fairml.MitigationMethod#getAlgorithm <em>Algorithm</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Algorithm</em>' attribute.
	 * @see #getAlgorithm()
	 * @generated
	 */
	void setAlgorithm(String value);

} // MitigationMethod
