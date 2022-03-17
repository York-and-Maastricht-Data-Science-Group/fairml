/**
 */
package fairml;

import org.eclipse.emf.common.util.EList;


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
 *   <li>{@link fairml.MitigationMethod#getFitParameters <em>Fit Parameters</em>}</li>
 *   <li>{@link fairml.MitigationMethod#getPredictParameters <em>Predict Parameters</em>}</li>
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

	/**
	 * Returns the value of the '<em><b>Fit Parameters</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Fit Parameters</em>' attribute list.
	 * @see fairml.FairmlPackage#getMitigationMethod_FitParameters()
	 * @model unique="false"
	 * @generated
	 */
	EList<String> getFitParameters();

	/**
	 * Returns the value of the '<em><b>Predict Parameters</b></em>' attribute list.
	 * The list contents are of type {@link java.lang.String}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Predict Parameters</em>' attribute list.
	 * @see fairml.FairmlPackage#getMitigationMethod_PredictParameters()
	 * @model unique="false"
	 * @generated
	 */
	EList<String> getPredictParameters();

} // MitigationMethod
