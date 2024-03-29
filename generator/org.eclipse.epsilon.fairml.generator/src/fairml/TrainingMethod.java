/**
 */
package fairml;

import org.eclipse.emf.common.util.EList;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Training Method</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link fairml.TrainingMethod#getAlgorithm <em>Algorithm</em>}</li>
 *   <li>{@link fairml.TrainingMethod#getFitParameters <em>Fit Parameters</em>}</li>
 *   <li>{@link fairml.TrainingMethod#getPredictParameters <em>Predict Parameters</em>}</li>
 *   <li>{@link fairml.TrainingMethod#isWithoutWeight <em>Without Weight</em>}</li>
 * </ul>
 *
 * @see fairml.FairmlPackage#getTrainingMethod()
 * @model
 * @generated
 */
public interface TrainingMethod extends Operation {
	/**
	 * Returns the value of the '<em><b>Algorithm</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Algorithm</em>' attribute.
	 * @see #setAlgorithm(String)
	 * @see fairml.FairmlPackage#getTrainingMethod_Algorithm()
	 * @model
	 * @generated
	 */
	String getAlgorithm();

	/**
	 * Sets the value of the '{@link fairml.TrainingMethod#getAlgorithm <em>Algorithm</em>}' attribute.
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
	 * @see fairml.FairmlPackage#getTrainingMethod_FitParameters()
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
	 * @see fairml.FairmlPackage#getTrainingMethod_PredictParameters()
	 * @model unique="false"
	 * @generated
	 */
	EList<String> getPredictParameters();

	/**
	 * Returns the value of the '<em><b>Without Weight</b></em>' attribute.
	 * The default value is <code>"true"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Without Weight</em>' attribute.
	 * @see #setWithoutWeight(boolean)
	 * @see fairml.FairmlPackage#getTrainingMethod_WithoutWeight()
	 * @model default="true"
	 * @generated
	 */
	boolean isWithoutWeight();

	/**
	 * Sets the value of the '{@link fairml.TrainingMethod#isWithoutWeight <em>Without Weight</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Without Weight</em>' attribute.
	 * @see #isWithoutWeight()
	 * @generated
	 */
	void setWithoutWeight(boolean value);

} // TrainingMethod
