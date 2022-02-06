/**
 */
package fairml;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Bias Metric</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link fairml.BiasMetric#getClassName <em>Class Name</em>}</li>
 * </ul>
 *
 * @see fairml.FairmlPackage#getBiasMetric()
 * @model
 * @generated
 */
public interface BiasMetric extends Operation {
	/**
	 * Returns the value of the '<em><b>Class Name</b></em>' attribute.
	 * The default value is <code>"ClassificationMetric"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Class Name</em>' attribute.
	 * @see #setClassName(String)
	 * @see fairml.FairmlPackage#getBiasMetric_ClassName()
	 * @model default="ClassificationMetric"
	 * @generated
	 */
	String getClassName();

	/**
	 * Sets the value of the '{@link fairml.BiasMetric#getClassName <em>Class Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Class Name</em>' attribute.
	 * @see #getClassName()
	 * @generated
	 */
	void setClassName(String value);

} // BiasMetric
