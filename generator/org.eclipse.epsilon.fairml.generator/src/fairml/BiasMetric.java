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
 *   <li>{@link fairml.BiasMetric#getDatasetType <em>Dataset Type</em>}</li>
 *   <li>{@link fairml.BiasMetric#isOptimalThreshold <em>Optimal Threshold</em>}</li>
 *   <li>{@link fairml.BiasMetric#isPlotThreshold <em>Plot Threshold</em>}</li>
 * </ul>
 *
 * @see fairml.FairmlPackage#getBiasMetric()
 * @model
 * @generated
 */
public interface BiasMetric extends Operation {
	/**
	 * Returns the value of the '<em><b>Class Name</b></em>' attribute.
	 * The default value is <code>"FairMLMetric"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Class Name</em>' attribute.
	 * @see #setClassName(String)
	 * @see fairml.FairmlPackage#getBiasMetric_ClassName()
	 * @model default="FairMLMetric"
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

	/**
	 * Returns the value of the '<em><b>Dataset Type</b></em>' attribute.
	 * The default value is <code>"test"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Dataset Type</em>' attribute.
	 * @see #setDatasetType(String)
	 * @see fairml.FairmlPackage#getBiasMetric_DatasetType()
	 * @model default="test"
	 * @generated
	 */
	String getDatasetType();

	/**
	 * Sets the value of the '{@link fairml.BiasMetric#getDatasetType <em>Dataset Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Dataset Type</em>' attribute.
	 * @see #getDatasetType()
	 * @generated
	 */
	void setDatasetType(String value);

	/**
	 * Returns the value of the '<em><b>Optimal Threshold</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Optimal Threshold</em>' attribute.
	 * @see #setOptimalThreshold(boolean)
	 * @see fairml.FairmlPackage#getBiasMetric_OptimalThreshold()
	 * @model default="false"
	 * @generated
	 */
	boolean isOptimalThreshold();

	/**
	 * Sets the value of the '{@link fairml.BiasMetric#isOptimalThreshold <em>Optimal Threshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Optimal Threshold</em>' attribute.
	 * @see #isOptimalThreshold()
	 * @generated
	 */
	void setOptimalThreshold(boolean value);

	/**
	 * Returns the value of the '<em><b>Plot Threshold</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Plot Threshold</em>' attribute.
	 * @see #setPlotThreshold(boolean)
	 * @see fairml.FairmlPackage#getBiasMetric_PlotThreshold()
	 * @model default="false"
	 * @generated
	 */
	boolean isPlotThreshold();

	/**
	 * Sets the value of the '{@link fairml.BiasMetric#isPlotThreshold <em>Plot Threshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Plot Threshold</em>' attribute.
	 * @see #isPlotThreshold()
	 * @generated
	 */
	void setPlotThreshold(boolean value);

} // BiasMetric
