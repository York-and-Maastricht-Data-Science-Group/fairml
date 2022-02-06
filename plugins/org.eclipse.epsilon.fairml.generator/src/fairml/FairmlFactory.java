/**
 */
package fairml;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see fairml.FairmlPackage
 * @generated
 */
public interface FairmlFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	FairmlFactory eINSTANCE = fairml.impl.FairmlFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Fair ML</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Fair ML</em>'.
	 * @generated
	 */
	FairML createFairML();

	/**
	 * Returns a new object of class '<em>Function</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Function</em>'.
	 * @generated
	 */
	Function createFunction();

	/**
	 * Returns a new object of class '<em>Training Method</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Training Method</em>'.
	 * @generated
	 */
	TrainingMethod createTrainingMethod();

	/**
	 * Returns a new object of class '<em>Mitigation Method</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Mitigation Method</em>'.
	 * @generated
	 */
	MitigationMethod createMitigationMethod();

	/**
	 * Returns a new object of class '<em>Bias Metric</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Bias Metric</em>'.
	 * @generated
	 */
	BiasMetric createBiasMetric();

	/**
	 * Returns a new object of class '<em>Dataset</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Dataset</em>'.
	 * @generated
	 */
	Dataset createDataset();

	/**
	 * Returns a new object of class '<em>Bias Mitigation</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Bias Mitigation</em>'.
	 * @generated
	 */
	BiasMitigation createBiasMitigation();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	FairmlPackage getFairmlPackage();

} //FairmlFactory
