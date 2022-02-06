/**
 */
package fairml;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Bias Mitigation</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link fairml.BiasMitigation#getName <em>Name</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isGroupFairness <em>Group Fairness</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isIndividualFairness <em>Individual Fairness</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isGroupIndividualSingleMetric <em>Group Individual Single Metric</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isEqualFairness <em>Equal Fairness</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isProportionalFairness <em>Proportional Fairness</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isCheckFalsePositive <em>Check False Positive</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isCheckFalseNegative <em>Check False Negative</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isCheckErrorRate <em>Check Error Rate</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isCheckEqualBenefit <em>Check Equal Benefit</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isPrepreprocessingMitigation <em>Prepreprocessing Mitigation</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isModifiableWeight <em>Modifiable Weight</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isAllowLatentSpace <em>Allow Latent Space</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isInpreprocessingMitigation <em>Inpreprocessing Mitigation</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isAllowRegularisation <em>Allow Regularisation</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isPostpreprocessingMitigation <em>Postpreprocessing Mitigation</em>}</li>
 *   <li>{@link fairml.BiasMitigation#isAllowRandomisation <em>Allow Randomisation</em>}</li>
 *   <li>{@link fairml.BiasMitigation#getDatasets <em>Datasets</em>}</li>
 *   <li>{@link fairml.BiasMitigation#getBiasMetrics <em>Bias Metrics</em>}</li>
 *   <li>{@link fairml.BiasMitigation#getMitigationMethods <em>Mitigation Methods</em>}</li>
 *   <li>{@link fairml.BiasMitigation#getTrainingMethods <em>Training Methods</em>}</li>
 * </ul>
 *
 * @see fairml.FairmlPackage#getBiasMitigation()
 * @model
 * @generated
 */
public interface BiasMitigation extends EObject {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see fairml.FairmlPackage#getBiasMitigation_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Group Fairness</b></em>' attribute.
	 * The default value is <code>"true"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Group Fairness</em>' attribute.
	 * @see #setGroupFairness(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_GroupFairness()
	 * @model default="true"
	 * @generated
	 */
	boolean isGroupFairness();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isGroupFairness <em>Group Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Group Fairness</em>' attribute.
	 * @see #isGroupFairness()
	 * @generated
	 */
	void setGroupFairness(boolean value);

	/**
	 * Returns the value of the '<em><b>Individual Fairness</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Individual Fairness</em>' attribute.
	 * @see #setIndividualFairness(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_IndividualFairness()
	 * @model default="false"
	 * @generated
	 */
	boolean isIndividualFairness();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isIndividualFairness <em>Individual Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Individual Fairness</em>' attribute.
	 * @see #isIndividualFairness()
	 * @generated
	 */
	void setIndividualFairness(boolean value);

	/**
	 * Returns the value of the '<em><b>Group Individual Single Metric</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Group Individual Single Metric</em>' attribute.
	 * @see #setGroupIndividualSingleMetric(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_GroupIndividualSingleMetric()
	 * @model default="false"
	 * @generated
	 */
	boolean isGroupIndividualSingleMetric();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isGroupIndividualSingleMetric <em>Group Individual Single Metric</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Group Individual Single Metric</em>' attribute.
	 * @see #isGroupIndividualSingleMetric()
	 * @generated
	 */
	void setGroupIndividualSingleMetric(boolean value);

	/**
	 * Returns the value of the '<em><b>Equal Fairness</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Equal Fairness</em>' attribute.
	 * @see #setEqualFairness(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_EqualFairness()
	 * @model default="false"
	 * @generated
	 */
	boolean isEqualFairness();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isEqualFairness <em>Equal Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Equal Fairness</em>' attribute.
	 * @see #isEqualFairness()
	 * @generated
	 */
	void setEqualFairness(boolean value);

	/**
	 * Returns the value of the '<em><b>Proportional Fairness</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Proportional Fairness</em>' attribute.
	 * @see #setProportionalFairness(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_ProportionalFairness()
	 * @model default="false"
	 * @generated
	 */
	boolean isProportionalFairness();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isProportionalFairness <em>Proportional Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Proportional Fairness</em>' attribute.
	 * @see #isProportionalFairness()
	 * @generated
	 */
	void setProportionalFairness(boolean value);

	/**
	 * Returns the value of the '<em><b>Check False Positive</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Check False Positive</em>' attribute.
	 * @see #setCheckFalsePositive(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_CheckFalsePositive()
	 * @model default="false"
	 * @generated
	 */
	boolean isCheckFalsePositive();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isCheckFalsePositive <em>Check False Positive</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Check False Positive</em>' attribute.
	 * @see #isCheckFalsePositive()
	 * @generated
	 */
	void setCheckFalsePositive(boolean value);

	/**
	 * Returns the value of the '<em><b>Check False Negative</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Check False Negative</em>' attribute.
	 * @see #setCheckFalseNegative(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_CheckFalseNegative()
	 * @model default="false"
	 * @generated
	 */
	boolean isCheckFalseNegative();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isCheckFalseNegative <em>Check False Negative</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Check False Negative</em>' attribute.
	 * @see #isCheckFalseNegative()
	 * @generated
	 */
	void setCheckFalseNegative(boolean value);

	/**
	 * Returns the value of the '<em><b>Check Error Rate</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Check Error Rate</em>' attribute.
	 * @see #setCheckErrorRate(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_CheckErrorRate()
	 * @model default="false"
	 * @generated
	 */
	boolean isCheckErrorRate();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isCheckErrorRate <em>Check Error Rate</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Check Error Rate</em>' attribute.
	 * @see #isCheckErrorRate()
	 * @generated
	 */
	void setCheckErrorRate(boolean value);

	/**
	 * Returns the value of the '<em><b>Check Equal Benefit</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Check Equal Benefit</em>' attribute.
	 * @see #setCheckEqualBenefit(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_CheckEqualBenefit()
	 * @model default="false"
	 * @generated
	 */
	boolean isCheckEqualBenefit();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isCheckEqualBenefit <em>Check Equal Benefit</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Check Equal Benefit</em>' attribute.
	 * @see #isCheckEqualBenefit()
	 * @generated
	 */
	void setCheckEqualBenefit(boolean value);

	/**
	 * Returns the value of the '<em><b>Prepreprocessing Mitigation</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Prepreprocessing Mitigation</em>' attribute.
	 * @see #setPrepreprocessingMitigation(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_PrepreprocessingMitigation()
	 * @model default="false"
	 * @generated
	 */
	boolean isPrepreprocessingMitigation();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isPrepreprocessingMitigation <em>Prepreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Prepreprocessing Mitigation</em>' attribute.
	 * @see #isPrepreprocessingMitigation()
	 * @generated
	 */
	void setPrepreprocessingMitigation(boolean value);

	/**
	 * Returns the value of the '<em><b>Modifiable Weight</b></em>' attribute.
	 * The default value is <code>"true"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Modifiable Weight</em>' attribute.
	 * @see #setModifiableWeight(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_ModifiableWeight()
	 * @model default="true"
	 * @generated
	 */
	boolean isModifiableWeight();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isModifiableWeight <em>Modifiable Weight</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Modifiable Weight</em>' attribute.
	 * @see #isModifiableWeight()
	 * @generated
	 */
	void setModifiableWeight(boolean value);

	/**
	 * Returns the value of the '<em><b>Allow Latent Space</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Allow Latent Space</em>' attribute.
	 * @see #setAllowLatentSpace(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_AllowLatentSpace()
	 * @model default="false"
	 * @generated
	 */
	boolean isAllowLatentSpace();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isAllowLatentSpace <em>Allow Latent Space</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Allow Latent Space</em>' attribute.
	 * @see #isAllowLatentSpace()
	 * @generated
	 */
	void setAllowLatentSpace(boolean value);

	/**
	 * Returns the value of the '<em><b>Inpreprocessing Mitigation</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Inpreprocessing Mitigation</em>' attribute.
	 * @see #setInpreprocessingMitigation(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_InpreprocessingMitigation()
	 * @model default="false"
	 * @generated
	 */
	boolean isInpreprocessingMitigation();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isInpreprocessingMitigation <em>Inpreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Inpreprocessing Mitigation</em>' attribute.
	 * @see #isInpreprocessingMitigation()
	 * @generated
	 */
	void setInpreprocessingMitigation(boolean value);

	/**
	 * Returns the value of the '<em><b>Allow Regularisation</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Allow Regularisation</em>' attribute.
	 * @see #setAllowRegularisation(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_AllowRegularisation()
	 * @model default="false"
	 * @generated
	 */
	boolean isAllowRegularisation();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isAllowRegularisation <em>Allow Regularisation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Allow Regularisation</em>' attribute.
	 * @see #isAllowRegularisation()
	 * @generated
	 */
	void setAllowRegularisation(boolean value);

	/**
	 * Returns the value of the '<em><b>Postpreprocessing Mitigation</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Postpreprocessing Mitigation</em>' attribute.
	 * @see #setPostpreprocessingMitigation(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_PostpreprocessingMitigation()
	 * @model default="false"
	 * @generated
	 */
	boolean isPostpreprocessingMitigation();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isPostpreprocessingMitigation <em>Postpreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Postpreprocessing Mitigation</em>' attribute.
	 * @see #isPostpreprocessingMitigation()
	 * @generated
	 */
	void setPostpreprocessingMitigation(boolean value);

	/**
	 * Returns the value of the '<em><b>Allow Randomisation</b></em>' attribute.
	 * The default value is <code>"false"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Allow Randomisation</em>' attribute.
	 * @see #setAllowRandomisation(boolean)
	 * @see fairml.FairmlPackage#getBiasMitigation_AllowRandomisation()
	 * @model default="false"
	 * @generated
	 */
	boolean isAllowRandomisation();

	/**
	 * Sets the value of the '{@link fairml.BiasMitigation#isAllowRandomisation <em>Allow Randomisation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Allow Randomisation</em>' attribute.
	 * @see #isAllowRandomisation()
	 * @generated
	 */
	void setAllowRandomisation(boolean value);

	/**
	 * Returns the value of the '<em><b>Datasets</b></em>' reference list.
	 * The list contents are of type {@link fairml.Dataset}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Datasets</em>' reference list.
	 * @see fairml.FairmlPackage#getBiasMitigation_Datasets()
	 * @model
	 * @generated
	 */
	EList<Dataset> getDatasets();

	/**
	 * Returns the value of the '<em><b>Bias Metrics</b></em>' containment reference list.
	 * The list contents are of type {@link fairml.BiasMetric}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Bias Metrics</em>' containment reference list.
	 * @see fairml.FairmlPackage#getBiasMitigation_BiasMetrics()
	 * @model containment="true"
	 * @generated
	 */
	EList<BiasMetric> getBiasMetrics();

	/**
	 * Returns the value of the '<em><b>Mitigation Methods</b></em>' containment reference list.
	 * The list contents are of type {@link fairml.MitigationMethod}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Mitigation Methods</em>' containment reference list.
	 * @see fairml.FairmlPackage#getBiasMitigation_MitigationMethods()
	 * @model containment="true"
	 * @generated
	 */
	EList<MitigationMethod> getMitigationMethods();

	/**
	 * Returns the value of the '<em><b>Training Methods</b></em>' containment reference list.
	 * The list contents are of type {@link fairml.TrainingMethod}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Training Methods</em>' containment reference list.
	 * @see fairml.FairmlPackage#getBiasMitigation_TrainingMethods()
	 * @model containment="true"
	 * @generated
	 */
	EList<TrainingMethod> getTrainingMethods();

} // BiasMitigation
