/**
 */
package fairml.impl;

import fairml.BiasMetric;
import fairml.BiasMitigation;
import fairml.Dataset;
import fairml.FairmlPackage;
import fairml.MitigationMethod;
import fairml.TrainingMethod;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.EObjectResolvingEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Bias Mitigation</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link fairml.impl.BiasMitigationImpl#getName <em>Name</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isGroupFairness <em>Group Fairness</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isIndividualFairness <em>Individual Fairness</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isGroupIndividualSingleMetric <em>Group Individual Single Metric</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isEqualFairness <em>Equal Fairness</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isProportionalFairness <em>Proportional Fairness</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isCheckFalsePositive <em>Check False Positive</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isCheckFalseNegative <em>Check False Negative</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isCheckErrorRate <em>Check Error Rate</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isCheckEqualBenefit <em>Check Equal Benefit</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isPrepreprocessingMitigation <em>Prepreprocessing Mitigation</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isModifiableWeight <em>Modifiable Weight</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isAllowLatentSpace <em>Allow Latent Space</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isInpreprocessingMitigation <em>Inpreprocessing Mitigation</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isAllowRegularisation <em>Allow Regularisation</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isPostpreprocessingMitigation <em>Postpreprocessing Mitigation</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#isAllowRandomisation <em>Allow Randomisation</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#getDatasets <em>Datasets</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#getBiasMetrics <em>Bias Metrics</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#getMitigationMethods <em>Mitigation Methods</em>}</li>
 *   <li>{@link fairml.impl.BiasMitigationImpl#getTrainingMethods <em>Training Methods</em>}</li>
 * </ul>
 *
 * @generated
 */
public class BiasMitigationImpl extends EObjectImpl implements BiasMitigation {
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
	 * The default value of the '{@link #isGroupFairness() <em>Group Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isGroupFairness()
	 * @generated
	 * @ordered
	 */
	protected static final boolean GROUP_FAIRNESS_EDEFAULT = true;

	/**
	 * The cached value of the '{@link #isGroupFairness() <em>Group Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isGroupFairness()
	 * @generated
	 * @ordered
	 */
	protected boolean groupFairness = GROUP_FAIRNESS_EDEFAULT;

	/**
	 * The default value of the '{@link #isIndividualFairness() <em>Individual Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isIndividualFairness()
	 * @generated
	 * @ordered
	 */
	protected static final boolean INDIVIDUAL_FAIRNESS_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isIndividualFairness() <em>Individual Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isIndividualFairness()
	 * @generated
	 * @ordered
	 */
	protected boolean individualFairness = INDIVIDUAL_FAIRNESS_EDEFAULT;

	/**
	 * The default value of the '{@link #isGroupIndividualSingleMetric() <em>Group Individual Single Metric</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isGroupIndividualSingleMetric()
	 * @generated
	 * @ordered
	 */
	protected static final boolean GROUP_INDIVIDUAL_SINGLE_METRIC_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isGroupIndividualSingleMetric() <em>Group Individual Single Metric</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isGroupIndividualSingleMetric()
	 * @generated
	 * @ordered
	 */
	protected boolean groupIndividualSingleMetric = GROUP_INDIVIDUAL_SINGLE_METRIC_EDEFAULT;

	/**
	 * The default value of the '{@link #isEqualFairness() <em>Equal Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isEqualFairness()
	 * @generated
	 * @ordered
	 */
	protected static final boolean EQUAL_FAIRNESS_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isEqualFairness() <em>Equal Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isEqualFairness()
	 * @generated
	 * @ordered
	 */
	protected boolean equalFairness = EQUAL_FAIRNESS_EDEFAULT;

	/**
	 * The default value of the '{@link #isProportionalFairness() <em>Proportional Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isProportionalFairness()
	 * @generated
	 * @ordered
	 */
	protected static final boolean PROPORTIONAL_FAIRNESS_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isProportionalFairness() <em>Proportional Fairness</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isProportionalFairness()
	 * @generated
	 * @ordered
	 */
	protected boolean proportionalFairness = PROPORTIONAL_FAIRNESS_EDEFAULT;

	/**
	 * The default value of the '{@link #isCheckFalsePositive() <em>Check False Positive</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isCheckFalsePositive()
	 * @generated
	 * @ordered
	 */
	protected static final boolean CHECK_FALSE_POSITIVE_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isCheckFalsePositive() <em>Check False Positive</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isCheckFalsePositive()
	 * @generated
	 * @ordered
	 */
	protected boolean checkFalsePositive = CHECK_FALSE_POSITIVE_EDEFAULT;

	/**
	 * The default value of the '{@link #isCheckFalseNegative() <em>Check False Negative</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isCheckFalseNegative()
	 * @generated
	 * @ordered
	 */
	protected static final boolean CHECK_FALSE_NEGATIVE_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isCheckFalseNegative() <em>Check False Negative</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isCheckFalseNegative()
	 * @generated
	 * @ordered
	 */
	protected boolean checkFalseNegative = CHECK_FALSE_NEGATIVE_EDEFAULT;

	/**
	 * The default value of the '{@link #isCheckErrorRate() <em>Check Error Rate</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isCheckErrorRate()
	 * @generated
	 * @ordered
	 */
	protected static final boolean CHECK_ERROR_RATE_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isCheckErrorRate() <em>Check Error Rate</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isCheckErrorRate()
	 * @generated
	 * @ordered
	 */
	protected boolean checkErrorRate = CHECK_ERROR_RATE_EDEFAULT;

	/**
	 * The default value of the '{@link #isCheckEqualBenefit() <em>Check Equal Benefit</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isCheckEqualBenefit()
	 * @generated
	 * @ordered
	 */
	protected static final boolean CHECK_EQUAL_BENEFIT_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isCheckEqualBenefit() <em>Check Equal Benefit</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isCheckEqualBenefit()
	 * @generated
	 * @ordered
	 */
	protected boolean checkEqualBenefit = CHECK_EQUAL_BENEFIT_EDEFAULT;

	/**
	 * The default value of the '{@link #isPrepreprocessingMitigation() <em>Prepreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isPrepreprocessingMitigation()
	 * @generated
	 * @ordered
	 */
	protected static final boolean PREPREPROCESSING_MITIGATION_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isPrepreprocessingMitigation() <em>Prepreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isPrepreprocessingMitigation()
	 * @generated
	 * @ordered
	 */
	protected boolean prepreprocessingMitigation = PREPREPROCESSING_MITIGATION_EDEFAULT;

	/**
	 * The default value of the '{@link #isModifiableWeight() <em>Modifiable Weight</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isModifiableWeight()
	 * @generated
	 * @ordered
	 */
	protected static final boolean MODIFIABLE_WEIGHT_EDEFAULT = true;

	/**
	 * The cached value of the '{@link #isModifiableWeight() <em>Modifiable Weight</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isModifiableWeight()
	 * @generated
	 * @ordered
	 */
	protected boolean modifiableWeight = MODIFIABLE_WEIGHT_EDEFAULT;

	/**
	 * The default value of the '{@link #isAllowLatentSpace() <em>Allow Latent Space</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isAllowLatentSpace()
	 * @generated
	 * @ordered
	 */
	protected static final boolean ALLOW_LATENT_SPACE_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isAllowLatentSpace() <em>Allow Latent Space</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isAllowLatentSpace()
	 * @generated
	 * @ordered
	 */
	protected boolean allowLatentSpace = ALLOW_LATENT_SPACE_EDEFAULT;

	/**
	 * The default value of the '{@link #isInpreprocessingMitigation() <em>Inpreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isInpreprocessingMitigation()
	 * @generated
	 * @ordered
	 */
	protected static final boolean INPREPROCESSING_MITIGATION_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isInpreprocessingMitigation() <em>Inpreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isInpreprocessingMitigation()
	 * @generated
	 * @ordered
	 */
	protected boolean inpreprocessingMitigation = INPREPROCESSING_MITIGATION_EDEFAULT;

	/**
	 * The default value of the '{@link #isAllowRegularisation() <em>Allow Regularisation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isAllowRegularisation()
	 * @generated
	 * @ordered
	 */
	protected static final boolean ALLOW_REGULARISATION_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isAllowRegularisation() <em>Allow Regularisation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isAllowRegularisation()
	 * @generated
	 * @ordered
	 */
	protected boolean allowRegularisation = ALLOW_REGULARISATION_EDEFAULT;

	/**
	 * The default value of the '{@link #isPostpreprocessingMitigation() <em>Postpreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isPostpreprocessingMitigation()
	 * @generated
	 * @ordered
	 */
	protected static final boolean POSTPREPROCESSING_MITIGATION_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isPostpreprocessingMitigation() <em>Postpreprocessing Mitigation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isPostpreprocessingMitigation()
	 * @generated
	 * @ordered
	 */
	protected boolean postpreprocessingMitigation = POSTPREPROCESSING_MITIGATION_EDEFAULT;

	/**
	 * The default value of the '{@link #isAllowRandomisation() <em>Allow Randomisation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isAllowRandomisation()
	 * @generated
	 * @ordered
	 */
	protected static final boolean ALLOW_RANDOMISATION_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isAllowRandomisation() <em>Allow Randomisation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isAllowRandomisation()
	 * @generated
	 * @ordered
	 */
	protected boolean allowRandomisation = ALLOW_RANDOMISATION_EDEFAULT;

	/**
	 * The cached value of the '{@link #getDatasets() <em>Datasets</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasets()
	 * @generated
	 * @ordered
	 */
	protected EList<Dataset> datasets;

	/**
	 * The cached value of the '{@link #getBiasMetrics() <em>Bias Metrics</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getBiasMetrics()
	 * @generated
	 * @ordered
	 */
	protected EList<BiasMetric> biasMetrics;

	/**
	 * The cached value of the '{@link #getMitigationMethods() <em>Mitigation Methods</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMitigationMethods()
	 * @generated
	 * @ordered
	 */
	protected EList<MitigationMethod> mitigationMethods;

	/**
	 * The cached value of the '{@link #getTrainingMethods() <em>Training Methods</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTrainingMethods()
	 * @generated
	 * @ordered
	 */
	protected EList<TrainingMethod> trainingMethods;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected BiasMitigationImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return FairmlPackage.Literals.BIAS_MITIGATION;
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
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isGroupFairness() {
		return groupFairness;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setGroupFairness(boolean newGroupFairness) {
		boolean oldGroupFairness = groupFairness;
		groupFairness = newGroupFairness;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__GROUP_FAIRNESS, oldGroupFairness, groupFairness));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isIndividualFairness() {
		return individualFairness;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setIndividualFairness(boolean newIndividualFairness) {
		boolean oldIndividualFairness = individualFairness;
		individualFairness = newIndividualFairness;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__INDIVIDUAL_FAIRNESS, oldIndividualFairness, individualFairness));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isGroupIndividualSingleMetric() {
		return groupIndividualSingleMetric;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setGroupIndividualSingleMetric(boolean newGroupIndividualSingleMetric) {
		boolean oldGroupIndividualSingleMetric = groupIndividualSingleMetric;
		groupIndividualSingleMetric = newGroupIndividualSingleMetric;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__GROUP_INDIVIDUAL_SINGLE_METRIC, oldGroupIndividualSingleMetric, groupIndividualSingleMetric));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isEqualFairness() {
		return equalFairness;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setEqualFairness(boolean newEqualFairness) {
		boolean oldEqualFairness = equalFairness;
		equalFairness = newEqualFairness;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__EQUAL_FAIRNESS, oldEqualFairness, equalFairness));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isProportionalFairness() {
		return proportionalFairness;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setProportionalFairness(boolean newProportionalFairness) {
		boolean oldProportionalFairness = proportionalFairness;
		proportionalFairness = newProportionalFairness;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__PROPORTIONAL_FAIRNESS, oldProportionalFairness, proportionalFairness));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isCheckFalsePositive() {
		return checkFalsePositive;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setCheckFalsePositive(boolean newCheckFalsePositive) {
		boolean oldCheckFalsePositive = checkFalsePositive;
		checkFalsePositive = newCheckFalsePositive;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_POSITIVE, oldCheckFalsePositive, checkFalsePositive));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isCheckFalseNegative() {
		return checkFalseNegative;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setCheckFalseNegative(boolean newCheckFalseNegative) {
		boolean oldCheckFalseNegative = checkFalseNegative;
		checkFalseNegative = newCheckFalseNegative;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_NEGATIVE, oldCheckFalseNegative, checkFalseNegative));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isCheckErrorRate() {
		return checkErrorRate;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setCheckErrorRate(boolean newCheckErrorRate) {
		boolean oldCheckErrorRate = checkErrorRate;
		checkErrorRate = newCheckErrorRate;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__CHECK_ERROR_RATE, oldCheckErrorRate, checkErrorRate));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isCheckEqualBenefit() {
		return checkEqualBenefit;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setCheckEqualBenefit(boolean newCheckEqualBenefit) {
		boolean oldCheckEqualBenefit = checkEqualBenefit;
		checkEqualBenefit = newCheckEqualBenefit;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__CHECK_EQUAL_BENEFIT, oldCheckEqualBenefit, checkEqualBenefit));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isPrepreprocessingMitigation() {
		return prepreprocessingMitigation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setPrepreprocessingMitigation(boolean newPrepreprocessingMitigation) {
		boolean oldPrepreprocessingMitigation = prepreprocessingMitigation;
		prepreprocessingMitigation = newPrepreprocessingMitigation;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__PREPREPROCESSING_MITIGATION, oldPrepreprocessingMitigation, prepreprocessingMitigation));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isModifiableWeight() {
		return modifiableWeight;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setModifiableWeight(boolean newModifiableWeight) {
		boolean oldModifiableWeight = modifiableWeight;
		modifiableWeight = newModifiableWeight;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__MODIFIABLE_WEIGHT, oldModifiableWeight, modifiableWeight));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isAllowLatentSpace() {
		return allowLatentSpace;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setAllowLatentSpace(boolean newAllowLatentSpace) {
		boolean oldAllowLatentSpace = allowLatentSpace;
		allowLatentSpace = newAllowLatentSpace;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__ALLOW_LATENT_SPACE, oldAllowLatentSpace, allowLatentSpace));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isInpreprocessingMitigation() {
		return inpreprocessingMitigation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setInpreprocessingMitigation(boolean newInpreprocessingMitigation) {
		boolean oldInpreprocessingMitigation = inpreprocessingMitigation;
		inpreprocessingMitigation = newInpreprocessingMitigation;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__INPREPROCESSING_MITIGATION, oldInpreprocessingMitigation, inpreprocessingMitigation));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isAllowRegularisation() {
		return allowRegularisation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setAllowRegularisation(boolean newAllowRegularisation) {
		boolean oldAllowRegularisation = allowRegularisation;
		allowRegularisation = newAllowRegularisation;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__ALLOW_REGULARISATION, oldAllowRegularisation, allowRegularisation));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isPostpreprocessingMitigation() {
		return postpreprocessingMitigation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setPostpreprocessingMitigation(boolean newPostpreprocessingMitigation) {
		boolean oldPostpreprocessingMitigation = postpreprocessingMitigation;
		postpreprocessingMitigation = newPostpreprocessingMitigation;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__POSTPREPROCESSING_MITIGATION, oldPostpreprocessingMitigation, postpreprocessingMitigation));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isAllowRandomisation() {
		return allowRandomisation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setAllowRandomisation(boolean newAllowRandomisation) {
		boolean oldAllowRandomisation = allowRandomisation;
		allowRandomisation = newAllowRandomisation;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_MITIGATION__ALLOW_RANDOMISATION, oldAllowRandomisation, allowRandomisation));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<Dataset> getDatasets() {
		if (datasets == null) {
			datasets = new EObjectResolvingEList<Dataset>(Dataset.class, this, FairmlPackage.BIAS_MITIGATION__DATASETS);
		}
		return datasets;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<BiasMetric> getBiasMetrics() {
		if (biasMetrics == null) {
			biasMetrics = new EObjectContainmentEList<BiasMetric>(BiasMetric.class, this, FairmlPackage.BIAS_MITIGATION__BIAS_METRICS);
		}
		return biasMetrics;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<MitigationMethod> getMitigationMethods() {
		if (mitigationMethods == null) {
			mitigationMethods = new EObjectContainmentEList<MitigationMethod>(MitigationMethod.class, this, FairmlPackage.BIAS_MITIGATION__MITIGATION_METHODS);
		}
		return mitigationMethods;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<TrainingMethod> getTrainingMethods() {
		if (trainingMethods == null) {
			trainingMethods = new EObjectContainmentEList<TrainingMethod>(TrainingMethod.class, this, FairmlPackage.BIAS_MITIGATION__TRAINING_METHODS);
		}
		return trainingMethods;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case FairmlPackage.BIAS_MITIGATION__BIAS_METRICS:
				return ((InternalEList<?>)getBiasMetrics()).basicRemove(otherEnd, msgs);
			case FairmlPackage.BIAS_MITIGATION__MITIGATION_METHODS:
				return ((InternalEList<?>)getMitigationMethods()).basicRemove(otherEnd, msgs);
			case FairmlPackage.BIAS_MITIGATION__TRAINING_METHODS:
				return ((InternalEList<?>)getTrainingMethods()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case FairmlPackage.BIAS_MITIGATION__NAME:
				return getName();
			case FairmlPackage.BIAS_MITIGATION__GROUP_FAIRNESS:
				return isGroupFairness();
			case FairmlPackage.BIAS_MITIGATION__INDIVIDUAL_FAIRNESS:
				return isIndividualFairness();
			case FairmlPackage.BIAS_MITIGATION__GROUP_INDIVIDUAL_SINGLE_METRIC:
				return isGroupIndividualSingleMetric();
			case FairmlPackage.BIAS_MITIGATION__EQUAL_FAIRNESS:
				return isEqualFairness();
			case FairmlPackage.BIAS_MITIGATION__PROPORTIONAL_FAIRNESS:
				return isProportionalFairness();
			case FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_POSITIVE:
				return isCheckFalsePositive();
			case FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_NEGATIVE:
				return isCheckFalseNegative();
			case FairmlPackage.BIAS_MITIGATION__CHECK_ERROR_RATE:
				return isCheckErrorRate();
			case FairmlPackage.BIAS_MITIGATION__CHECK_EQUAL_BENEFIT:
				return isCheckEqualBenefit();
			case FairmlPackage.BIAS_MITIGATION__PREPREPROCESSING_MITIGATION:
				return isPrepreprocessingMitigation();
			case FairmlPackage.BIAS_MITIGATION__MODIFIABLE_WEIGHT:
				return isModifiableWeight();
			case FairmlPackage.BIAS_MITIGATION__ALLOW_LATENT_SPACE:
				return isAllowLatentSpace();
			case FairmlPackage.BIAS_MITIGATION__INPREPROCESSING_MITIGATION:
				return isInpreprocessingMitigation();
			case FairmlPackage.BIAS_MITIGATION__ALLOW_REGULARISATION:
				return isAllowRegularisation();
			case FairmlPackage.BIAS_MITIGATION__POSTPREPROCESSING_MITIGATION:
				return isPostpreprocessingMitigation();
			case FairmlPackage.BIAS_MITIGATION__ALLOW_RANDOMISATION:
				return isAllowRandomisation();
			case FairmlPackage.BIAS_MITIGATION__DATASETS:
				return getDatasets();
			case FairmlPackage.BIAS_MITIGATION__BIAS_METRICS:
				return getBiasMetrics();
			case FairmlPackage.BIAS_MITIGATION__MITIGATION_METHODS:
				return getMitigationMethods();
			case FairmlPackage.BIAS_MITIGATION__TRAINING_METHODS:
				return getTrainingMethods();
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
			case FairmlPackage.BIAS_MITIGATION__NAME:
				setName((String)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__GROUP_FAIRNESS:
				setGroupFairness((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__INDIVIDUAL_FAIRNESS:
				setIndividualFairness((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__GROUP_INDIVIDUAL_SINGLE_METRIC:
				setGroupIndividualSingleMetric((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__EQUAL_FAIRNESS:
				setEqualFairness((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__PROPORTIONAL_FAIRNESS:
				setProportionalFairness((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_POSITIVE:
				setCheckFalsePositive((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_NEGATIVE:
				setCheckFalseNegative((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__CHECK_ERROR_RATE:
				setCheckErrorRate((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__CHECK_EQUAL_BENEFIT:
				setCheckEqualBenefit((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__PREPREPROCESSING_MITIGATION:
				setPrepreprocessingMitigation((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__MODIFIABLE_WEIGHT:
				setModifiableWeight((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_LATENT_SPACE:
				setAllowLatentSpace((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__INPREPROCESSING_MITIGATION:
				setInpreprocessingMitigation((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_REGULARISATION:
				setAllowRegularisation((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__POSTPREPROCESSING_MITIGATION:
				setPostpreprocessingMitigation((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_RANDOMISATION:
				setAllowRandomisation((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__DATASETS:
				getDatasets().clear();
				getDatasets().addAll((Collection<? extends Dataset>)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__BIAS_METRICS:
				getBiasMetrics().clear();
				getBiasMetrics().addAll((Collection<? extends BiasMetric>)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__MITIGATION_METHODS:
				getMitigationMethods().clear();
				getMitigationMethods().addAll((Collection<? extends MitigationMethod>)newValue);
				return;
			case FairmlPackage.BIAS_MITIGATION__TRAINING_METHODS:
				getTrainingMethods().clear();
				getTrainingMethods().addAll((Collection<? extends TrainingMethod>)newValue);
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
			case FairmlPackage.BIAS_MITIGATION__NAME:
				setName(NAME_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__GROUP_FAIRNESS:
				setGroupFairness(GROUP_FAIRNESS_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__INDIVIDUAL_FAIRNESS:
				setIndividualFairness(INDIVIDUAL_FAIRNESS_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__GROUP_INDIVIDUAL_SINGLE_METRIC:
				setGroupIndividualSingleMetric(GROUP_INDIVIDUAL_SINGLE_METRIC_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__EQUAL_FAIRNESS:
				setEqualFairness(EQUAL_FAIRNESS_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__PROPORTIONAL_FAIRNESS:
				setProportionalFairness(PROPORTIONAL_FAIRNESS_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_POSITIVE:
				setCheckFalsePositive(CHECK_FALSE_POSITIVE_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_NEGATIVE:
				setCheckFalseNegative(CHECK_FALSE_NEGATIVE_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__CHECK_ERROR_RATE:
				setCheckErrorRate(CHECK_ERROR_RATE_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__CHECK_EQUAL_BENEFIT:
				setCheckEqualBenefit(CHECK_EQUAL_BENEFIT_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__PREPREPROCESSING_MITIGATION:
				setPrepreprocessingMitigation(PREPREPROCESSING_MITIGATION_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__MODIFIABLE_WEIGHT:
				setModifiableWeight(MODIFIABLE_WEIGHT_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_LATENT_SPACE:
				setAllowLatentSpace(ALLOW_LATENT_SPACE_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__INPREPROCESSING_MITIGATION:
				setInpreprocessingMitigation(INPREPROCESSING_MITIGATION_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_REGULARISATION:
				setAllowRegularisation(ALLOW_REGULARISATION_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__POSTPREPROCESSING_MITIGATION:
				setPostpreprocessingMitigation(POSTPREPROCESSING_MITIGATION_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_RANDOMISATION:
				setAllowRandomisation(ALLOW_RANDOMISATION_EDEFAULT);
				return;
			case FairmlPackage.BIAS_MITIGATION__DATASETS:
				getDatasets().clear();
				return;
			case FairmlPackage.BIAS_MITIGATION__BIAS_METRICS:
				getBiasMetrics().clear();
				return;
			case FairmlPackage.BIAS_MITIGATION__MITIGATION_METHODS:
				getMitigationMethods().clear();
				return;
			case FairmlPackage.BIAS_MITIGATION__TRAINING_METHODS:
				getTrainingMethods().clear();
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
			case FairmlPackage.BIAS_MITIGATION__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case FairmlPackage.BIAS_MITIGATION__GROUP_FAIRNESS:
				return groupFairness != GROUP_FAIRNESS_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__INDIVIDUAL_FAIRNESS:
				return individualFairness != INDIVIDUAL_FAIRNESS_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__GROUP_INDIVIDUAL_SINGLE_METRIC:
				return groupIndividualSingleMetric != GROUP_INDIVIDUAL_SINGLE_METRIC_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__EQUAL_FAIRNESS:
				return equalFairness != EQUAL_FAIRNESS_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__PROPORTIONAL_FAIRNESS:
				return proportionalFairness != PROPORTIONAL_FAIRNESS_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_POSITIVE:
				return checkFalsePositive != CHECK_FALSE_POSITIVE_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__CHECK_FALSE_NEGATIVE:
				return checkFalseNegative != CHECK_FALSE_NEGATIVE_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__CHECK_ERROR_RATE:
				return checkErrorRate != CHECK_ERROR_RATE_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__CHECK_EQUAL_BENEFIT:
				return checkEqualBenefit != CHECK_EQUAL_BENEFIT_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__PREPREPROCESSING_MITIGATION:
				return prepreprocessingMitigation != PREPREPROCESSING_MITIGATION_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__MODIFIABLE_WEIGHT:
				return modifiableWeight != MODIFIABLE_WEIGHT_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_LATENT_SPACE:
				return allowLatentSpace != ALLOW_LATENT_SPACE_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__INPREPROCESSING_MITIGATION:
				return inpreprocessingMitigation != INPREPROCESSING_MITIGATION_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_REGULARISATION:
				return allowRegularisation != ALLOW_REGULARISATION_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__POSTPREPROCESSING_MITIGATION:
				return postpreprocessingMitigation != POSTPREPROCESSING_MITIGATION_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__ALLOW_RANDOMISATION:
				return allowRandomisation != ALLOW_RANDOMISATION_EDEFAULT;
			case FairmlPackage.BIAS_MITIGATION__DATASETS:
				return datasets != null && !datasets.isEmpty();
			case FairmlPackage.BIAS_MITIGATION__BIAS_METRICS:
				return biasMetrics != null && !biasMetrics.isEmpty();
			case FairmlPackage.BIAS_MITIGATION__MITIGATION_METHODS:
				return mitigationMethods != null && !mitigationMethods.isEmpty();
			case FairmlPackage.BIAS_MITIGATION__TRAINING_METHODS:
				return trainingMethods != null && !trainingMethods.isEmpty();
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
		result.append(", groupFairness: ");
		result.append(groupFairness);
		result.append(", individualFairness: ");
		result.append(individualFairness);
		result.append(", groupIndividualSingleMetric: ");
		result.append(groupIndividualSingleMetric);
		result.append(", equalFairness: ");
		result.append(equalFairness);
		result.append(", proportionalFairness: ");
		result.append(proportionalFairness);
		result.append(", checkFalsePositive: ");
		result.append(checkFalsePositive);
		result.append(", checkFalseNegative: ");
		result.append(checkFalseNegative);
		result.append(", checkErrorRate: ");
		result.append(checkErrorRate);
		result.append(", checkEqualBenefit: ");
		result.append(checkEqualBenefit);
		result.append(", prepreprocessingMitigation: ");
		result.append(prepreprocessingMitigation);
		result.append(", modifiableWeight: ");
		result.append(modifiableWeight);
		result.append(", allowLatentSpace: ");
		result.append(allowLatentSpace);
		result.append(", inpreprocessingMitigation: ");
		result.append(inpreprocessingMitigation);
		result.append(", allowRegularisation: ");
		result.append(allowRegularisation);
		result.append(", postpreprocessingMitigation: ");
		result.append(postpreprocessingMitigation);
		result.append(", allowRandomisation: ");
		result.append(allowRandomisation);
		result.append(')');
		return result.toString();
	}

} //BiasMitigationImpl
