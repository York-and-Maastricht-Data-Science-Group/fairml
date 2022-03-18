/**
 */
package fairml.impl;

import fairml.FairmlPackage;
import fairml.TrainingMethod;

import java.util.Collection;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.util.EDataTypeEList;
import org.eclipse.emf.ecore.util.EDataTypeUniqueEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Training Method</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link fairml.impl.TrainingMethodImpl#getAlgorithm <em>Algorithm</em>}</li>
 *   <li>{@link fairml.impl.TrainingMethodImpl#getFitParameters <em>Fit Parameters</em>}</li>
 *   <li>{@link fairml.impl.TrainingMethodImpl#getPredictParameters <em>Predict Parameters</em>}</li>
 *   <li>{@link fairml.impl.TrainingMethodImpl#isWithoutWeight <em>Without Weight</em>}</li>
 * </ul>
 *
 * @generated
 */
public class TrainingMethodImpl extends OperationImpl implements TrainingMethod {
	/**
	 * The default value of the '{@link #getAlgorithm() <em>Algorithm</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAlgorithm()
	 * @generated
	 * @ordered
	 */
	protected static final String ALGORITHM_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getAlgorithm() <em>Algorithm</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAlgorithm()
	 * @generated
	 * @ordered
	 */
	protected String algorithm = ALGORITHM_EDEFAULT;

	/**
	 * The cached value of the '{@link #getFitParameters() <em>Fit Parameters</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getFitParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<String> fitParameters;

	/**
	 * The cached value of the '{@link #getPredictParameters() <em>Predict Parameters</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPredictParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<String> predictParameters;

	/**
	 * The default value of the '{@link #isWithoutWeight() <em>Without Weight</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isWithoutWeight()
	 * @generated
	 * @ordered
	 */
	protected static final boolean WITHOUT_WEIGHT_EDEFAULT = true;

	/**
	 * The cached value of the '{@link #isWithoutWeight() <em>Without Weight</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isWithoutWeight()
	 * @generated
	 * @ordered
	 */
	protected boolean withoutWeight = WITHOUT_WEIGHT_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected TrainingMethodImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return FairmlPackage.Literals.TRAINING_METHOD;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getAlgorithm() {
		return algorithm;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setAlgorithm(String newAlgorithm) {
		String oldAlgorithm = algorithm;
		algorithm = newAlgorithm;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.TRAINING_METHOD__ALGORITHM, oldAlgorithm, algorithm));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getFitParameters() {
		if (fitParameters == null) {
			fitParameters = new EDataTypeEList<String>(String.class, this, FairmlPackage.TRAINING_METHOD__FIT_PARAMETERS);
		}
		return fitParameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getPredictParameters() {
		if (predictParameters == null) {
			predictParameters = new EDataTypeEList<String>(String.class, this, FairmlPackage.TRAINING_METHOD__PREDICT_PARAMETERS);
		}
		return predictParameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isWithoutWeight() {
		return withoutWeight;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setWithoutWeight(boolean newWithoutWeight) {
		boolean oldWithoutWeight = withoutWeight;
		withoutWeight = newWithoutWeight;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.TRAINING_METHOD__WITHOUT_WEIGHT, oldWithoutWeight, withoutWeight));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case FairmlPackage.TRAINING_METHOD__ALGORITHM:
				return getAlgorithm();
			case FairmlPackage.TRAINING_METHOD__FIT_PARAMETERS:
				return getFitParameters();
			case FairmlPackage.TRAINING_METHOD__PREDICT_PARAMETERS:
				return getPredictParameters();
			case FairmlPackage.TRAINING_METHOD__WITHOUT_WEIGHT:
				return isWithoutWeight();
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
			case FairmlPackage.TRAINING_METHOD__ALGORITHM:
				setAlgorithm((String)newValue);
				return;
			case FairmlPackage.TRAINING_METHOD__FIT_PARAMETERS:
				getFitParameters().clear();
				getFitParameters().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.TRAINING_METHOD__PREDICT_PARAMETERS:
				getPredictParameters().clear();
				getPredictParameters().addAll((Collection<? extends String>)newValue);
				return;
			case FairmlPackage.TRAINING_METHOD__WITHOUT_WEIGHT:
				setWithoutWeight((Boolean)newValue);
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
			case FairmlPackage.TRAINING_METHOD__ALGORITHM:
				setAlgorithm(ALGORITHM_EDEFAULT);
				return;
			case FairmlPackage.TRAINING_METHOD__FIT_PARAMETERS:
				getFitParameters().clear();
				return;
			case FairmlPackage.TRAINING_METHOD__PREDICT_PARAMETERS:
				getPredictParameters().clear();
				return;
			case FairmlPackage.TRAINING_METHOD__WITHOUT_WEIGHT:
				setWithoutWeight(WITHOUT_WEIGHT_EDEFAULT);
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
			case FairmlPackage.TRAINING_METHOD__ALGORITHM:
				return ALGORITHM_EDEFAULT == null ? algorithm != null : !ALGORITHM_EDEFAULT.equals(algorithm);
			case FairmlPackage.TRAINING_METHOD__FIT_PARAMETERS:
				return fitParameters != null && !fitParameters.isEmpty();
			case FairmlPackage.TRAINING_METHOD__PREDICT_PARAMETERS:
				return predictParameters != null && !predictParameters.isEmpty();
			case FairmlPackage.TRAINING_METHOD__WITHOUT_WEIGHT:
				return withoutWeight != WITHOUT_WEIGHT_EDEFAULT;
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
		result.append(" (algorithm: ");
		result.append(algorithm);
		result.append(", fitParameters: ");
		result.append(fitParameters);
		result.append(", predictParameters: ");
		result.append(predictParameters);
		result.append(", withoutWeight: ");
		result.append(withoutWeight);
		result.append(')');
		return result.toString();
	}

} //TrainingMethodImpl
