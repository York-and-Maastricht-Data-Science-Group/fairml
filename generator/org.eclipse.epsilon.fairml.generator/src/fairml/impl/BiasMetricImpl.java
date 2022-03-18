/**
 */
package fairml.impl;

import fairml.BiasMetric;
import fairml.FairmlPackage;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Bias Metric</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link fairml.impl.BiasMetricImpl#getClassName <em>Class Name</em>}</li>
 *   <li>{@link fairml.impl.BiasMetricImpl#getDatasetType <em>Dataset Type</em>}</li>
 *   <li>{@link fairml.impl.BiasMetricImpl#isOptimalThreshold <em>Optimal Threshold</em>}</li>
 *   <li>{@link fairml.impl.BiasMetricImpl#isPlotThreshold <em>Plot Threshold</em>}</li>
 * </ul>
 *
 * @generated
 */
public class BiasMetricImpl extends OperationImpl implements BiasMetric {
	/**
	 * The default value of the '{@link #getClassName() <em>Class Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getClassName()
	 * @generated
	 * @ordered
	 */
	protected static final String CLASS_NAME_EDEFAULT = "FairMLMetric";

	/**
	 * The cached value of the '{@link #getClassName() <em>Class Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getClassName()
	 * @generated
	 * @ordered
	 */
	protected String className = CLASS_NAME_EDEFAULT;

	/**
	 * The default value of the '{@link #getDatasetType() <em>Dataset Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasetType()
	 * @generated
	 * @ordered
	 */
	protected static final String DATASET_TYPE_EDEFAULT = "test";

	/**
	 * The cached value of the '{@link #getDatasetType() <em>Dataset Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasetType()
	 * @generated
	 * @ordered
	 */
	protected String datasetType = DATASET_TYPE_EDEFAULT;

	/**
	 * The default value of the '{@link #isOptimalThreshold() <em>Optimal Threshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isOptimalThreshold()
	 * @generated
	 * @ordered
	 */
	protected static final boolean OPTIMAL_THRESHOLD_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isOptimalThreshold() <em>Optimal Threshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isOptimalThreshold()
	 * @generated
	 * @ordered
	 */
	protected boolean optimalThreshold = OPTIMAL_THRESHOLD_EDEFAULT;

	/**
	 * The default value of the '{@link #isPlotThreshold() <em>Plot Threshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isPlotThreshold()
	 * @generated
	 * @ordered
	 */
	protected static final boolean PLOT_THRESHOLD_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isPlotThreshold() <em>Plot Threshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isPlotThreshold()
	 * @generated
	 * @ordered
	 */
	protected boolean plotThreshold = PLOT_THRESHOLD_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected BiasMetricImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return FairmlPackage.Literals.BIAS_METRIC;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getClassName() {
		return className;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setClassName(String newClassName) {
		String oldClassName = className;
		className = newClassName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_METRIC__CLASS_NAME, oldClassName, className));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getDatasetType() {
		return datasetType;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setDatasetType(String newDatasetType) {
		String oldDatasetType = datasetType;
		datasetType = newDatasetType;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_METRIC__DATASET_TYPE, oldDatasetType, datasetType));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isOptimalThreshold() {
		return optimalThreshold;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setOptimalThreshold(boolean newOptimalThreshold) {
		boolean oldOptimalThreshold = optimalThreshold;
		optimalThreshold = newOptimalThreshold;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_METRIC__OPTIMAL_THRESHOLD, oldOptimalThreshold, optimalThreshold));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isPlotThreshold() {
		return plotThreshold;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setPlotThreshold(boolean newPlotThreshold) {
		boolean oldPlotThreshold = plotThreshold;
		plotThreshold = newPlotThreshold;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.BIAS_METRIC__PLOT_THRESHOLD, oldPlotThreshold, plotThreshold));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case FairmlPackage.BIAS_METRIC__CLASS_NAME:
				return getClassName();
			case FairmlPackage.BIAS_METRIC__DATASET_TYPE:
				return getDatasetType();
			case FairmlPackage.BIAS_METRIC__OPTIMAL_THRESHOLD:
				return isOptimalThreshold();
			case FairmlPackage.BIAS_METRIC__PLOT_THRESHOLD:
				return isPlotThreshold();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case FairmlPackage.BIAS_METRIC__CLASS_NAME:
				setClassName((String)newValue);
				return;
			case FairmlPackage.BIAS_METRIC__DATASET_TYPE:
				setDatasetType((String)newValue);
				return;
			case FairmlPackage.BIAS_METRIC__OPTIMAL_THRESHOLD:
				setOptimalThreshold((Boolean)newValue);
				return;
			case FairmlPackage.BIAS_METRIC__PLOT_THRESHOLD:
				setPlotThreshold((Boolean)newValue);
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
			case FairmlPackage.BIAS_METRIC__CLASS_NAME:
				setClassName(CLASS_NAME_EDEFAULT);
				return;
			case FairmlPackage.BIAS_METRIC__DATASET_TYPE:
				setDatasetType(DATASET_TYPE_EDEFAULT);
				return;
			case FairmlPackage.BIAS_METRIC__OPTIMAL_THRESHOLD:
				setOptimalThreshold(OPTIMAL_THRESHOLD_EDEFAULT);
				return;
			case FairmlPackage.BIAS_METRIC__PLOT_THRESHOLD:
				setPlotThreshold(PLOT_THRESHOLD_EDEFAULT);
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
			case FairmlPackage.BIAS_METRIC__CLASS_NAME:
				return CLASS_NAME_EDEFAULT == null ? className != null : !CLASS_NAME_EDEFAULT.equals(className);
			case FairmlPackage.BIAS_METRIC__DATASET_TYPE:
				return DATASET_TYPE_EDEFAULT == null ? datasetType != null : !DATASET_TYPE_EDEFAULT.equals(datasetType);
			case FairmlPackage.BIAS_METRIC__OPTIMAL_THRESHOLD:
				return optimalThreshold != OPTIMAL_THRESHOLD_EDEFAULT;
			case FairmlPackage.BIAS_METRIC__PLOT_THRESHOLD:
				return plotThreshold != PLOT_THRESHOLD_EDEFAULT;
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
		result.append(" (className: ");
		result.append(className);
		result.append(", datasetType: ");
		result.append(datasetType);
		result.append(", optimalThreshold: ");
		result.append(optimalThreshold);
		result.append(", plotThreshold: ");
		result.append(plotThreshold);
		result.append(')');
		return result.toString();
	}

} //BiasMetricImpl
