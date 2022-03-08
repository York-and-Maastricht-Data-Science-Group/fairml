/**
 */
package fairml.impl;

import fairml.BiasMitigation;
import fairml.Dataset;
import fairml.FairML;
import fairml.FairmlPackage;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EDataTypeUniqueEList;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Fair ML</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link fairml.impl.FairMLImpl#getName <em>Name</em>}</li>
 *   <li>{@link fairml.impl.FairMLImpl#getDescription <em>Description</em>}</li>
 *   <li>{@link fairml.impl.FairMLImpl#getFilename <em>Filename</em>}</li>
 *   <li>{@link fairml.impl.FairMLImpl#getDatasets <em>Datasets</em>}</li>
 *   <li>{@link fairml.impl.FairMLImpl#getBiasMitigations <em>Bias Mitigations</em>}</li>
 *   <li>{@link fairml.impl.FairMLImpl#getModules <em>Modules</em>}</li>
 * </ul>
 *
 * @generated
 */
public class FairMLImpl extends EObjectImpl implements FairML {
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
	 * The default value of the '{@link #getDescription() <em>Description</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDescription()
	 * @generated
	 * @ordered
	 */
	protected static final String DESCRIPTION_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getDescription() <em>Description</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDescription()
	 * @generated
	 * @ordered
	 */
	protected String description = DESCRIPTION_EDEFAULT;

	/**
	 * The default value of the '{@link #getFilename() <em>Filename</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getFilename()
	 * @generated
	 * @ordered
	 */
	protected static final String FILENAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getFilename() <em>Filename</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getFilename()
	 * @generated
	 * @ordered
	 */
	protected String filename = FILENAME_EDEFAULT;

	/**
	 * The cached value of the '{@link #getDatasets() <em>Datasets</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDatasets()
	 * @generated
	 * @ordered
	 */
	protected EList<Dataset> datasets;

	/**
	 * The cached value of the '{@link #getBiasMitigations() <em>Bias Mitigations</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getBiasMitigations()
	 * @generated
	 * @ordered
	 */
	protected EList<BiasMitigation> biasMitigations;

	/**
	 * The cached value of the '{@link #getModules() <em>Modules</em>}' attribute list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getModules()
	 * @generated
	 * @ordered
	 */
	protected EList<String> modules;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected FairMLImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return FairmlPackage.Literals.FAIR_ML;
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
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.FAIR_ML__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getDescription() {
		return description;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setDescription(String newDescription) {
		String oldDescription = description;
		description = newDescription;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.FAIR_ML__DESCRIPTION, oldDescription, description));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getFilename() {
		return filename;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void setFilename(String newFilename) {
		String oldFilename = filename;
		filename = newFilename;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, FairmlPackage.FAIR_ML__FILENAME, oldFilename, filename));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<Dataset> getDatasets() {
		if (datasets == null) {
			datasets = new EObjectContainmentEList<Dataset>(Dataset.class, this, FairmlPackage.FAIR_ML__DATASETS);
		}
		return datasets;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<BiasMitigation> getBiasMitigations() {
		if (biasMitigations == null) {
			biasMitigations = new EObjectContainmentEList<BiasMitigation>(BiasMitigation.class, this, FairmlPackage.FAIR_ML__BIAS_MITIGATIONS);
		}
		return biasMitigations;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EList<String> getModules() {
		if (modules == null) {
			modules = new EDataTypeUniqueEList<String>(String.class, this, FairmlPackage.FAIR_ML__MODULES);
		}
		return modules;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case FairmlPackage.FAIR_ML__DATASETS:
				return ((InternalEList<?>)getDatasets()).basicRemove(otherEnd, msgs);
			case FairmlPackage.FAIR_ML__BIAS_MITIGATIONS:
				return ((InternalEList<?>)getBiasMitigations()).basicRemove(otherEnd, msgs);
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
			case FairmlPackage.FAIR_ML__NAME:
				return getName();
			case FairmlPackage.FAIR_ML__DESCRIPTION:
				return getDescription();
			case FairmlPackage.FAIR_ML__FILENAME:
				return getFilename();
			case FairmlPackage.FAIR_ML__DATASETS:
				return getDatasets();
			case FairmlPackage.FAIR_ML__BIAS_MITIGATIONS:
				return getBiasMitigations();
			case FairmlPackage.FAIR_ML__MODULES:
				return getModules();
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
			case FairmlPackage.FAIR_ML__NAME:
				setName((String)newValue);
				return;
			case FairmlPackage.FAIR_ML__DESCRIPTION:
				setDescription((String)newValue);
				return;
			case FairmlPackage.FAIR_ML__FILENAME:
				setFilename((String)newValue);
				return;
			case FairmlPackage.FAIR_ML__DATASETS:
				getDatasets().clear();
				getDatasets().addAll((Collection<? extends Dataset>)newValue);
				return;
			case FairmlPackage.FAIR_ML__BIAS_MITIGATIONS:
				getBiasMitigations().clear();
				getBiasMitigations().addAll((Collection<? extends BiasMitigation>)newValue);
				return;
			case FairmlPackage.FAIR_ML__MODULES:
				getModules().clear();
				getModules().addAll((Collection<? extends String>)newValue);
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
			case FairmlPackage.FAIR_ML__NAME:
				setName(NAME_EDEFAULT);
				return;
			case FairmlPackage.FAIR_ML__DESCRIPTION:
				setDescription(DESCRIPTION_EDEFAULT);
				return;
			case FairmlPackage.FAIR_ML__FILENAME:
				setFilename(FILENAME_EDEFAULT);
				return;
			case FairmlPackage.FAIR_ML__DATASETS:
				getDatasets().clear();
				return;
			case FairmlPackage.FAIR_ML__BIAS_MITIGATIONS:
				getBiasMitigations().clear();
				return;
			case FairmlPackage.FAIR_ML__MODULES:
				getModules().clear();
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
			case FairmlPackage.FAIR_ML__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case FairmlPackage.FAIR_ML__DESCRIPTION:
				return DESCRIPTION_EDEFAULT == null ? description != null : !DESCRIPTION_EDEFAULT.equals(description);
			case FairmlPackage.FAIR_ML__FILENAME:
				return FILENAME_EDEFAULT == null ? filename != null : !FILENAME_EDEFAULT.equals(filename);
			case FairmlPackage.FAIR_ML__DATASETS:
				return datasets != null && !datasets.isEmpty();
			case FairmlPackage.FAIR_ML__BIAS_MITIGATIONS:
				return biasMitigations != null && !biasMitigations.isEmpty();
			case FairmlPackage.FAIR_ML__MODULES:
				return modules != null && !modules.isEmpty();
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
		result.append(", description: ");
		result.append(description);
		result.append(", filename: ");
		result.append(filename);
		result.append(", modules: ");
		result.append(modules);
		result.append(')');
		return result.toString();
	}

} //FairMLImpl
