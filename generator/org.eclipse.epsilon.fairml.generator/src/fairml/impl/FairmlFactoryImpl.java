/**
 */
package fairml.impl;

import fairml.*;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class FairmlFactoryImpl extends EFactoryImpl implements FairmlFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static FairmlFactory init() {
		try {
			FairmlFactory theFairmlFactory = (FairmlFactory)EPackage.Registry.INSTANCE.getEFactory(FairmlPackage.eNS_URI);
			if (theFairmlFactory != null) {
				return theFairmlFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new FairmlFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public FairmlFactoryImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case FairmlPackage.FAIR_ML: return createFairML();
			case FairmlPackage.FUNCTION: return createFunction();
			case FairmlPackage.TRAINING_METHOD: return createTrainingMethod();
			case FairmlPackage.MITIGATION_METHOD: return createMitigationMethod();
			case FairmlPackage.BIAS_METRIC: return createBiasMetric();
			case FairmlPackage.DATASET: return createDataset();
			case FairmlPackage.BIAS_MITIGATION: return createBiasMitigation();
			default:
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public FairML createFairML() {
		FairMLImpl fairML = new FairMLImpl();
		return fairML;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Function createFunction() {
		FunctionImpl function = new FunctionImpl();
		return function;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public TrainingMethod createTrainingMethod() {
		TrainingMethodImpl trainingMethod = new TrainingMethodImpl();
		return trainingMethod;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public MitigationMethod createMitigationMethod() {
		MitigationMethodImpl mitigationMethod = new MitigationMethodImpl();
		return mitigationMethod;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public BiasMetric createBiasMetric() {
		BiasMetricImpl biasMetric = new BiasMetricImpl();
		return biasMetric;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Dataset createDataset() {
		DatasetImpl dataset = new DatasetImpl();
		return dataset;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public BiasMitigation createBiasMitigation() {
		BiasMitigationImpl biasMitigation = new BiasMitigationImpl();
		return biasMitigation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public FairmlPackage getFairmlPackage() {
		return (FairmlPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static FairmlPackage getPackage() {
		return FairmlPackage.eINSTANCE;
	}

} //FairmlFactoryImpl
