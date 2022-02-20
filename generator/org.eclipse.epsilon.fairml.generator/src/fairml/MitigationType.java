/**
 */
package fairml;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>Mitigation Type</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see fairml.FairmlPackage#getMitigationType()
 * @model
 * @generated
 */
public enum MitigationType implements Enumerator {
	/**
	 * The '<em><b>Preprocessing</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #PREPROCESSING_VALUE
	 * @generated
	 * @ordered
	 */
	PREPROCESSING(0, "Preprocessing", "Preprocessing"),

	/**
	 * The '<em><b>Inprocessing</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INPROCESSING_VALUE
	 * @generated
	 * @ordered
	 */
	INPROCESSING(1, "Inprocessing", "Inprocessing"),

	/**
	 * The '<em><b>Postprocessing</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #POSTPROCESSING_VALUE
	 * @generated
	 * @ordered
	 */
	POSTPROCESSING(2, "Postprocessing", "Postprocessing");

	/**
	 * The '<em><b>Preprocessing</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #PREPROCESSING
	 * @model name="Preprocessing"
	 * @generated
	 * @ordered
	 */
	public static final int PREPROCESSING_VALUE = 0;

	/**
	 * The '<em><b>Inprocessing</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INPROCESSING
	 * @model name="Inprocessing"
	 * @generated
	 * @ordered
	 */
	public static final int INPROCESSING_VALUE = 1;

	/**
	 * The '<em><b>Postprocessing</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #POSTPROCESSING
	 * @model name="Postprocessing"
	 * @generated
	 * @ordered
	 */
	public static final int POSTPROCESSING_VALUE = 2;

	/**
	 * An array of all the '<em><b>Mitigation Type</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final MitigationType[] VALUES_ARRAY =
		new MitigationType[] {
			PREPROCESSING,
			INPROCESSING,
			POSTPROCESSING,
		};

	/**
	 * A public read-only list of all the '<em><b>Mitigation Type</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<MitigationType> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Mitigation Type</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param literal the literal.
	 * @return the matching enumerator or <code>null</code>.
	 * @generated
	 */
	public static MitigationType get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			MitigationType result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Mitigation Type</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param name the name.
	 * @return the matching enumerator or <code>null</code>.
	 * @generated
	 */
	public static MitigationType getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			MitigationType result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Mitigation Type</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the integer value.
	 * @return the matching enumerator or <code>null</code>.
	 * @generated
	 */
	public static MitigationType get(int value) {
		switch (value) {
			case PREPROCESSING_VALUE: return PREPROCESSING;
			case INPROCESSING_VALUE: return INPROCESSING;
			case POSTPROCESSING_VALUE: return POSTPROCESSING;
		}
		return null;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final int value;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final String name;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final String literal;

	/**
	 * Only this class can construct instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private MitigationType(int value, String name, String literal) {
		this.value = value;
		this.name = name;
		this.literal = literal;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public int getValue() {
	  return value;
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
	public String getLiteral() {
	  return literal;
	}

	/**
	 * Returns the literal value of the enumerator, which is its string representation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		return literal;
	}
	
} //MitigationType
