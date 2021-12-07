/*********************************************************************
 * Copyright (c) 2020 The University of York.
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
package org.eclipse.epsilon.fairml.dt;

import org.eclipse.epsilon.common.dt.AbstractEpsilonUIPlugin;

/**
 * FairMLPlugin.
 *
 * @author Alfonso de la Vega
 * @since 2.1
 */
public class FairMLPlugin extends AbstractEpsilonUIPlugin {

	public static FairMLPlugin getDefault() {
		return (FairMLPlugin) plugins.get(FairMLPlugin.class);
	}

}
