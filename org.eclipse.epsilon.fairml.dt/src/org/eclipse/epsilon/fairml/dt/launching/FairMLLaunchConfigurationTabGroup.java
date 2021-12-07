/*********************************************************************
 * Copyright (c) 2020 The University of York.
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
package org.eclipse.epsilon.fairml.dt.launching;

import org.eclipse.debug.ui.ILaunchConfigurationTab;
import org.eclipse.epsilon.common.dt.launching.tabs.EpsilonLaunchConfigurationTabGroup;
import org.eclipse.epsilon.fairml.dt.launching.tabs.FairMLAdvancedConfigurationTab;
import org.eclipse.epsilon.fairml.dt.launching.tabs.FairMLSourceConfigurationTab;

/**
 * FairMLLaunchConfigurationTabGroup.
 *
 * @author Alfonso de la Vega
 * @since 2.1
 */
public class FairMLLaunchConfigurationTabGroup extends EpsilonLaunchConfigurationTabGroup {

	@Override
	public ILaunchConfigurationTab getSourceConfigurationTab() {
		return new FairMLSourceConfigurationTab();
	}

	@Override
	public ILaunchConfigurationTab[] getOtherConfigurationTabs() {
		return new ILaunchConfigurationTab[] {};
	}

	@Override
	public ILaunchConfigurationTab getAdvancedConfigurationTab() {
		return new FairMLAdvancedConfigurationTab();
	}

}
