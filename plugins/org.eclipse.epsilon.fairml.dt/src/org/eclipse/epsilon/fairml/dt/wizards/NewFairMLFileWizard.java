/*********************************************************************
 * Copyright (c) 2021 The University of York.
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
package org.eclipse.epsilon.fairml.dt.wizards;

import org.eclipse.epsilon.common.dt.wizards.AbstractNewFileWizard2;

/**
 * NewFairMLFileWizard.
 *
 * @author Alfa Yohannis
 */
public class NewFairMLFileWizard extends AbstractNewFileWizard2 {

	@Override
	public String getTitle() {
		return "New FairML file";
	}

	@Override
	public String getExtension() {
		return "fairml";
	}

	@Override
	public String getDescription() {
		return "This wizard creates a new FairML file with *.fairml extension.";
	}

}
