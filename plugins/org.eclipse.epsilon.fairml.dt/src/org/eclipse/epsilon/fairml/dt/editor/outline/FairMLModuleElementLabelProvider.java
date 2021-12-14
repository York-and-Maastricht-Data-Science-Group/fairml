/*********************************************************************
 * Copyright (c) 2020 The University of York.
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
package org.eclipse.epsilon.fairml.dt.editor.outline;

import org.eclipse.epsilon.eol.dt.editor.outline.EolModuleElementLabelProvider;
import org.eclipse.epsilon.fairml.FairMLRule;
import org.eclipse.epsilon.fairml.dt.FairMLPlugin;
import org.eclipse.swt.graphics.Image;

/**
 * FairMLModuleElementLabelProvider.
 *
 * @author Alfas Yohannis
 */
public class FairMLModuleElementLabelProvider extends EolModuleElementLabelProvider {

	@Override
	public Image getImage(Object element) {
		if (element instanceof FairMLRule) {
			return FairMLPlugin.getDefault().createImage("icons/process-rule.png");
		}
		else {
			return super.getImage(element);
		}
	}

	@Override
	public String getText(Object element) {
		if (element instanceof FairMLRule) {
			FairMLRule processRule = (FairMLRule) element;
			return processRule.getName();
		}
		return super.getText(element);
	}
}
