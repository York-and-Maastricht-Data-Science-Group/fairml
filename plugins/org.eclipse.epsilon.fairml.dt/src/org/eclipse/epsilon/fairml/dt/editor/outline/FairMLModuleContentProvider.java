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

import java.util.ArrayList;
import java.util.List;

import org.eclipse.epsilon.common.module.ModuleElement;
import org.eclipse.epsilon.eol.dt.editor.outline.EolModuleContentProvider;
import org.eclipse.epsilon.fairml.FairMLModule;
import org.eclipse.epsilon.fairml.FairMLRule;

/**
 * FairMLModuleContentProvider.
 *
 * @author Alfa Yohannis
 */
public class FairMLModuleContentProvider extends EolModuleContentProvider {

	@Override
	public ModuleElement getFocusedModuleElement(ModuleElement moduleElement) {

		if (moduleElement instanceof FairMLRule) {
//			return ((FairMLRule) moduleElement).getParameter();
			return moduleElement;
		}

		return super.getFocusedModuleElement(moduleElement);
	}

	@Override
	public List<ModuleElement> getVisibleChildren(ModuleElement moduleElement) {
		
		if (moduleElement.getClass() == FairMLModule.class) {
			FairMLModule module = (FairMLModule) moduleElement;
			List<ModuleElement> visible = new ArrayList<>();
			visible.addAll(module.getImports());
			visible.addAll(module.getDeclaredModelDeclarations());
			visible.addAll(module.getDatasetRules());
			visible.addAll(module.getDeclaredOperations());
			return visible;
		}
		else {
			return super.getVisibleChildren(moduleElement);
		}
	}

}
