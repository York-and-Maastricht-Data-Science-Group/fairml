/*********************************************************************
 * Copyright (c) 2020 The University of York.
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
package org.eclipse.epsilon.fairml.dt.editor;

import java.util.ArrayList;
import java.util.List;

import org.eclipse.epsilon.common.dt.editor.outline.ModuleContentProvider;
import org.eclipse.epsilon.common.dt.editor.outline.ModuleElementLabelProvider;
import org.eclipse.epsilon.common.module.IModule;
import org.eclipse.epsilon.eol.dt.editor.EolEditor;
import org.eclipse.epsilon.fairml.FairMLModule;
import org.eclipse.epsilon.fairml.dt.editor.outline.FairMLModuleContentProvider;
import org.eclipse.epsilon.fairml.dt.editor.outline.FairMLModuleElementLabelProvider;

/**
 * FairMLEditor.
 *
 * @author Alfa Yohannis
 */
public class FairMLEditor extends EolEditor {

	public FairMLEditor() {
	}

	@Override
	public List<String> getKeywords() {
		List<String> keywords = new ArrayList<>();
		keywords.add("pre");
		keywords.add("post");
		keywords.add("fairml");
		keywords.add("guard");
		keywords.add("source");
		keywords.add("protect");
		keywords.add("predict");
		keywords.add("algorithm");
		keywords.add("checking");
		keywords.add("mitigation");
		
		keywords.addAll(super.getKeywords());
		return keywords;
	}

	@Override
	public ModuleElementLabelProvider createModuleElementLabelProvider() {
		return new FairMLModuleElementLabelProvider();
	}

	@Override
	protected ModuleContentProvider createModuleContentProvider() {
		return new FairMLModuleContentProvider();
	}

	@Override
	public IModule createModule() {
		return new FairMLModule();
	}

}
