/*********************************************************************
 * Copyright (c) 2020 The University of York.
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
package org.eclipse.epsilon.fairml;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.antlr.runtime.ANTLRInputStream;
import org.antlr.runtime.Lexer;
import org.antlr.runtime.TokenStream;
import org.eclipse.epsilon.common.module.IModule;
import org.eclipse.epsilon.common.module.ModuleElement;
import org.eclipse.epsilon.common.parse.AST;
import org.eclipse.epsilon.common.parse.EpsilonParser;
import org.eclipse.epsilon.common.util.AstUtil;
import org.eclipse.epsilon.eol.dom.ExecutableBlock;
import org.eclipse.epsilon.eol.exceptions.EolRuntimeException;
import org.eclipse.epsilon.erl.ErlModule;
import org.eclipse.epsilon.fairml.parse.FairMLLexer;
import org.eclipse.epsilon.fairml.parse.FairMLParser;

/**
 * FairMLModule.
 *
 * @author Alfonso de la Vega
 * @since 2.1
 */
public class FairMLModule extends ErlModule {

	protected List<FairMLRule> fairMLRules = new ArrayList<>();
	protected String outputFolder = "";
	protected String extension = ".ipynb";
	protected String prefix = "";
	protected boolean silent = false;
	protected boolean persistNotebook = true;

	@Override
	public ModuleElement adapt(AST cst, ModuleElement parentAst) {
		switch (cst.getType()) {
		case FairMLParser.FAIRML:
			return new FairMLRule();
		case FairMLParser.SOURCE:
		case FairMLParser.PROTECT:
		case FairMLParser.PREDICT:
		case FairMLParser.ALGORITHM:
		case FairMLParser.CHECKING:
		case FairMLParser.MITIGATION:
			return new ExecutableBlock<>(Collection.class);
		}
		return super.adapt(cst, parentAst);
	}

	@Override
	public void build(AST cst, IModule module) {
		super.build(cst, module);
		for (AST processRuleAst : AstUtil.getChildren(cst, FairMLParser.FAIRML)) {
			fairMLRules.add((FairMLRule) module.createAst(processRuleAst, this));
		}
	}

	@Override
	public Lexer createLexer(ANTLRInputStream inputStream) {
		return new FairMLLexer(inputStream);
	}

	@Override
	public EpsilonParser createParser(TokenStream tokenStream) {
		return new FairMLParser(tokenStream);
	}

	@Override
	public String getMainRule() {
		return "fairmlModule";
	}

	@Override
	public HashMap<String, Class<?>> getImportConfiguration() {
		HashMap<String, Class<?>> importConfiguration = super.getImportConfiguration();
		importConfiguration.put("fairml", FairMLModule.class);
		return importConfiguration;
	}

	@Override
	protected Object processRules() throws EolRuntimeException {
		for (FairMLRule fairMLRule : fairMLRules) {
			fairMLRule.execute(context);
			if (persistNotebook) {
				try {
					new IPYNBFile(getFilePath(fairMLRule), fairMLRule.getFairML()).save();
				}
				catch (FileNotFoundException e) {
					throw new EolRuntimeException(e);
				}
				fairMLRule.dispose();
			}
		}
		return null;
	}

	public void preExecution() throws EolRuntimeException {
		execute(getPre(), getContext());
	}

	public List<FairMLRule> getDatasetRules() {
		return fairMLRules;
	}

	public FairMLRule getDatasetRule(String ruleName) {
		for (FairMLRule rule : fairMLRules) {
			if (rule.getName().equalsIgnoreCase(ruleName)) {
				return rule;
			}
		}
		return null;
	}

	public void setOutputFolder(String attribute) {
		outputFolder = attribute;
	}

	public String getOutputFolder() {
		if (outputFolder.equals("") && getSourceFile() != null) {
			outputFolder = getSourceFile().getParent();
		}
		return outputFolder;
	}

	public String getFilePath(FairMLRule rule) {
		return String.format("%s/%s",
				getOutputFolder(), getFileName(rule));
	}

	public String getFileName(FairMLRule rule) {
		return String.format("%s%s%s",
				getPrefix(), rule.getName(), getExtension());
	}

	public String getPrefix() {
		return prefix;
	}

	public void setPrefix(String prefix) {
		this.prefix = prefix;
	}

	public String getExtension() {
		return extension;
	}

	public void setExtension(String extension) {
		this.extension = extension;
	}

	public boolean isSilent() {
		return silent;
	}

	public void setSilent(boolean silent) {
		this.silent = silent;
	}
	

	/**
	 * Set whether the notebooks must be persisted into output files or not
	 */
	public void persistNotebooks(boolean persistNotebooks) {
		this.persistNotebook = persistNotebooks;
	}
}
