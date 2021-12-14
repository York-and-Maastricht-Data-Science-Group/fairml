package org.eclipse.epsilon.fairml;

import java.util.Collection;

import org.eclipse.epsilon.common.module.IModule;
import org.eclipse.epsilon.common.parse.AST;
import org.eclipse.epsilon.common.util.AstUtil;
import org.eclipse.epsilon.eol.dom.AnnotatableModuleElement;
import org.eclipse.epsilon.eol.dom.ExecutableBlock;
import org.eclipse.epsilon.eol.dom.Parameter;
import org.eclipse.epsilon.eol.exceptions.EolRuntimeException;
import org.eclipse.epsilon.eol.execute.context.IEolContext;
import org.eclipse.epsilon.fairml.parse.FairMLParser;

public class FairMLRule extends AnnotatableModuleElement {

	protected String name;
	protected FairML fairML;

	protected ExecutableBlock<Boolean> guardBlock;
	protected ExecutableBlock<Collection<String>> sourceBlock;
	protected ExecutableBlock<Collection<String>> protectBlock;
	protected ExecutableBlock<Collection<String>> predictBlock;
	protected ExecutableBlock<Collection<String>> algorithmBlock;
	protected ExecutableBlock<Collection<String>> checkingBlock;
	protected ExecutableBlock<Collection<String>> mitigationBlock;

	@SuppressWarnings("unchecked")
	@Override
	public void build(AST cst, IModule module) {
		super.build(cst, module);
		name = cst.getFirstChild().getText();
		guardBlock = (ExecutableBlock<Boolean>) module.createAst(AstUtil.getChild(cst, FairMLParser.GUARD), this);
		AST x = AstUtil.getChild(cst, FairMLParser.SOURCE);
		Object y = module.createAst(x, this);
		
		sourceBlock = (ExecutableBlock<Collection<String>>) module.createAst(AstUtil.getChild(cst, FairMLParser.SOURCE),
				this);
		protectBlock = (ExecutableBlock<Collection<String>>) module
				.createAst(AstUtil.getChild(cst, FairMLParser.PROTECT), this);
		predictBlock = (ExecutableBlock<Collection<String>>) module
				.createAst(AstUtil.getChild(cst, FairMLParser.PREDICT), this);
		algorithmBlock = (ExecutableBlock<Collection<String>>) module
				.createAst(AstUtil.getChild(cst, FairMLParser.ALGORITHM), this);
		checkingBlock = (ExecutableBlock<Collection<String>>) module
				.createAst(AstUtil.getChild(cst, FairMLParser.CHECKING), this);
		mitigationBlock = (ExecutableBlock<Collection<String>>) module
				.createAst(AstUtil.getChild(cst, FairMLParser.MITIGATION), this);
	}

	public void execute(IEolContext context) throws EolRuntimeException {

		final Collection<String> sources = sourceBlock != null ? sourceBlock.execute(context, false) : null;
		final Collection<String> protectedAttributes = protectBlock != null ? protectBlock.execute(context, false)
				: null;
		final Collection<String> predictedAttributes = predictBlock != null ? predictBlock.execute(context, false)
				: null;
		final Collection<String> algorithms = algorithmBlock != null ? algorithmBlock.execute(context, false) : null;
		final Collection<String> checkingMethods = checkingBlock != null ? checkingBlock.execute(context, false) : null;
		final Collection<String> mitigationMethods = mitigationBlock != null ? mitigationBlock.execute(context, false)
				: null;

		fairML = new FairML();

		fairML.setName(name);
		fairML.getSources().addAll(sources);
		fairML.getProtectedAttributes().addAll(protectedAttributes);
		fairML.getPredictedAttributes().addAll(predictedAttributes);
		fairML.getAlgorithms().addAll(algorithms);
		fairML.getCheckingMethods().addAll(checkingMethods);
		fairML.getMitigationMethods().addAll(mitigationMethods);
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public FairML getFairML() {
		return fairML;
	}

	public void dispose() {
		fairML = null;
	}
}
