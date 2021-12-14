/*******************************************************************************
 * Copyright (c) 2021 The University of York.
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *
 * Contributors:
 *     Dimitrios Kolovos - initial API and EDL demo implementation
 *     Pablo Sanchez - API and language discussion
 *     Alfonso de la Vega - initial API and implementation
 *     Alfa Yohannis - initial API and implementation
 * -----------------------------------------------------------------------------
 * ANTLR 3 License
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/
parser grammar FairMLParserRules;

options {backtrack=true; output=AST;}

tokens {
  FAIRML;
  SOURCE; 
  PROTECT;
  PREDICT;
  ALGORITHM;
  CHECKING;
  MITIGATION;  
}

/***
	FairML: Domain-specific Language for Fair Machine Learning
***/

fairmlRule
  @after {
    $tree.getExtraTokens().add($ob);
    $tree.getExtraTokens().add($cb);
  }
  : r='fairml'^ NAME ob='{'! generationRuleConstructs cb='}'!
  {$r.setType(FAIRML);}
  ;

source
	:	g='source'^ expressionOrStatementBlock
	{$g.setType(SOURCE);}
	;
	
protect
	:	g='protect'^ expressionOrStatementBlock
	{$g.setType(PROTECT);}
	;

predict
	:	g='predict'^ expressionOrStatementBlock
	{$g.setType(PREDICT);}
	;
	
algorithm
	:	g='algorithm'^ expressionOrStatementBlock
	{$g.setType(ALGORITHM);}
	;

checking
	:	g='checking'^ expressionOrStatementBlock
	{$g.setType(CHECKING);}
	;
	
mitigation
	:	g='mitigation'^ expressionOrStatementBlock
	{$g.setType(MITIGATION);}
	;
	
generationRuleConstructs
	:	(source | protect| predict| algorithm | checking | mitigation )*
	;
