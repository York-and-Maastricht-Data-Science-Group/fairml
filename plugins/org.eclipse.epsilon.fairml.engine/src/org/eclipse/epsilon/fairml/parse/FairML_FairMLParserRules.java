package org.eclipse.epsilon.fairml.parse;

// $ANTLR 3.1b1 FairMLParserRules.g 2021-12-08 14:48:18

import org.antlr.runtime.*;
import java.util.Stack;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

import org.antlr.runtime.tree.*;

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
public class FairML_FairMLParserRules extends org.eclipse.epsilon.common.parse.EpsilonParser {
    public static final int T__144=144;
    public static final int T__143=143;
    public static final int T__146=146;
    public static final int MODELDECLARATIONPARAMETER=78;
    public static final int T__145=145;
    public static final int BREAKALL=44;
    public static final int T__140=140;
    public static final int T__142=142;
    public static final int VAR=53;
    public static final int MODELDECLARATIONPARAMETERS=77;
    public static final int T__141=141;
    public static final int THROW=58;
    public static final int SpecialTypeName=19;
    public static final int PARAMLIST=29;
    public static final int EXPRLIST=59;
    public static final int EXPRRANGE=60;
    public static final int BREAK=43;
    public static final int ELSE=36;
    public static final int T__137=137;
    public static final int T__136=136;
    public static final int FORMAL=28;
    public static final int IF=35;
    public static final int MultiplicativeExpression=62;
    public static final int TYPE=70;
    public static final int T__139=139;
    public static final int T__138=138;
    public static final int T__133=133;
    public static final int T__132=132;
    public static final int T__135=135;
    public static final int T__134=134;
    public static final int T__131=131;
    public static final int NewExpression=52;
    public static final int T__130=130;
    public static final int CASE=40;
    public static final int Letter=20;
    public static final int LINE_COMMENT=26;
    public static final int SOURCE=88;
    public static final int MITIGATION=93;
    public static final int T__129=129;
    public static final int T__126=126;
    public static final int JavaIDDigit=22;
    public static final int T__125=125;
    public static final int LAMBDAEXPR=69;
    public static final int MAP=80;
    public static final int T__128=128;
    public static final int T__127=127;
    public static final int T__166=166;
    public static final int T__165=165;
    public static final int T__168=168;
    public static final int T__167=167;
    public static final int T__162=162;
    public static final int T__161=161;
    public static final int T__164=164;
    public static final int MODELDECLARATION=73;
    public static final int T__163=163;
    public static final int EXPRESSIONINBRACKETS=64;
    public static final int T__160=160;
    public static final int TERNARY=37;
    public static final int TRANSACTION=46;
    public static final int FLOAT_TYPE_SUFFIX=7;
    public static final int ITEMSELECTOR=79;
    public static final int COMMENT=25;
    public static final int ModelElementType=50;
    public static final int IMPORT=72;
    public static final int DELETE=57;
    public static final int ARROW=11;
    public static final int MapTypeName=18;
    public static final int T__159=159;
    public static final int T__158=158;
    public static final int T__155=155;
    public static final int SPECIAL_ASSIGNMENT=31;
    public static final int T__154=154;
    public static final int T__157=157;
    public static final int T__156=156;
    public static final int T__151=151;
    public static final int T__150=150;
    public static final int T__153=153;
    public static final int T__152=152;
    public static final int Annotation=27;
    public static final int CONTINUE=45;
    public static final int ENUMERATION_VALUE=71;
    public static final int OPERATOR=63;
    public static final int PREDICT=90;
    public static final int EXPONENT=6;
    public static final int STRING=15;
    public static final int T__148=148;
    public static final int T__147=147;
    public static final int T__149=149;
    public static final int T__100=100;
    public static final int NAMESPACE=74;
    public static final int COLLECTION=47;
    public static final int NEW=54;
    public static final int EXTENDS=85;
    public static final int T__102=102;
    public static final int PRE=83;
    public static final int T__101=101;
    public static final int POST=84;
    public static final int ALIAS=75;
    public static final int DRIVER=76;
    public static final int KEYVAL=81;
    public static final int POINT_POINT=10;
    public static final int GUARD=86;
    public static final int T__99=99;
    public static final int CHECKING=92;
    public static final int FAIRMLMODULE=94;
    public static final int T__95=95;
    public static final int HELPERMETHOD=32;
    public static final int T__96=96;
    public static final int T__97=97;
    public static final int StatementBlock=33;
    public static final int T__98=98;
    public static final int ABORT=48;
    public static final int T__173=173;
    public static final int StrangeNameLiteral=16;
    public static final int T__172=172;
    public static final int FOR=34;
    public static final int BLOCK=67;
    public static final int T__171=171;
    public static final int T__170=170;
    public static final int PARAMETERS=51;
    public static final int SpecialNameChar=21;
    public static final int BOOLEAN=13;
    public static final int NAME=23;
    public static final int SWITCH=39;
    public static final int T__169=169;
    public static final int FeatureCall=65;
    public static final int T__122=122;
    public static final int T__121=121;
    public static final int T__124=124;
    public static final int FLOAT=4;
    public static final int T__123=123;
    public static final int T__120=120;
    public static final int ALGORITHM=91;
    public static final int NativeType=61;
    public static final int INT=8;
    public static final int PROTECT=89;
    public static final int ANNOTATIONBLOCK=55;
    public static final int RETURN=42;
    public static final int KEYVALLIST=82;
    public static final int FEATURECALL=68;
    public static final int CollectionType=49;
    public static final int T__119=119;
    public static final int ASSIGNMENT=30;
    public static final int T__118=118;
    public static final int T__115=115;
    public static final int WS=24;
    public static final int EOF=-1;
    public static final int T__114=114;
    public static final int T__117=117;
    public static final int T__116=116;
    public static final int T__111=111;
    public static final int T__110=110;
    public static final int T__113=113;
    public static final int T__112=112;
    public static final int EscapeSequence=14;
    public static final int EOLMODULE=66;
    public static final int FAIRML=87;
    public static final int CollectionTypeName=17;
    public static final int DIGIT=5;
    public static final int EXECUTABLEANNOTATION=56;
    public static final int T__108=108;
    public static final int T__107=107;
    public static final int WHILE=38;
    public static final int T__109=109;
    public static final int NAVIGATION=12;
    public static final int T__104=104;
    public static final int POINT=9;
    public static final int T__103=103;
    public static final int T__106=106;
    public static final int DEFAULT=41;
    public static final int T__105=105;

    // delegates
    // delegators
    public FairMLParser gFairML;


        public FairML_FairMLParserRules(TokenStream input, FairMLParser gFairML) {
            this(input, new RecognizerSharedState(), gFairML);
        }
        public FairML_FairMLParserRules(TokenStream input, RecognizerSharedState state, FairMLParser gFairML) {
            super(input, state);
            this.gFairML = gFairML;
        }
        
    protected TreeAdaptor adaptor = new CommonTreeAdaptor();

    public void setTreeAdaptor(TreeAdaptor adaptor) {
        this.adaptor = adaptor;
    }
    public TreeAdaptor getTreeAdaptor() {
        return adaptor;
    }

    public String[] getTokenNames() { return FairMLParser.tokenNames; }
    public String getGrammarFileName() { return "FairMLParserRules.g"; }


    public static class fairmlRule_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start fairmlRule
    // FairMLParserRules.g:57:1: fairmlRule : r= 'fairml' NAME ob= '{' generationRuleConstructs cb= '}' ;
    public final FairML_FairMLParserRules.fairmlRule_return fairmlRule() throws RecognitionException {
        FairML_FairMLParserRules.fairmlRule_return retval = new FairML_FairMLParserRules.fairmlRule_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token r=null;
        Token ob=null;
        Token cb=null;
        Token NAME1=null;
        FairML_FairMLParserRules.generationRuleConstructs_return generationRuleConstructs2 = null;


        org.eclipse.epsilon.common.parse.AST r_tree=null;
        org.eclipse.epsilon.common.parse.AST ob_tree=null;
        org.eclipse.epsilon.common.parse.AST cb_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME1_tree=null;

        try {
            // FairMLParserRules.g:66:3: (r= 'fairml' NAME ob= '{' generationRuleConstructs cb= '}' )
            // FairMLParserRules.g:66:5: r= 'fairml' NAME ob= '{' generationRuleConstructs cb= '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            r=(Token)match(input,167,FOLLOW_167_in_fairmlRule84); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            r_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(r);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(r_tree, root_0);
            }
            NAME1=(Token)match(input,NAME,FOLLOW_NAME_in_fairmlRule87); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME1_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME1);
            adaptor.addChild(root_0, NAME1_tree);
            }
            ob=(Token)match(input,100,FOLLOW_100_in_fairmlRule91); if (state.failed) return retval;
            pushFollow(FOLLOW_generationRuleConstructs_in_fairmlRule94);
            generationRuleConstructs2=generationRuleConstructs();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, generationRuleConstructs2.getTree());
            cb=(Token)match(input,101,FOLLOW_101_in_fairmlRule98); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              r.setType(FAIRML);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

                  ((org.eclipse.epsilon.common.parse.AST)retval.tree).getExtraTokens().add(ob);
                  ((org.eclipse.epsilon.common.parse.AST)retval.tree).getExtraTokens().add(cb);
                
            }
        }
        catch (RecognitionException re) {
            reportError(re);
            recover(input,re);
    	retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.errorNode(input, retval.start, input.LT(-1), re);

        }
        finally {
        }
        return retval;
    }
    // $ANTLR end fairmlRule

    public static class source_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start source
    // FairMLParserRules.g:70:1: source : g= 'source' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.source_return source() throws RecognitionException {
        FairML_FairMLParserRules.source_return retval = new FairML_FairMLParserRules.source_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token g=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock3 = null;


        org.eclipse.epsilon.common.parse.AST g_tree=null;

        try {
            // FairMLParserRules.g:71:2: (g= 'source' expressionOrStatementBlock )
            // FairMLParserRules.g:71:4: g= 'source' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            g=(Token)match(input,168,FOLLOW_168_in_source117); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            g_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(g);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(g_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_source120);
            expressionOrStatementBlock3=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock3.getTree());
            if ( state.backtracking==0 ) {
              g.setType(SOURCE);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
        }
        catch (RecognitionException re) {
            reportError(re);
            recover(input,re);
    	retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.errorNode(input, retval.start, input.LT(-1), re);

        }
        finally {
        }
        return retval;
    }
    // $ANTLR end source

    public static class protect_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start protect
    // FairMLParserRules.g:75:1: protect : g= 'protect' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.protect_return protect() throws RecognitionException {
        FairML_FairMLParserRules.protect_return retval = new FairML_FairMLParserRules.protect_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token g=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock4 = null;


        org.eclipse.epsilon.common.parse.AST g_tree=null;

        try {
            // FairMLParserRules.g:76:2: (g= 'protect' expressionOrStatementBlock )
            // FairMLParserRules.g:76:4: g= 'protect' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            g=(Token)match(input,169,FOLLOW_169_in_protect137); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            g_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(g);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(g_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_protect140);
            expressionOrStatementBlock4=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock4.getTree());
            if ( state.backtracking==0 ) {
              g.setType(PROTECT);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
        }
        catch (RecognitionException re) {
            reportError(re);
            recover(input,re);
    	retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.errorNode(input, retval.start, input.LT(-1), re);

        }
        finally {
        }
        return retval;
    }
    // $ANTLR end protect

    public static class predict_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start predict
    // FairMLParserRules.g:80:1: predict : g= 'predict' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.predict_return predict() throws RecognitionException {
        FairML_FairMLParserRules.predict_return retval = new FairML_FairMLParserRules.predict_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token g=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock5 = null;


        org.eclipse.epsilon.common.parse.AST g_tree=null;

        try {
            // FairMLParserRules.g:81:2: (g= 'predict' expressionOrStatementBlock )
            // FairMLParserRules.g:81:4: g= 'predict' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            g=(Token)match(input,170,FOLLOW_170_in_predict156); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            g_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(g);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(g_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_predict159);
            expressionOrStatementBlock5=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock5.getTree());
            if ( state.backtracking==0 ) {
              g.setType(PREDICT);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
        }
        catch (RecognitionException re) {
            reportError(re);
            recover(input,re);
    	retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.errorNode(input, retval.start, input.LT(-1), re);

        }
        finally {
        }
        return retval;
    }
    // $ANTLR end predict

    public static class algorithm_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start algorithm
    // FairMLParserRules.g:85:1: algorithm : g= 'algorithm' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.algorithm_return algorithm() throws RecognitionException {
        FairML_FairMLParserRules.algorithm_return retval = new FairML_FairMLParserRules.algorithm_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token g=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock6 = null;


        org.eclipse.epsilon.common.parse.AST g_tree=null;

        try {
            // FairMLParserRules.g:86:2: (g= 'algorithm' expressionOrStatementBlock )
            // FairMLParserRules.g:86:4: g= 'algorithm' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            g=(Token)match(input,171,FOLLOW_171_in_algorithm176); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            g_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(g);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(g_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_algorithm179);
            expressionOrStatementBlock6=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock6.getTree());
            if ( state.backtracking==0 ) {
              g.setType(ALGORITHM);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
        }
        catch (RecognitionException re) {
            reportError(re);
            recover(input,re);
    	retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.errorNode(input, retval.start, input.LT(-1), re);

        }
        finally {
        }
        return retval;
    }
    // $ANTLR end algorithm

    public static class checking_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start checking
    // FairMLParserRules.g:90:1: checking : g= 'checking' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.checking_return checking() throws RecognitionException {
        FairML_FairMLParserRules.checking_return retval = new FairML_FairMLParserRules.checking_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token g=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock7 = null;


        org.eclipse.epsilon.common.parse.AST g_tree=null;

        try {
            // FairMLParserRules.g:91:2: (g= 'checking' expressionOrStatementBlock )
            // FairMLParserRules.g:91:4: g= 'checking' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            g=(Token)match(input,172,FOLLOW_172_in_checking195); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            g_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(g);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(g_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_checking198);
            expressionOrStatementBlock7=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock7.getTree());
            if ( state.backtracking==0 ) {
              g.setType(CHECKING);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
        }
        catch (RecognitionException re) {
            reportError(re);
            recover(input,re);
    	retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.errorNode(input, retval.start, input.LT(-1), re);

        }
        finally {
        }
        return retval;
    }
    // $ANTLR end checking

    public static class mitigation_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start mitigation
    // FairMLParserRules.g:95:1: mitigation : g= 'mitigation' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.mitigation_return mitigation() throws RecognitionException {
        FairML_FairMLParserRules.mitigation_return retval = new FairML_FairMLParserRules.mitigation_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token g=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock8 = null;


        org.eclipse.epsilon.common.parse.AST g_tree=null;

        try {
            // FairMLParserRules.g:96:2: (g= 'mitigation' expressionOrStatementBlock )
            // FairMLParserRules.g:96:4: g= 'mitigation' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            g=(Token)match(input,173,FOLLOW_173_in_mitigation215); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            g_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(g);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(g_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_mitigation218);
            expressionOrStatementBlock8=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock8.getTree());
            if ( state.backtracking==0 ) {
              g.setType(MITIGATION);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
        }
        catch (RecognitionException re) {
            reportError(re);
            recover(input,re);
    	retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.errorNode(input, retval.start, input.LT(-1), re);

        }
        finally {
        }
        return retval;
    }
    // $ANTLR end mitigation

    public static class generationRuleConstructs_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start generationRuleConstructs
    // FairMLParserRules.g:100:1: generationRuleConstructs : ( source | protect | predict | algorithm | checking | mitigation )* ;
    public final FairML_FairMLParserRules.generationRuleConstructs_return generationRuleConstructs() throws RecognitionException {
        FairML_FairMLParserRules.generationRuleConstructs_return retval = new FairML_FairMLParserRules.generationRuleConstructs_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        FairML_FairMLParserRules.source_return source9 = null;

        FairML_FairMLParserRules.protect_return protect10 = null;

        FairML_FairMLParserRules.predict_return predict11 = null;

        FairML_FairMLParserRules.algorithm_return algorithm12 = null;

        FairML_FairMLParserRules.checking_return checking13 = null;

        FairML_FairMLParserRules.mitigation_return mitigation14 = null;



        try {
            // FairMLParserRules.g:101:2: ( ( source | protect | predict | algorithm | checking | mitigation )* )
            // FairMLParserRules.g:101:4: ( source | protect | predict | algorithm | checking | mitigation )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            // FairMLParserRules.g:101:4: ( source | protect | predict | algorithm | checking | mitigation )*
            loop1:
            do {
                int alt1=7;
                switch ( input.LA(1) ) {
                case 168:
                    {
                    alt1=1;
                    }
                    break;
                case 169:
                    {
                    alt1=2;
                    }
                    break;
                case 170:
                    {
                    alt1=3;
                    }
                    break;
                case 171:
                    {
                    alt1=4;
                    }
                    break;
                case 172:
                    {
                    alt1=5;
                    }
                    break;
                case 173:
                    {
                    alt1=6;
                    }
                    break;

                }

                switch (alt1) {
            	case 1 :
            	    // FairMLParserRules.g:101:5: source
            	    {
            	    pushFollow(FOLLOW_source_in_generationRuleConstructs234);
            	    source9=source();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, source9.getTree());

            	    }
            	    break;
            	case 2 :
            	    // FairMLParserRules.g:101:14: protect
            	    {
            	    pushFollow(FOLLOW_protect_in_generationRuleConstructs238);
            	    protect10=protect();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, protect10.getTree());

            	    }
            	    break;
            	case 3 :
            	    // FairMLParserRules.g:101:23: predict
            	    {
            	    pushFollow(FOLLOW_predict_in_generationRuleConstructs241);
            	    predict11=predict();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, predict11.getTree());

            	    }
            	    break;
            	case 4 :
            	    // FairMLParserRules.g:101:32: algorithm
            	    {
            	    pushFollow(FOLLOW_algorithm_in_generationRuleConstructs244);
            	    algorithm12=algorithm();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, algorithm12.getTree());

            	    }
            	    break;
            	case 5 :
            	    // FairMLParserRules.g:101:44: checking
            	    {
            	    pushFollow(FOLLOW_checking_in_generationRuleConstructs248);
            	    checking13=checking();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, checking13.getTree());

            	    }
            	    break;
            	case 6 :
            	    // FairMLParserRules.g:101:55: mitigation
            	    {
            	    pushFollow(FOLLOW_mitigation_in_generationRuleConstructs252);
            	    mitigation14=mitigation();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, mitigation14.getTree());

            	    }
            	    break;

            	default :
            	    break loop1;
                }
            } while (true);


            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
        }
        catch (RecognitionException re) {
            reportError(re);
            recover(input,re);
    	retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.errorNode(input, retval.start, input.LT(-1), re);

        }
        finally {
        }
        return retval;
    }
    // $ANTLR end generationRuleConstructs

    // Delegated rules


 

    public static final BitSet FOLLOW_167_in_fairmlRule84 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_fairmlRule87 = new BitSet(new long[]{0x0000000000000000L,0x0000001000000000L});
    public static final BitSet FOLLOW_100_in_fairmlRule91 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L,0x00003F0000000000L});
    public static final BitSet FOLLOW_generationRuleConstructs_in_fairmlRule94 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_fairmlRule98 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_168_in_source117 = new BitSet(new long[]{0x0000000000000000L,0x0000081000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_source120 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_169_in_protect137 = new BitSet(new long[]{0x0000000000000000L,0x0000081000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_protect140 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_170_in_predict156 = new BitSet(new long[]{0x0000000000000000L,0x0000081000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_predict159 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_171_in_algorithm176 = new BitSet(new long[]{0x0000000000000000L,0x0000081000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_algorithm179 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_172_in_checking195 = new BitSet(new long[]{0x0000000000000000L,0x0000081000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_checking198 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_173_in_mitigation215 = new BitSet(new long[]{0x0000000000000000L,0x0000081000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_mitigation218 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_source_in_generationRuleConstructs234 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x00003F0000000000L});
    public static final BitSet FOLLOW_protect_in_generationRuleConstructs238 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x00003F0000000000L});
    public static final BitSet FOLLOW_predict_in_generationRuleConstructs241 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x00003F0000000000L});
    public static final BitSet FOLLOW_algorithm_in_generationRuleConstructs244 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x00003F0000000000L});
    public static final BitSet FOLLOW_checking_in_generationRuleConstructs248 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x00003F0000000000L});
    public static final BitSet FOLLOW_mitigation_in_generationRuleConstructs252 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x00003F0000000000L});

}
