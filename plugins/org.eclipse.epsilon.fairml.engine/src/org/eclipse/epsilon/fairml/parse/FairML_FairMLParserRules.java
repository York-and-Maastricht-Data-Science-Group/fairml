package org.eclipse.epsilon.fairml.parse;

// $ANTLR 3.1b1 FairMLParserRules.g 2021-12-08 11:44:31

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
    public static final int ALIASEDNAME=89;
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
    public static final int T__129=129;
    public static final int T__126=126;
    public static final int JavaIDDigit=22;
    public static final int GRIDHEADER=95;
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
    public static final int GRIDBODY=96;
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
    public static final int GRID=93;
    public static final int Annotation=27;
    public static final int CONTINUE=45;
    public static final int ENUMERATION_VALUE=71;
    public static final int OPERATOR=63;
    public static final int EXPONENT=6;
    public static final int STRING=15;
    public static final int T__148=148;
    public static final int T__147=147;
    public static final int T__149=149;
    public static final int NAMESLIST=88;
    public static final int T__100=100;
    public static final int NAMESPACE=74;
    public static final int COLLECTION=47;
    public static final int NEW=54;
    public static final int EXTENDS=85;
    public static final int T__102=102;
    public static final int PRE=83;
    public static final int T__101=101;
    public static final int PROPERTIES=90;
    public static final int POST=84;
    public static final int ALIAS=75;
    public static final int DRIVER=76;
    public static final int COLUMN=91;
    public static final int T__180=180;
    public static final int DATASET=87;
    public static final int T__182=182;
    public static final int T__181=181;
    public static final int FROM=97;
    public static final int KEYVAL=81;
    public static final int POINT_POINT=10;
    public static final int GUARD=86;
    public static final int FAIRMLMODULE=99;
    public static final int HELPERMETHOD=32;
    public static final int StatementBlock=33;
    public static final int GRIDKEYS=94;
    public static final int T__177=177;
    public static final int T__176=176;
    public static final int T__179=179;
    public static final int T__178=178;
    public static final int ABORT=48;
    public static final int T__173=173;
    public static final int StrangeNameLiteral=16;
    public static final int T__172=172;
    public static final int T__175=175;
    public static final int T__174=174;
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
    public static final int NativeType=61;
    public static final int INT=8;
    public static final int ANNOTATIONBLOCK=55;
    public static final int RETURN=42;
    public static final int KEYVALLIST=82;
    public static final int FEATURECALL=68;
    public static final int CollectionType=49;
    public static final int T__119=119;
    public static final int ASSIGNMENT=30;
    public static final int REFERENCE=92;
    public static final int T__118=118;
    public static final int T__115=115;
    public static final int WS=24;
    public static final int NESTEDFROM=98;
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


    public static class datasetRule_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start datasetRule
    // FairMLParserRules.g:63:1: datasetRule : r= 'dataset' NAME 'over' formalParameter ( from )? ob= '{' ( guard )? ( properties )? ( columnGenerator )* cb= '}' ;
    public final FairML_FairMLParserRules.datasetRule_return datasetRule() throws RecognitionException {
        FairML_FairMLParserRules.datasetRule_return retval = new FairML_FairMLParserRules.datasetRule_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token r=null;
        Token ob=null;
        Token cb=null;
        Token NAME1=null;
        Token string_literal2=null;
        FairML_EolParserRules.formalParameter_return formalParameter3 = null;

        FairML_FairMLParserRules.from_return from4 = null;

        FairML_ErlParserRules.guard_return guard5 = null;

        FairML_FairMLParserRules.properties_return properties6 = null;

        FairML_FairMLParserRules.columnGenerator_return columnGenerator7 = null;


        org.eclipse.epsilon.common.parse.AST r_tree=null;
        org.eclipse.epsilon.common.parse.AST ob_tree=null;
        org.eclipse.epsilon.common.parse.AST cb_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME1_tree=null;
        org.eclipse.epsilon.common.parse.AST string_literal2_tree=null;

        try {
            // FairMLParserRules.g:68:3: (r= 'dataset' NAME 'over' formalParameter ( from )? ob= '{' ( guard )? ( properties )? ( columnGenerator )* cb= '}' )
            // FairMLParserRules.g:68:5: r= 'dataset' NAME 'over' formalParameter ( from )? ob= '{' ( guard )? ( properties )? ( columnGenerator )* cb= '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            r=(Token)match(input,172,FOLLOW_172_in_datasetRule108); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            r_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(r);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(r_tree, root_0);
            }
            NAME1=(Token)match(input,NAME,FOLLOW_NAME_in_datasetRule111); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME1_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME1);
            adaptor.addChild(root_0, NAME1_tree);
            }
            string_literal2=(Token)match(input,173,FOLLOW_173_in_datasetRule113); if (state.failed) return retval;
            pushFollow(FOLLOW_formalParameter_in_datasetRule116);
            formalParameter3=gFairML.formalParameter();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, formalParameter3.getTree());
            // FairMLParserRules.g:68:47: ( from )?
            int alt1=2;
            int LA1_0 = input.LA(1);

            if ( (LA1_0==174) ) {
                alt1=1;
            }
            switch (alt1) {
                case 1 :
                    // FairMLParserRules.g:0:0: from
                    {
                    pushFollow(FOLLOW_from_in_datasetRule118);
                    from4=from();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, from4.getTree());

                    }
                    break;

            }

            ob=(Token)match(input,105,FOLLOW_105_in_datasetRule123); if (state.failed) return retval;
            // FairMLParserRules.g:69:5: ( guard )?
            int alt2=2;
            int LA2_0 = input.LA(1);

            if ( (LA2_0==170) ) {
                alt2=1;
            }
            switch (alt2) {
                case 1 :
                    // FairMLParserRules.g:0:0: guard
                    {
                    pushFollow(FOLLOW_guard_in_datasetRule130);
                    guard5=gFairML.guard();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, guard5.getTree());

                    }
                    break;

            }

            // FairMLParserRules.g:70:5: ( properties )?
            int alt3=2;
            int LA3_0 = input.LA(1);

            if ( (LA3_0==176) ) {
                alt3=1;
            }
            switch (alt3) {
                case 1 :
                    // FairMLParserRules.g:0:0: properties
                    {
                    pushFollow(FOLLOW_properties_in_datasetRule137);
                    properties6=properties();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, properties6.getTree());

                    }
                    break;

            }

            // FairMLParserRules.g:71:5: ( columnGenerator )*
            loop4:
            do {
                int alt4=2;
                int LA4_0 = input.LA(1);

                if ( (LA4_0==Annotation||LA4_0==114||LA4_0==174||(LA4_0>=177 && LA4_0<=179)) ) {
                    alt4=1;
                }


                switch (alt4) {
            	case 1 :
            	    // FairMLParserRules.g:0:0: columnGenerator
            	    {
            	    pushFollow(FOLLOW_columnGenerator_in_datasetRule144);
            	    columnGenerator7=columnGenerator();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, columnGenerator7.getTree());

            	    }
            	    break;

            	default :
            	    break loop4;
                }
            } while (true);

            cb=(Token)match(input,106,FOLLOW_106_in_datasetRule151); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              r.setType(DATASET);
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
    // $ANTLR end datasetRule

    public static class columnGenerator_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start columnGenerator
    // FairMLParserRules.g:76:1: columnGenerator : ( reference | ( annotationBlock )? column | ( annotationBlock )? grid | nestedFrom );
    public final FairML_FairMLParserRules.columnGenerator_return columnGenerator() throws RecognitionException {
        FairML_FairMLParserRules.columnGenerator_return retval = new FairML_FairMLParserRules.columnGenerator_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        FairML_FairMLParserRules.reference_return reference8 = null;

        FairML_EolParserRules.annotationBlock_return annotationBlock9 = null;

        FairML_FairMLParserRules.column_return column10 = null;

        FairML_EolParserRules.annotationBlock_return annotationBlock11 = null;

        FairML_FairMLParserRules.grid_return grid12 = null;

        FairML_FairMLParserRules.nestedFrom_return nestedFrom13 = null;



        try {
            // FairMLParserRules.g:77:3: ( reference | ( annotationBlock )? column | ( annotationBlock )? grid | nestedFrom )
            int alt7=4;
            switch ( input.LA(1) ) {
            case 177:
                {
                alt7=1;
                }
                break;
            case Annotation:
                {
                int LA7_2 = input.LA(2);

                if ( (synpred7_FairMLParserRules()) ) {
                    alt7=2;
                }
                else if ( (synpred9_FairMLParserRules()) ) {
                    alt7=3;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return retval;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 7, 2, input);

                    throw nvae;
                }
                }
                break;
            case 114:
                {
                int LA7_3 = input.LA(2);

                if ( (synpred7_FairMLParserRules()) ) {
                    alt7=2;
                }
                else if ( (synpred9_FairMLParserRules()) ) {
                    alt7=3;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return retval;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 7, 3, input);

                    throw nvae;
                }
                }
                break;
            case 178:
                {
                alt7=2;
                }
                break;
            case 179:
                {
                alt7=3;
                }
                break;
            case 174:
                {
                alt7=4;
                }
                break;
            default:
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 7, 0, input);

                throw nvae;
            }

            switch (alt7) {
                case 1 :
                    // FairMLParserRules.g:77:5: reference
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_reference_in_columnGenerator169);
                    reference8=reference();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, reference8.getTree());

                    }
                    break;
                case 2 :
                    // FairMLParserRules.g:78:5: ( annotationBlock )? column
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    // FairMLParserRules.g:78:5: ( annotationBlock )?
                    int alt5=2;
                    int LA5_0 = input.LA(1);

                    if ( (LA5_0==Annotation||LA5_0==114) ) {
                        alt5=1;
                    }
                    switch (alt5) {
                        case 1 :
                            // FairMLParserRules.g:0:0: annotationBlock
                            {
                            pushFollow(FOLLOW_annotationBlock_in_columnGenerator177);
                            annotationBlock9=gFairML.annotationBlock();

                            state._fsp--;
                            if (state.failed) return retval;
                            if ( state.backtracking==0 ) adaptor.addChild(root_0, annotationBlock9.getTree());

                            }
                            break;

                    }

                    pushFollow(FOLLOW_column_in_columnGenerator180);
                    column10=column();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, column10.getTree());

                    }
                    break;
                case 3 :
                    // FairMLParserRules.g:79:5: ( annotationBlock )? grid
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    // FairMLParserRules.g:79:5: ( annotationBlock )?
                    int alt6=2;
                    int LA6_0 = input.LA(1);

                    if ( (LA6_0==Annotation||LA6_0==114) ) {
                        alt6=1;
                    }
                    switch (alt6) {
                        case 1 :
                            // FairMLParserRules.g:0:0: annotationBlock
                            {
                            pushFollow(FOLLOW_annotationBlock_in_columnGenerator188);
                            annotationBlock11=gFairML.annotationBlock();

                            state._fsp--;
                            if (state.failed) return retval;
                            if ( state.backtracking==0 ) adaptor.addChild(root_0, annotationBlock11.getTree());

                            }
                            break;

                    }

                    pushFollow(FOLLOW_grid_in_columnGenerator191);
                    grid12=grid();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, grid12.getTree());

                    }
                    break;
                case 4 :
                    // FairMLParserRules.g:80:5: nestedFrom
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_nestedFrom_in_columnGenerator199);
                    nestedFrom13=nestedFrom();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, nestedFrom13.getTree());

                    }
                    break;

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
    // $ANTLR end columnGenerator

    public static class nestedFrom_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start nestedFrom
    // FairMLParserRules.g:83:1: nestedFrom : nf= 'from' NAME expressionOrStatementBlock '{' ( properties )? ( columnGenerator )* '}' ;
    public final FairML_FairMLParserRules.nestedFrom_return nestedFrom() throws RecognitionException {
        FairML_FairMLParserRules.nestedFrom_return retval = new FairML_FairMLParserRules.nestedFrom_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token nf=null;
        Token NAME14=null;
        Token char_literal16=null;
        Token char_literal19=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock15 = null;

        FairML_FairMLParserRules.properties_return properties17 = null;

        FairML_FairMLParserRules.columnGenerator_return columnGenerator18 = null;


        org.eclipse.epsilon.common.parse.AST nf_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME14_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal16_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal19_tree=null;

        try {
            // FairMLParserRules.g:84:3: (nf= 'from' NAME expressionOrStatementBlock '{' ( properties )? ( columnGenerator )* '}' )
            // FairMLParserRules.g:84:5: nf= 'from' NAME expressionOrStatementBlock '{' ( properties )? ( columnGenerator )* '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            nf=(Token)match(input,174,FOLLOW_174_in_nestedFrom214); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            nf_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(nf);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(nf_tree, root_0);
            }
            NAME14=(Token)match(input,NAME,FOLLOW_NAME_in_nestedFrom217); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME14_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME14);
            adaptor.addChild(root_0, NAME14_tree);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_nestedFrom219);
            expressionOrStatementBlock15=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock15.getTree());
            char_literal16=(Token)match(input,105,FOLLOW_105_in_nestedFrom221); if (state.failed) return retval;
            // FairMLParserRules.g:85:5: ( properties )?
            int alt8=2;
            int LA8_0 = input.LA(1);

            if ( (LA8_0==176) ) {
                alt8=1;
            }
            switch (alt8) {
                case 1 :
                    // FairMLParserRules.g:0:0: properties
                    {
                    pushFollow(FOLLOW_properties_in_nestedFrom228);
                    properties17=properties();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, properties17.getTree());

                    }
                    break;

            }

            // FairMLParserRules.g:86:5: ( columnGenerator )*
            loop9:
            do {
                int alt9=2;
                int LA9_0 = input.LA(1);

                if ( (LA9_0==Annotation||LA9_0==114||LA9_0==174||(LA9_0>=177 && LA9_0<=179)) ) {
                    alt9=1;
                }


                switch (alt9) {
            	case 1 :
            	    // FairMLParserRules.g:0:0: columnGenerator
            	    {
            	    pushFollow(FOLLOW_columnGenerator_in_nestedFrom235);
            	    columnGenerator18=columnGenerator();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, columnGenerator18.getTree());

            	    }
            	    break;

            	default :
            	    break loop9;
                }
            } while (true);

            char_literal19=(Token)match(input,106,FOLLOW_106_in_nestedFrom240); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              nf.setType(NESTEDFROM);
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
    // $ANTLR end nestedFrom

    public static class nameslist_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start nameslist
    // FairMLParserRules.g:91:1: nameslist : nl= '[' aliasedName ( ',' aliasedName )* cb= ']' ;
    public final FairML_FairMLParserRules.nameslist_return nameslist() throws RecognitionException {
        FairML_FairMLParserRules.nameslist_return retval = new FairML_FairMLParserRules.nameslist_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token nl=null;
        Token cb=null;
        Token char_literal21=null;
        FairML_FairMLParserRules.aliasedName_return aliasedName20 = null;

        FairML_FairMLParserRules.aliasedName_return aliasedName22 = null;


        org.eclipse.epsilon.common.parse.AST nl_tree=null;
        org.eclipse.epsilon.common.parse.AST cb_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal21_tree=null;

        try {
            // FairMLParserRules.g:95:2: (nl= '[' aliasedName ( ',' aliasedName )* cb= ']' )
            // FairMLParserRules.g:95:4: nl= '[' aliasedName ( ',' aliasedName )* cb= ']'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            nl=(Token)match(input,161,FOLLOW_161_in_nameslist267); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            nl_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(nl);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(nl_tree, root_0);
            }
            pushFollow(FOLLOW_aliasedName_in_nameslist270);
            aliasedName20=aliasedName();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, aliasedName20.getTree());
            // FairMLParserRules.g:95:24: ( ',' aliasedName )*
            loop10:
            do {
                int alt10=2;
                int LA10_0 = input.LA(1);

                if ( (LA10_0==103) ) {
                    alt10=1;
                }


                switch (alt10) {
            	case 1 :
            	    // FairMLParserRules.g:95:25: ',' aliasedName
            	    {
            	    char_literal21=(Token)match(input,103,FOLLOW_103_in_nameslist273); if (state.failed) return retval;
            	    pushFollow(FOLLOW_aliasedName_in_nameslist276);
            	    aliasedName22=aliasedName();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, aliasedName22.getTree());

            	    }
            	    break;

            	default :
            	    break loop10;
                }
            } while (true);

            cb=(Token)match(input,162,FOLLOW_162_in_nameslist282); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              nl.setType(NAMESLIST);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

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
    // $ANTLR end nameslist

    public static class aliasedName_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start aliasedName
    // FairMLParserRules.g:99:1: aliasedName : an= NAME ( 'as' NAME )? ;
    public final FairML_FairMLParserRules.aliasedName_return aliasedName() throws RecognitionException {
        FairML_FairMLParserRules.aliasedName_return retval = new FairML_FairMLParserRules.aliasedName_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token an=null;
        Token string_literal23=null;
        Token NAME24=null;

        org.eclipse.epsilon.common.parse.AST an_tree=null;
        org.eclipse.epsilon.common.parse.AST string_literal23_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME24_tree=null;

        try {
            // FairMLParserRules.g:100:2: (an= NAME ( 'as' NAME )? )
            // FairMLParserRules.g:100:4: an= NAME ( 'as' NAME )?
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            an=(Token)match(input,NAME,FOLLOW_NAME_in_aliasedName301); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            an_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(an);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(an_tree, root_0);
            }
            // FairMLParserRules.g:100:13: ( 'as' NAME )?
            int alt11=2;
            int LA11_0 = input.LA(1);

            if ( (LA11_0==175) ) {
                alt11=1;
            }
            switch (alt11) {
                case 1 :
                    // FairMLParserRules.g:100:14: 'as' NAME
                    {
                    string_literal23=(Token)match(input,175,FOLLOW_175_in_aliasedName305); if (state.failed) return retval;
                    NAME24=(Token)match(input,NAME,FOLLOW_NAME_in_aliasedName308); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    NAME24_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME24);
                    adaptor.addChild(root_0, NAME24_tree);
                    }

                    }
                    break;

            }

            if ( state.backtracking==0 ) {
              an.setType(ALIASEDNAME);
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
    // $ANTLR end aliasedName

    public static class properties_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start properties
    // FairMLParserRules.g:104:1: properties : sf= 'properties' nameslist ;
    public final FairML_FairMLParserRules.properties_return properties() throws RecognitionException {
        FairML_FairMLParserRules.properties_return retval = new FairML_FairMLParserRules.properties_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token sf=null;
        FairML_FairMLParserRules.nameslist_return nameslist25 = null;


        org.eclipse.epsilon.common.parse.AST sf_tree=null;

        try {
            // FairMLParserRules.g:105:3: (sf= 'properties' nameslist )
            // FairMLParserRules.g:106:3: sf= 'properties' nameslist
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            sf=(Token)match(input,176,FOLLOW_176_in_properties331); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            sf_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(sf);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(sf_tree, root_0);
            }
            pushFollow(FOLLOW_nameslist_in_properties334);
            nameslist25=nameslist();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, nameslist25.getTree());
            if ( state.backtracking==0 ) {
              sf.setType(PROPERTIES);
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
    // $ANTLR end properties

    public static class reference_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start reference
    // FairMLParserRules.g:110:1: reference : r= 'reference' NAME nameslist ;
    public final FairML_FairMLParserRules.reference_return reference() throws RecognitionException {
        FairML_FairMLParserRules.reference_return retval = new FairML_FairMLParserRules.reference_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token r=null;
        Token NAME26=null;
        FairML_FairMLParserRules.nameslist_return nameslist27 = null;


        org.eclipse.epsilon.common.parse.AST r_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME26_tree=null;

        try {
            // FairMLParserRules.g:111:3: (r= 'reference' NAME nameslist )
            // FairMLParserRules.g:111:5: r= 'reference' NAME nameslist
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            r=(Token)match(input,177,FOLLOW_177_in_reference353); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            r_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(r);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(r_tree, root_0);
            }
            NAME26=(Token)match(input,NAME,FOLLOW_NAME_in_reference356); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME26_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME26);
            adaptor.addChild(root_0, NAME26_tree);
            }
            pushFollow(FOLLOW_nameslist_in_reference358);
            nameslist27=nameslist();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, nameslist27.getTree());
            if ( state.backtracking==0 ) {
              r.setType(REFERENCE);
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
    // $ANTLR end reference

    public static class column_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start column
    // FairMLParserRules.g:115:1: column : cd= 'column' NAME expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.column_return column() throws RecognitionException {
        FairML_FairMLParserRules.column_return retval = new FairML_FairMLParserRules.column_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token cd=null;
        Token NAME28=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock29 = null;


        org.eclipse.epsilon.common.parse.AST cd_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME28_tree=null;

        try {
            // FairMLParserRules.g:116:3: (cd= 'column' NAME expressionOrStatementBlock )
            // FairMLParserRules.g:116:5: cd= 'column' NAME expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            cd=(Token)match(input,178,FOLLOW_178_in_column377); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            cd_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(cd);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(cd_tree, root_0);
            }
            NAME28=(Token)match(input,NAME,FOLLOW_NAME_in_column380); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME28_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME28);
            adaptor.addChild(root_0, NAME28_tree);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_column382);
            expressionOrStatementBlock29=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock29.getTree());
            if ( state.backtracking==0 ) {
              cd.setType(COLUMN);
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
    // $ANTLR end column

    public static class grid_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start grid
    // FairMLParserRules.g:120:1: grid : cd= 'grid' ob= '{' gkeys header gbody cb= '}' ;
    public final FairML_FairMLParserRules.grid_return grid() throws RecognitionException {
        FairML_FairMLParserRules.grid_return retval = new FairML_FairMLParserRules.grid_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token cd=null;
        Token ob=null;
        Token cb=null;
        FairML_FairMLParserRules.gkeys_return gkeys30 = null;

        FairML_FairMLParserRules.header_return header31 = null;

        FairML_FairMLParserRules.gbody_return gbody32 = null;


        org.eclipse.epsilon.common.parse.AST cd_tree=null;
        org.eclipse.epsilon.common.parse.AST ob_tree=null;
        org.eclipse.epsilon.common.parse.AST cb_tree=null;

        try {
            // FairMLParserRules.g:125:3: (cd= 'grid' ob= '{' gkeys header gbody cb= '}' )
            // FairMLParserRules.g:125:5: cd= 'grid' ob= '{' gkeys header gbody cb= '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            cd=(Token)match(input,179,FOLLOW_179_in_grid408); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            cd_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(cd);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(cd_tree, root_0);
            }
            ob=(Token)match(input,105,FOLLOW_105_in_grid413); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            ob_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(ob);
            adaptor.addChild(root_0, ob_tree);
            }
            pushFollow(FOLLOW_gkeys_in_grid419);
            gkeys30=gkeys();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, gkeys30.getTree());
            pushFollow(FOLLOW_header_in_grid425);
            header31=header();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, header31.getTree());
            pushFollow(FOLLOW_gbody_in_grid431);
            gbody32=gbody();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, gbody32.getTree());
            cb=(Token)match(input,106,FOLLOW_106_in_grid437); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            cb_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(cb);
            adaptor.addChild(root_0, cb_tree);
            }
            if ( state.backtracking==0 ) {
              cd.setType(GRID);
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
    // $ANTLR end grid

    public static class gkeys_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start gkeys
    // FairMLParserRules.g:133:1: gkeys : gk= 'keys' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.gkeys_return gkeys() throws RecognitionException {
        FairML_FairMLParserRules.gkeys_return retval = new FairML_FairMLParserRules.gkeys_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token gk=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock33 = null;


        org.eclipse.epsilon.common.parse.AST gk_tree=null;

        try {
            // FairMLParserRules.g:134:3: (gk= 'keys' expressionOrStatementBlock )
            // FairMLParserRules.g:134:5: gk= 'keys' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            gk=(Token)match(input,180,FOLLOW_180_in_gkeys456); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            gk_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(gk);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(gk_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_gkeys459);
            expressionOrStatementBlock33=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock33.getTree());
            if ( state.backtracking==0 ) {
              gk.setType(GRIDKEYS);
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
    // $ANTLR end gkeys

    public static class header_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start header
    // FairMLParserRules.g:138:1: header : gh= 'header' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.header_return header() throws RecognitionException {
        FairML_FairMLParserRules.header_return retval = new FairML_FairMLParserRules.header_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token gh=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock34 = null;


        org.eclipse.epsilon.common.parse.AST gh_tree=null;

        try {
            // FairMLParserRules.g:139:3: (gh= 'header' expressionOrStatementBlock )
            // FairMLParserRules.g:139:5: gh= 'header' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            gh=(Token)match(input,181,FOLLOW_181_in_header478); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            gh_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(gh);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(gh_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_header481);
            expressionOrStatementBlock34=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock34.getTree());
            if ( state.backtracking==0 ) {
              gh.setType(GRIDHEADER);
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
    // $ANTLR end header

    public static class gbody_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start gbody
    // FairMLParserRules.g:143:1: gbody : gb= 'body' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.gbody_return gbody() throws RecognitionException {
        FairML_FairMLParserRules.gbody_return retval = new FairML_FairMLParserRules.gbody_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token gb=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock35 = null;


        org.eclipse.epsilon.common.parse.AST gb_tree=null;

        try {
            // FairMLParserRules.g:144:3: (gb= 'body' expressionOrStatementBlock )
            // FairMLParserRules.g:144:5: gb= 'body' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            gb=(Token)match(input,182,FOLLOW_182_in_gbody500); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            gb_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(gb);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(gb_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_gbody503);
            expressionOrStatementBlock35=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock35.getTree());
            if ( state.backtracking==0 ) {
              gb.setType(GRIDBODY);
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
    // $ANTLR end gbody

    public static class from_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        public Object getTree() { return tree; }
    };

    // $ANTLR start from
    // FairMLParserRules.g:148:1: from : ff= 'from' expressionOrStatementBlock ;
    public final FairML_FairMLParserRules.from_return from() throws RecognitionException {
        FairML_FairMLParserRules.from_return retval = new FairML_FairMLParserRules.from_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token ff=null;
        FairML_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock36 = null;


        org.eclipse.epsilon.common.parse.AST ff_tree=null;

        try {
            // FairMLParserRules.g:149:3: (ff= 'from' expressionOrStatementBlock )
            // FairMLParserRules.g:149:5: ff= 'from' expressionOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            ff=(Token)match(input,174,FOLLOW_174_in_from522); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            ff_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(ff);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(ff_tree, root_0);
            }
            pushFollow(FOLLOW_expressionOrStatementBlock_in_from525);
            expressionOrStatementBlock36=gFairML.expressionOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionOrStatementBlock36.getTree());
            if ( state.backtracking==0 ) {
              ff.setType(FROM);
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
    // $ANTLR end from

    // $ANTLR start synpred7_FairMLParserRules
    public final void synpred7_FairMLParserRules_fragment() throws RecognitionException {   
        // FairMLParserRules.g:78:5: ( ( annotationBlock )? column )
        // FairMLParserRules.g:78:5: ( annotationBlock )? column
        {
        // FairMLParserRules.g:78:5: ( annotationBlock )?
        int alt12=2;
        int LA12_0 = input.LA(1);

        if ( (LA12_0==Annotation||LA12_0==114) ) {
            alt12=1;
        }
        switch (alt12) {
            case 1 :
                // FairMLParserRules.g:0:0: annotationBlock
                {
                pushFollow(FOLLOW_annotationBlock_in_synpred7_FairMLParserRules177);
                gFairML.annotationBlock();

                state._fsp--;
                if (state.failed) return ;

                }
                break;

        }

        pushFollow(FOLLOW_column_in_synpred7_FairMLParserRules180);
        column();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred7_FairMLParserRules

    // $ANTLR start synpred9_FairMLParserRules
    public final void synpred9_FairMLParserRules_fragment() throws RecognitionException {   
        // FairMLParserRules.g:79:5: ( ( annotationBlock )? grid )
        // FairMLParserRules.g:79:5: ( annotationBlock )? grid
        {
        // FairMLParserRules.g:79:5: ( annotationBlock )?
        int alt13=2;
        int LA13_0 = input.LA(1);

        if ( (LA13_0==Annotation||LA13_0==114) ) {
            alt13=1;
        }
        switch (alt13) {
            case 1 :
                // FairMLParserRules.g:0:0: annotationBlock
                {
                pushFollow(FOLLOW_annotationBlock_in_synpred9_FairMLParserRules188);
                gFairML.annotationBlock();

                state._fsp--;
                if (state.failed) return ;

                }
                break;

        }

        pushFollow(FOLLOW_grid_in_synpred9_FairMLParserRules191);
        grid();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred9_FairMLParserRules

    // Delegated rules

    public final boolean synpred7_FairMLParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred7_FairMLParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred9_FairMLParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred9_FairMLParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }


 

    public static final BitSet FOLLOW_172_in_datasetRule108 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_datasetRule111 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0000200000000000L});
    public static final BitSet FOLLOW_173_in_datasetRule113 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_formalParameter_in_datasetRule116 = new BitSet(new long[]{0x0000000000000000L,0x0000020000000000L,0x0000400000000000L});
    public static final BitSet FOLLOW_from_in_datasetRule118 = new BitSet(new long[]{0x0000000000000000L,0x0000020000000000L});
    public static final BitSet FOLLOW_105_in_datasetRule123 = new BitSet(new long[]{0x0000000008000000L,0x0004040000000000L,0x000F440000000000L});
    public static final BitSet FOLLOW_guard_in_datasetRule130 = new BitSet(new long[]{0x0000000008000000L,0x0004040000000000L,0x000F400000000000L});
    public static final BitSet FOLLOW_properties_in_datasetRule137 = new BitSet(new long[]{0x0000000008000000L,0x0004040000000000L,0x000E400000000000L});
    public static final BitSet FOLLOW_columnGenerator_in_datasetRule144 = new BitSet(new long[]{0x0000000008000000L,0x0004040000000000L,0x000E400000000000L});
    public static final BitSet FOLLOW_106_in_datasetRule151 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_reference_in_columnGenerator169 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_annotationBlock_in_columnGenerator177 = new BitSet(new long[]{0x0000000008000000L,0x0004000000000000L,0x0004000000000000L});
    public static final BitSet FOLLOW_column_in_columnGenerator180 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_annotationBlock_in_columnGenerator188 = new BitSet(new long[]{0x0000000008000000L,0x0004000000000000L,0x0008000000000000L});
    public static final BitSet FOLLOW_grid_in_columnGenerator191 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_nestedFrom_in_columnGenerator199 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_174_in_nestedFrom214 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_nestedFrom217 = new BitSet(new long[]{0x0000000000000000L,0x0001020000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_nestedFrom219 = new BitSet(new long[]{0x0000000000000000L,0x0000020000000000L});
    public static final BitSet FOLLOW_105_in_nestedFrom221 = new BitSet(new long[]{0x0000000008000000L,0x0004040000000000L,0x000F400000000000L});
    public static final BitSet FOLLOW_properties_in_nestedFrom228 = new BitSet(new long[]{0x0000000008000000L,0x0004040000000000L,0x000E400000000000L});
    public static final BitSet FOLLOW_columnGenerator_in_nestedFrom235 = new BitSet(new long[]{0x0000000008000000L,0x0004040000000000L,0x000E400000000000L});
    public static final BitSet FOLLOW_106_in_nestedFrom240 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_161_in_nameslist267 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_aliasedName_in_nameslist270 = new BitSet(new long[]{0x0000000000000000L,0x0000008000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_103_in_nameslist273 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_aliasedName_in_nameslist276 = new BitSet(new long[]{0x0000000000000000L,0x0000008000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_162_in_nameslist282 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_NAME_in_aliasedName301 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_175_in_aliasedName305 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_aliasedName308 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_176_in_properties331 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0000000200000000L});
    public static final BitSet FOLLOW_nameslist_in_properties334 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_177_in_reference353 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_reference356 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0000000200000000L});
    public static final BitSet FOLLOW_nameslist_in_reference358 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_178_in_column377 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_column380 = new BitSet(new long[]{0x0000000000000000L,0x0001020000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_column382 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_179_in_grid408 = new BitSet(new long[]{0x0000000000000000L,0x0000020000000000L});
    public static final BitSet FOLLOW_105_in_grid413 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0010000000000000L});
    public static final BitSet FOLLOW_gkeys_in_grid419 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0020000000000000L});
    public static final BitSet FOLLOW_header_in_grid425 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0040000000000000L});
    public static final BitSet FOLLOW_gbody_in_grid431 = new BitSet(new long[]{0x0000000000000000L,0x0000040000000000L});
    public static final BitSet FOLLOW_106_in_grid437 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_180_in_gkeys456 = new BitSet(new long[]{0x0000000000000000L,0x0001020000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_gkeys459 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_181_in_header478 = new BitSet(new long[]{0x0000000000000000L,0x0001020000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_header481 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_182_in_gbody500 = new BitSet(new long[]{0x0000000000000000L,0x0001020000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_gbody503 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_174_in_from522 = new BitSet(new long[]{0x0000000000000000L,0x0001020000000000L});
    public static final BitSet FOLLOW_expressionOrStatementBlock_in_from525 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_annotationBlock_in_synpred7_FairMLParserRules177 = new BitSet(new long[]{0x0000000008000000L,0x0004000000000000L,0x0004000000000000L});
    public static final BitSet FOLLOW_column_in_synpred7_FairMLParserRules180 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_annotationBlock_in_synpred9_FairMLParserRules188 = new BitSet(new long[]{0x0000000008000000L,0x0004000000000000L,0x0008000000000000L});
    public static final BitSet FOLLOW_grid_in_synpred9_FairMLParserRules191 = new BitSet(new long[]{0x0000000000000002L});

}
