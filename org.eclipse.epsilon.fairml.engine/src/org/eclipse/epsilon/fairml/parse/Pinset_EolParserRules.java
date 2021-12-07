package org.eclipse.epsilon.fairml.parse;

// $ANTLR 3.1b1 EolParserRules.g 2020-06-29 12:42:03

import org.antlr.runtime.*;
import org.antlr.runtime.tree.*;

/*******************************************************************************
 * Copyright (c) 2008 The University of York.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v20.html
 * 
 * Contributors:
 *     Dimitrios Kolovos - initial API and implementation
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
public class Pinset_EolParserRules extends org.eclipse.epsilon.common.parse.EpsilonParser {
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
    public static final int PINSETMODULE=99;
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
    public static final int DATASET=87;
    public static final int FROM=97;
    public static final int KEYVAL=81;
    public static final int POINT_POINT=10;
    public static final int GUARD=86;
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
    public PinsetParser gPinset;


        public Pinset_EolParserRules(TokenStream input, PinsetParser gPinset) {
            this(input, new RecognizerSharedState(), gPinset);
        }
        public Pinset_EolParserRules(TokenStream input, RecognizerSharedState state, PinsetParser gPinset) {
            super(input, state);
            this.gPinset = gPinset;
        }
        
    protected TreeAdaptor adaptor = new CommonTreeAdaptor();

    @Override
	public void setTreeAdaptor(TreeAdaptor adaptor) {
        this.adaptor = adaptor;
    }
    @Override
	public TreeAdaptor getTreeAdaptor() {
        return adaptor;
    }

    @Override
	public String[] getTokenNames() { return PinsetParser.tokenNames; }
    @Override
	public String getGrammarFileName() { return "EolParserRules.g"; }



    public void setTokenType(ParserRuleReturnScope tree, int type) {
    	((CommonTree) tree.getTree()).getToken().setType(type);
    }



    public static class operationDeclarationOrAnnotationBlock_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start operationDeclarationOrAnnotationBlock
    // EolParserRules.g:107:1: operationDeclarationOrAnnotationBlock : ( operationDeclaration | annotationBlock );
    public final Pinset_EolParserRules.operationDeclarationOrAnnotationBlock_return operationDeclarationOrAnnotationBlock() throws RecognitionException {
        Pinset_EolParserRules.operationDeclarationOrAnnotationBlock_return retval = new Pinset_EolParserRules.operationDeclarationOrAnnotationBlock_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.operationDeclaration_return operationDeclaration1 = null;

        Pinset_EolParserRules.annotationBlock_return annotationBlock2 = null;



        try {
            // EolParserRules.g:108:2: ( operationDeclaration | annotationBlock )
            int alt1=2;
            int LA1_0 = input.LA(1);

            if ( ((LA1_0>=108 && LA1_0<=109)) ) {
                alt1=1;
            }
            else if ( (LA1_0==Annotation||LA1_0==114) ) {
                alt1=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 1, 0, input);

                throw nvae;
            }
            switch (alt1) {
                case 1 :
                    // EolParserRules.g:108:4: operationDeclaration
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_operationDeclaration_in_operationDeclarationOrAnnotationBlock263);
                    operationDeclaration1=operationDeclaration();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, operationDeclaration1.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:108:27: annotationBlock
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_annotationBlock_in_operationDeclarationOrAnnotationBlock267);
                    annotationBlock2=annotationBlock();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, annotationBlock2.getTree());

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
    // $ANTLR end operationDeclarationOrAnnotationBlock

    public static class modelDeclaration_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start modelDeclaration
    // EolParserRules.g:111:1: modelDeclaration : m= 'model' NAME ( modelAlias )? ( modelDriver )? ( modelDeclarationParameters )? sem= ';' ;
    public final Pinset_EolParserRules.modelDeclaration_return modelDeclaration() throws RecognitionException {
        Pinset_EolParserRules.modelDeclaration_return retval = new Pinset_EolParserRules.modelDeclaration_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token m=null;
        Token sem=null;
        Token NAME3=null;
        Pinset_EolParserRules.modelAlias_return modelAlias4 = null;

        Pinset_EolParserRules.modelDriver_return modelDriver5 = null;

        Pinset_EolParserRules.modelDeclarationParameters_return modelDeclarationParameters6 = null;


        org.eclipse.epsilon.common.parse.AST m_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME3_tree=null;

        try {
            // EolParserRules.g:116:2: (m= 'model' NAME ( modelAlias )? ( modelDriver )? ( modelDeclarationParameters )? sem= ';' )
            // EolParserRules.g:116:4: m= 'model' NAME ( modelAlias )? ( modelDriver )? ( modelDeclarationParameters )? sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            m=(Token)match(input,100,FOLLOW_100_in_modelDeclaration286); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            m_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(m);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(m_tree, root_0);
            }
            NAME3=(Token)match(input,NAME,FOLLOW_NAME_in_modelDeclaration289); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME3_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME3);
            adaptor.addChild(root_0, NAME3_tree);
            }
            // EolParserRules.g:116:20: ( modelAlias )?
            int alt2=2;
            int LA2_0 = input.LA(1);

            if ( (LA2_0==102) ) {
                alt2=1;
            }
            switch (alt2) {
                case 1 :
                    // EolParserRules.g:0:0: modelAlias
                    {
                    pushFollow(FOLLOW_modelAlias_in_modelDeclaration291);
                    modelAlias4=modelAlias();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, modelAlias4.getTree());

                    }
                    break;

            }

            // EolParserRules.g:116:32: ( modelDriver )?
            int alt3=2;
            int LA3_0 = input.LA(1);

            if ( (LA3_0==104) ) {
                alt3=1;
            }
            switch (alt3) {
                case 1 :
                    // EolParserRules.g:0:0: modelDriver
                    {
                    pushFollow(FOLLOW_modelDriver_in_modelDeclaration294);
                    modelDriver5=modelDriver();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, modelDriver5.getTree());

                    }
                    break;

            }

            // EolParserRules.g:116:45: ( modelDeclarationParameters )?
            int alt4=2;
            int LA4_0 = input.LA(1);

            if ( (LA4_0==105) ) {
                alt4=1;
            }
            switch (alt4) {
                case 1 :
                    // EolParserRules.g:0:0: modelDeclarationParameters
                    {
                    pushFollow(FOLLOW_modelDeclarationParameters_in_modelDeclaration297);
                    modelDeclarationParameters6=modelDeclarationParameters();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, modelDeclarationParameters6.getTree());

                    }
                    break;

            }

            sem=(Token)match(input,101,FOLLOW_101_in_modelDeclaration302); if (state.failed) return retval;

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              		retval.tree.getToken().setType(MODELDECLARATION);
              	
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
    // $ANTLR end modelDeclaration

    public static class modelAlias_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start modelAlias
    // EolParserRules.g:119:1: modelAlias : a= 'alias' NAME ( ',' NAME )* ;
    public final Pinset_EolParserRules.modelAlias_return modelAlias() throws RecognitionException {
        Pinset_EolParserRules.modelAlias_return retval = new Pinset_EolParserRules.modelAlias_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token a=null;
        Token NAME7=null;
        Token char_literal8=null;
        Token NAME9=null;

        org.eclipse.epsilon.common.parse.AST a_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME7_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal8_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME9_tree=null;

        try {
            // EolParserRules.g:120:2: (a= 'alias' NAME ( ',' NAME )* )
            // EolParserRules.g:120:5: a= 'alias' NAME ( ',' NAME )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            a=(Token)match(input,102,FOLLOW_102_in_modelAlias317); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            a_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(a);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(a_tree, root_0);
            }
            NAME7=(Token)match(input,NAME,FOLLOW_NAME_in_modelAlias320); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME7_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME7);
            adaptor.addChild(root_0, NAME7_tree);
            }
            // EolParserRules.g:120:21: ( ',' NAME )*
            loop5:
            do {
                int alt5=2;
                int LA5_0 = input.LA(1);

                if ( (LA5_0==103) ) {
                    alt5=1;
                }


                switch (alt5) {
            	case 1 :
            	    // EolParserRules.g:120:22: ',' NAME
            	    {
            	    char_literal8=(Token)match(input,103,FOLLOW_103_in_modelAlias323); if (state.failed) return retval;
            	    NAME9=(Token)match(input,NAME,FOLLOW_NAME_in_modelAlias326); if (state.failed) return retval;
            	    if ( state.backtracking==0 ) {
            	    NAME9_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME9);
            	    adaptor.addChild(root_0, NAME9_tree);
            	    }

            	    }
            	    break;

            	default :
            	    break loop5;
                }
            } while (true);

            if ( state.backtracking==0 ) {
              a.setType(ALIAS);
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
    // $ANTLR end modelAlias

    public static class modelDriver_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start modelDriver
    // EolParserRules.g:124:1: modelDriver : d= 'driver' NAME ;
    public final Pinset_EolParserRules.modelDriver_return modelDriver() throws RecognitionException {
        Pinset_EolParserRules.modelDriver_return retval = new Pinset_EolParserRules.modelDriver_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token d=null;
        Token NAME10=null;

        org.eclipse.epsilon.common.parse.AST d_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME10_tree=null;

        try {
            // EolParserRules.g:125:2: (d= 'driver' NAME )
            // EolParserRules.g:125:5: d= 'driver' NAME
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            d=(Token)match(input,104,FOLLOW_104_in_modelDriver345); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            d_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(d);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(d_tree, root_0);
            }
            NAME10=(Token)match(input,NAME,FOLLOW_NAME_in_modelDriver348); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME10_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME10);
            adaptor.addChild(root_0, NAME10_tree);
            }
            if ( state.backtracking==0 ) {
              d.setType(DRIVER);
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
    // $ANTLR end modelDriver

    public static class modelDeclarationParameters_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start modelDeclarationParameters
    // EolParserRules.g:129:1: modelDeclarationParameters : s= '{' ( modelDeclarationParameter )? ( ',' modelDeclarationParameter )* cb= '}' ;
    public final Pinset_EolParserRules.modelDeclarationParameters_return modelDeclarationParameters() throws RecognitionException {
        Pinset_EolParserRules.modelDeclarationParameters_return retval = new Pinset_EolParserRules.modelDeclarationParameters_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token s=null;
        Token cb=null;
        Token char_literal12=null;
        Pinset_EolParserRules.modelDeclarationParameter_return modelDeclarationParameter11 = null;

        Pinset_EolParserRules.modelDeclarationParameter_return modelDeclarationParameter13 = null;


        org.eclipse.epsilon.common.parse.AST s_tree=null;
        org.eclipse.epsilon.common.parse.AST cb_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal12_tree=null;

        try {
            // EolParserRules.g:134:2: (s= '{' ( modelDeclarationParameter )? ( ',' modelDeclarationParameter )* cb= '}' )
            // EolParserRules.g:134:4: s= '{' ( modelDeclarationParameter )? ( ',' modelDeclarationParameter )* cb= '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            s=(Token)match(input,105,FOLLOW_105_in_modelDeclarationParameters370); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            s_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(s);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(s_tree, root_0);
            }
            // EolParserRules.g:134:11: ( modelDeclarationParameter )?
            int alt6=2;
            int LA6_0 = input.LA(1);

            if ( (LA6_0==NAME) ) {
                alt6=1;
            }
            switch (alt6) {
                case 1 :
                    // EolParserRules.g:0:0: modelDeclarationParameter
                    {
                    pushFollow(FOLLOW_modelDeclarationParameter_in_modelDeclarationParameters373);
                    modelDeclarationParameter11=modelDeclarationParameter();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, modelDeclarationParameter11.getTree());

                    }
                    break;

            }

            // EolParserRules.g:134:38: ( ',' modelDeclarationParameter )*
            loop7:
            do {
                int alt7=2;
                int LA7_0 = input.LA(1);

                if ( (LA7_0==103) ) {
                    alt7=1;
                }


                switch (alt7) {
            	case 1 :
            	    // EolParserRules.g:134:39: ',' modelDeclarationParameter
            	    {
            	    char_literal12=(Token)match(input,103,FOLLOW_103_in_modelDeclarationParameters377); if (state.failed) return retval;
            	    pushFollow(FOLLOW_modelDeclarationParameter_in_modelDeclarationParameters380);
            	    modelDeclarationParameter13=modelDeclarationParameter();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, modelDeclarationParameter13.getTree());

            	    }
            	    break;

            	default :
            	    break loop7;
                }
            } while (true);

            cb=(Token)match(input,106,FOLLOW_106_in_modelDeclarationParameters386); if (state.failed) return retval;

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(cb);
              		retval.tree.getToken().setType(MODELDECLARATIONPARAMETERS);
              	
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
    // $ANTLR end modelDeclarationParameters

    public static class modelDeclarationParameter_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start modelDeclarationParameter
    // EolParserRules.g:137:1: modelDeclarationParameter : NAME e= '=' STRING ;
    public final Pinset_EolParserRules.modelDeclarationParameter_return modelDeclarationParameter() throws RecognitionException {
        Pinset_EolParserRules.modelDeclarationParameter_return retval = new Pinset_EolParserRules.modelDeclarationParameter_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token e=null;
        Token NAME14=null;
        Token STRING15=null;

        org.eclipse.epsilon.common.parse.AST e_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME14_tree=null;
        org.eclipse.epsilon.common.parse.AST STRING15_tree=null;

        try {
            // EolParserRules.g:138:2: ( NAME e= '=' STRING )
            // EolParserRules.g:138:4: NAME e= '=' STRING
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            NAME14=(Token)match(input,NAME,FOLLOW_NAME_in_modelDeclarationParameter399); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME14_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME14);
            adaptor.addChild(root_0, NAME14_tree);
            }
            e=(Token)match(input,107,FOLLOW_107_in_modelDeclarationParameter403); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            e_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(e);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(e_tree, root_0);
            }
            STRING15=(Token)match(input,STRING,FOLLOW_STRING_in_modelDeclarationParameter406); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            STRING15_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(STRING15);
            adaptor.addChild(root_0, STRING15_tree);
            }
            if ( state.backtracking==0 ) {
              e.setType(MODELDECLARATIONPARAMETER);
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
    // $ANTLR end modelDeclarationParameter

    public static class operationDeclaration_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start operationDeclaration
    // EolParserRules.g:142:1: operationDeclaration : ( 'operation' | 'function' ) (ctx= typeName )? operationName= NAME op= '(' ( formalParameterList )? cp= ')' ( ':' returnType= typeName )? statementBlock ;
    public final Pinset_EolParserRules.operationDeclaration_return operationDeclaration() throws RecognitionException {
        Pinset_EolParserRules.operationDeclaration_return retval = new Pinset_EolParserRules.operationDeclaration_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token operationName=null;
        Token op=null;
        Token cp=null;
        Token set16=null;
        Token char_literal18=null;
        Pinset_EolParserRules.typeName_return ctx = null;

        Pinset_EolParserRules.typeName_return returnType = null;

        Pinset_EolParserRules.formalParameterList_return formalParameterList17 = null;

        Pinset_EolParserRules.statementBlock_return statementBlock19 = null;


        org.eclipse.epsilon.common.parse.AST operationName_tree=null;
        org.eclipse.epsilon.common.parse.AST op_tree=null;
        org.eclipse.epsilon.common.parse.AST cp_tree=null;
        org.eclipse.epsilon.common.parse.AST set16_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal18_tree=null;

        try {
            // EolParserRules.g:147:2: ( ( 'operation' | 'function' ) (ctx= typeName )? operationName= NAME op= '(' ( formalParameterList )? cp= ')' ( ':' returnType= typeName )? statementBlock )
            // EolParserRules.g:147:4: ( 'operation' | 'function' ) (ctx= typeName )? operationName= NAME op= '(' ( formalParameterList )? cp= ')' ( ':' returnType= typeName )? statementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            set16=input.LT(1);
            set16=input.LT(1);
            if ( (input.LA(1)>=108 && input.LA(1)<=109) ) {
                input.consume();
                if ( state.backtracking==0 ) root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(set16), root_0);
                state.errorRecovery=false;state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                throw mse;
            }

            // EolParserRules.g:147:30: (ctx= typeName )?
            int alt8=2;
            int LA8_0 = input.LA(1);

            if ( (LA8_0==NAME) ) {
                int LA8_1 = input.LA(2);

                if ( (LA8_1==NAME||(LA8_1>=115 && LA8_1<=117)) ) {
                    alt8=1;
                }
            }
            else if ( ((LA8_0>=CollectionTypeName && LA8_0<=SpecialTypeName)) ) {
                alt8=1;
            }
            switch (alt8) {
                case 1 :
                    // EolParserRules.g:147:31: ctx= typeName
                    {
                    pushFollow(FOLLOW_typeName_in_operationDeclaration437);
                    ctx=typeName();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, ctx.getTree());
                    if ( state.backtracking==0 ) {
                      setTokenType(ctx,TYPE);
                    }

                    }
                    break;

            }

            operationName=(Token)match(input,NAME,FOLLOW_NAME_in_operationDeclaration447); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            operationName_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(operationName);
            adaptor.addChild(root_0, operationName_tree);
            }
            op=(Token)match(input,110,FOLLOW_110_in_operationDeclaration451); if (state.failed) return retval;
            // EolParserRules.g:148:30: ( formalParameterList )?
            int alt9=2;
            int LA9_0 = input.LA(1);

            if ( (LA9_0==NAME) ) {
                alt9=1;
            }
            switch (alt9) {
                case 1 :
                    // EolParserRules.g:0:0: formalParameterList
                    {
                    pushFollow(FOLLOW_formalParameterList_in_operationDeclaration454);
                    formalParameterList17=formalParameterList();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, formalParameterList17.getTree());

                    }
                    break;

            }

            cp=(Token)match(input,111,FOLLOW_111_in_operationDeclaration459); if (state.failed) return retval;
            // EolParserRules.g:149:3: ( ':' returnType= typeName )?
            int alt10=2;
            int LA10_0 = input.LA(1);

            if ( (LA10_0==112) ) {
                alt10=1;
            }
            switch (alt10) {
                case 1 :
                    // EolParserRules.g:149:4: ':' returnType= typeName
                    {
                    char_literal18=(Token)match(input,112,FOLLOW_112_in_operationDeclaration465); if (state.failed) return retval;
                    pushFollow(FOLLOW_typeName_in_operationDeclaration470);
                    returnType=typeName();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, returnType.getTree());
                    if ( state.backtracking==0 ) {
                      setTokenType(returnType,TYPE);
                    }

                    }
                    break;

            }

            pushFollow(FOLLOW_statementBlock_in_operationDeclaration476);
            statementBlock19=statementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, statementBlock19.getTree());

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(cp);
              		retval.tree.getToken().setType(HELPERMETHOD);
              	
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
    // $ANTLR end operationDeclaration

    public static class importStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start importStatement
    // EolParserRules.g:152:1: importStatement : i= 'import' STRING sem= ';' ;
    public final Pinset_EolParserRules.importStatement_return importStatement() throws RecognitionException {
        Pinset_EolParserRules.importStatement_return retval = new Pinset_EolParserRules.importStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token i=null;
        Token sem=null;
        Token STRING20=null;

        org.eclipse.epsilon.common.parse.AST i_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;
        org.eclipse.epsilon.common.parse.AST STRING20_tree=null;

        try {
            // EolParserRules.g:156:2: (i= 'import' STRING sem= ';' )
            // EolParserRules.g:156:4: i= 'import' STRING sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            i=(Token)match(input,113,FOLLOW_113_in_importStatement496); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            i_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(i);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(i_tree, root_0);
            }
            STRING20=(Token)match(input,STRING,FOLLOW_STRING_in_importStatement499); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            STRING20_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(STRING20);
            adaptor.addChild(root_0, STRING20_tree);
            }
            sem=(Token)match(input,101,FOLLOW_101_in_importStatement503); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              i.setType(IMPORT);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end importStatement

    public static class block_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start block
    // EolParserRules.g:160:1: block : ( statement )* -> ^( BLOCK ( statement )* ) ;
    public final Pinset_EolParserRules.block_return block() throws RecognitionException {
        Pinset_EolParserRules.block_return retval = new Pinset_EolParserRules.block_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.statement_return statement21 = null;


        RewriteRuleSubtreeStream stream_statement=new RewriteRuleSubtreeStream(adaptor,"rule statement");
        try {
            // EolParserRules.g:164:2: ( ( statement )* -> ^( BLOCK ( statement )* ) )
            // EolParserRules.g:164:4: ( statement )*
            {
            // EolParserRules.g:164:4: ( statement )*
            loop11:
            do {
                int alt11=2;
                int LA11_0 = input.LA(1);

                if ( (LA11_0==FLOAT||LA11_0==INT||LA11_0==BOOLEAN||LA11_0==STRING||(LA11_0>=CollectionTypeName && LA11_0<=SpecialTypeName)||LA11_0==NAME||LA11_0==110||LA11_0==120||LA11_0==122||LA11_0==125||(LA11_0>=127 && LA11_0<=135)||LA11_0==152||LA11_0==155||(LA11_0>=162 && LA11_0<=164)) ) {
                    alt11=1;
                }


                switch (alt11) {
            	case 1 :
            	    // EolParserRules.g:0:0: statement
            	    {
            	    pushFollow(FOLLOW_statement_in_block524);
            	    statement21=statement();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) stream_statement.add(statement21.getTree());

            	    }
            	    break;

            	default :
            	    break loop11;
                }
            } while (true);



            // AST REWRITE
            // elements: statement
            // token labels: 
            // rule labels: retval
            // token list labels: 
            // rule list labels: 
            if ( state.backtracking==0 ) {
            retval.tree = root_0;
            RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"token retval",retval!=null?retval.tree:null);

            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
            // 165:2: -> ^( BLOCK ( statement )* )
            {
                // EolParserRules.g:165:5: ^( BLOCK ( statement )* )
                {
                org.eclipse.epsilon.common.parse.AST root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
                root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(BLOCK, "BLOCK"), root_1);

                // EolParserRules.g:165:13: ( statement )*
                while ( stream_statement.hasNext() ) {
                    adaptor.addChild(root_1, stream_statement.nextTree());

                }
                stream_statement.reset();

                adaptor.addChild(root_0, root_1);
                }

            }

            retval.tree = root_0;retval.tree = root_0;}
            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.setImaginary(true);
              	
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
    // $ANTLR end block

    public static class statementBlock_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start statementBlock
    // EolParserRules.g:168:1: statementBlock : s= '{' block e= '}' ;
    public final Pinset_EolParserRules.statementBlock_return statementBlock() throws RecognitionException {
        Pinset_EolParserRules.statementBlock_return retval = new Pinset_EolParserRules.statementBlock_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token s=null;
        Token e=null;
        Pinset_EolParserRules.block_return block22 = null;


        org.eclipse.epsilon.common.parse.AST s_tree=null;
        org.eclipse.epsilon.common.parse.AST e_tree=null;

        try {
            // EolParserRules.g:173:2: (s= '{' block e= '}' )
            // EolParserRules.g:173:4: s= '{' block e= '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            s=(Token)match(input,105,FOLLOW_105_in_statementBlock554); if (state.failed) return retval;
            pushFollow(FOLLOW_block_in_statementBlock557);
            block22=block();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, block22.getTree());
            e=(Token)match(input,106,FOLLOW_106_in_statementBlock561); if (state.failed) return retval;

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(s); 
              		retval.tree.getExtraTokens().add(e);
              	
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
    // $ANTLR end statementBlock

    public static class formalParameter_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start formalParameter
    // EolParserRules.g:176:1: formalParameter : NAME ( ':' pt= typeName )? -> ^( FORMAL NAME ( typeName )? ) ;
    public final Pinset_EolParserRules.formalParameter_return formalParameter() throws RecognitionException {
        Pinset_EolParserRules.formalParameter_return retval = new Pinset_EolParserRules.formalParameter_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token NAME23=null;
        Token char_literal24=null;
        Pinset_EolParserRules.typeName_return pt = null;


        org.eclipse.epsilon.common.parse.AST NAME23_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal24_tree=null;
        RewriteRuleTokenStream stream_112=new RewriteRuleTokenStream(adaptor,"token 112");
        RewriteRuleTokenStream stream_NAME=new RewriteRuleTokenStream(adaptor,"token NAME");
        RewriteRuleSubtreeStream stream_typeName=new RewriteRuleSubtreeStream(adaptor,"rule typeName");
        try {
            // EolParserRules.g:180:2: ( NAME ( ':' pt= typeName )? -> ^( FORMAL NAME ( typeName )? ) )
            // EolParserRules.g:180:4: NAME ( ':' pt= typeName )?
            {
            NAME23=(Token)match(input,NAME,FOLLOW_NAME_in_formalParameter579); if (state.failed) return retval; 
            if ( state.backtracking==0 ) stream_NAME.add(NAME23);

            // EolParserRules.g:180:9: ( ':' pt= typeName )?
            int alt12=2;
            int LA12_0 = input.LA(1);

            if ( (LA12_0==112) ) {
                alt12=1;
            }
            switch (alt12) {
                case 1 :
                    // EolParserRules.g:180:10: ':' pt= typeName
                    {
                    char_literal24=(Token)match(input,112,FOLLOW_112_in_formalParameter582); if (state.failed) return retval; 
                    if ( state.backtracking==0 ) stream_112.add(char_literal24);

                    pushFollow(FOLLOW_typeName_in_formalParameter586);
                    pt=typeName();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) stream_typeName.add(pt.getTree());
                    if ( state.backtracking==0 ) {
                      setTokenType(pt,TYPE);
                    }

                    }
                    break;

            }



            // AST REWRITE
            // elements: NAME, typeName
            // token labels: 
            // rule labels: retval
            // token list labels: 
            // rule list labels: 
            if ( state.backtracking==0 ) {
            retval.tree = root_0;
            RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"token retval",retval!=null?retval.tree:null);

            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
            // 181:3: -> ^( FORMAL NAME ( typeName )? )
            {
                // EolParserRules.g:181:6: ^( FORMAL NAME ( typeName )? )
                {
                org.eclipse.epsilon.common.parse.AST root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
                root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(FORMAL, "FORMAL"), root_1);

                adaptor.addChild(root_1, stream_NAME.nextNode());
                // EolParserRules.g:181:20: ( typeName )?
                if ( stream_typeName.hasNext() ) {
                    adaptor.addChild(root_1, stream_typeName.nextTree());

                }
                stream_typeName.reset();

                adaptor.addChild(root_0, root_1);
                }

            }

            retval.tree = root_0;retval.tree = root_0;}
            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              	//	((org.eclipse.epsilon.common.parse.AST)retval.tree).setImaginary(true);
              	
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
    // $ANTLR end formalParameter

    public static class formalParameterList_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start formalParameterList
    // EolParserRules.g:184:1: formalParameterList : formalParameter ( ',' formalParameter )* -> ^( PARAMLIST ( formalParameter )* ) ;
    public final Pinset_EolParserRules.formalParameterList_return formalParameterList() throws RecognitionException {
        Pinset_EolParserRules.formalParameterList_return retval = new Pinset_EolParserRules.formalParameterList_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token char_literal26=null;
        Pinset_EolParserRules.formalParameter_return formalParameter25 = null;

        Pinset_EolParserRules.formalParameter_return formalParameter27 = null;


        org.eclipse.epsilon.common.parse.AST char_literal26_tree=null;
        RewriteRuleTokenStream stream_103=new RewriteRuleTokenStream(adaptor,"token 103");
        RewriteRuleSubtreeStream stream_formalParameter=new RewriteRuleSubtreeStream(adaptor,"rule formalParameter");
        try {
            // EolParserRules.g:188:2: ( formalParameter ( ',' formalParameter )* -> ^( PARAMLIST ( formalParameter )* ) )
            // EolParserRules.g:188:4: formalParameter ( ',' formalParameter )*
            {
            pushFollow(FOLLOW_formalParameter_in_formalParameterList620);
            formalParameter25=formalParameter();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) stream_formalParameter.add(formalParameter25.getTree());
            // EolParserRules.g:188:20: ( ',' formalParameter )*
            loop13:
            do {
                int alt13=2;
                int LA13_0 = input.LA(1);

                if ( (LA13_0==103) ) {
                    alt13=1;
                }


                switch (alt13) {
            	case 1 :
            	    // EolParserRules.g:188:21: ',' formalParameter
            	    {
            	    char_literal26=(Token)match(input,103,FOLLOW_103_in_formalParameterList623); if (state.failed) return retval; 
            	    if ( state.backtracking==0 ) stream_103.add(char_literal26);

            	    pushFollow(FOLLOW_formalParameter_in_formalParameterList625);
            	    formalParameter27=formalParameter();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) stream_formalParameter.add(formalParameter27.getTree());

            	    }
            	    break;

            	default :
            	    break loop13;
                }
            } while (true);



            // AST REWRITE
            // elements: formalParameter
            // token labels: 
            // rule labels: retval
            // token list labels: 
            // rule list labels: 
            if ( state.backtracking==0 ) {
            retval.tree = root_0;
            RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"token retval",retval!=null?retval.tree:null);

            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
            // 189:2: -> ^( PARAMLIST ( formalParameter )* )
            {
                // EolParserRules.g:189:5: ^( PARAMLIST ( formalParameter )* )
                {
                org.eclipse.epsilon.common.parse.AST root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
                root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(PARAMLIST, "PARAMLIST"), root_1);

                // EolParserRules.g:189:17: ( formalParameter )*
                while ( stream_formalParameter.hasNext() ) {
                    adaptor.addChild(root_1, stream_formalParameter.nextTree());

                }
                stream_formalParameter.reset();

                adaptor.addChild(root_0, root_1);
                }

            }

            retval.tree = root_0;retval.tree = root_0;}
            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.setImaginary(true);
              	
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
    // $ANTLR end formalParameterList

    public static class executableAnnotation_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start executableAnnotation
    // EolParserRules.g:192:1: executableAnnotation : d= '$' x= . logicalExpression ;
    public final Pinset_EolParserRules.executableAnnotation_return executableAnnotation() throws RecognitionException {
        Pinset_EolParserRules.executableAnnotation_return retval = new Pinset_EolParserRules.executableAnnotation_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token d=null;
        Token x=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression28 = null;


        org.eclipse.epsilon.common.parse.AST d_tree=null;
        org.eclipse.epsilon.common.parse.AST x_tree=null;

        try {
            // EolParserRules.g:193:2: (d= '$' x= . logicalExpression )
            // EolParserRules.g:193:4: d= '$' x= . logicalExpression
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            d=(Token)match(input,114,FOLLOW_114_in_executableAnnotation650); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            d_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(d);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(d_tree, root_0);
            }
            x=input.LT(1);
            matchAny(input); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            x_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(x);
            adaptor.addChild(root_0, x_tree);
            }
            pushFollow(FOLLOW_logicalExpression_in_executableAnnotation657);
            logicalExpression28=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression28.getTree());
            if ( state.backtracking==0 ) {
              d.setType(EXECUTABLEANNOTATION);
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
    // $ANTLR end executableAnnotation

    public static class annotation_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start annotation
    // EolParserRules.g:197:1: annotation : ( Annotation | executableAnnotation );
    public final Pinset_EolParserRules.annotation_return annotation() throws RecognitionException {
        Pinset_EolParserRules.annotation_return retval = new Pinset_EolParserRules.annotation_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token Annotation29=null;
        Pinset_EolParserRules.executableAnnotation_return executableAnnotation30 = null;


        org.eclipse.epsilon.common.parse.AST Annotation29_tree=null;

        try {
            // EolParserRules.g:198:2: ( Annotation | executableAnnotation )
            int alt14=2;
            int LA14_0 = input.LA(1);

            if ( (LA14_0==Annotation) ) {
                alt14=1;
            }
            else if ( (LA14_0==114) ) {
                alt14=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 14, 0, input);

                throw nvae;
            }
            switch (alt14) {
                case 1 :
                    // EolParserRules.g:198:4: Annotation
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    Annotation29=(Token)match(input,Annotation,FOLLOW_Annotation_in_annotation671); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    Annotation29_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(Annotation29);
                    adaptor.addChild(root_0, Annotation29_tree);
                    }

                    }
                    break;
                case 2 :
                    // EolParserRules.g:198:17: executableAnnotation
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_executableAnnotation_in_annotation675);
                    executableAnnotation30=executableAnnotation();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, executableAnnotation30.getTree());

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
    // $ANTLR end annotation

    public static class annotationBlock_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start annotationBlock
    // EolParserRules.g:201:1: annotationBlock : ( annotation )+ -> ^( ANNOTATIONBLOCK ( annotation )+ ) ;
    public final Pinset_EolParserRules.annotationBlock_return annotationBlock() throws RecognitionException {
        Pinset_EolParserRules.annotationBlock_return retval = new Pinset_EolParserRules.annotationBlock_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.annotation_return annotation31 = null;


        RewriteRuleSubtreeStream stream_annotation=new RewriteRuleSubtreeStream(adaptor,"rule annotation");
        try {
            // EolParserRules.g:205:2: ( ( annotation )+ -> ^( ANNOTATIONBLOCK ( annotation )+ ) )
            // EolParserRules.g:205:4: ( annotation )+
            {
            // EolParserRules.g:205:4: ( annotation )+
            int cnt15=0;
            loop15:
            do {
                int alt15=2;
                int LA15_0 = input.LA(1);

                if ( (LA15_0==Annotation) ) {
                    int LA15_2 = input.LA(2);

                    if ( (synpred16_EolParserRules()) ) {
                        alt15=1;
                    }


                }
                else if ( (LA15_0==114) ) {
                    int LA15_3 = input.LA(2);

                    if ( (synpred16_EolParserRules()) ) {
                        alt15=1;
                    }


                }


                switch (alt15) {
            	case 1 :
            	    // EolParserRules.g:0:0: annotation
            	    {
            	    pushFollow(FOLLOW_annotation_in_annotationBlock692);
            	    annotation31=annotation();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) stream_annotation.add(annotation31.getTree());

            	    }
            	    break;

            	default :
            	    if ( cnt15 >= 1 ) break loop15;
            	    if (state.backtracking>0) {state.failed=true; return retval;}
                        EarlyExitException eee =
                            new EarlyExitException(15, input);
                        throw eee;
                }
                cnt15++;
            } while (true);



            // AST REWRITE
            // elements: annotation
            // token labels: 
            // rule labels: retval
            // token list labels: 
            // rule list labels: 
            if ( state.backtracking==0 ) {
            retval.tree = root_0;
            RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"token retval",retval!=null?retval.tree:null);

            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
            // 206:2: -> ^( ANNOTATIONBLOCK ( annotation )+ )
            {
                // EolParserRules.g:206:5: ^( ANNOTATIONBLOCK ( annotation )+ )
                {
                org.eclipse.epsilon.common.parse.AST root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
                root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(ANNOTATIONBLOCK, "ANNOTATIONBLOCK"), root_1);

                if ( !(stream_annotation.hasNext()) ) {
                    throw new RewriteEarlyExitException();
                }
                while ( stream_annotation.hasNext() ) {
                    adaptor.addChild(root_1, stream_annotation.nextTree());

                }
                stream_annotation.reset();

                adaptor.addChild(root_0, root_1);
                }

            }

            retval.tree = root_0;retval.tree = root_0;}
            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.setImaginary(true);
              	
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
    // $ANTLR end annotationBlock

    public static class typeName_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start typeName
    // EolParserRules.g:209:1: typeName : ( pathName | collectionType | specialType );
    public final Pinset_EolParserRules.typeName_return typeName() throws RecognitionException {
        Pinset_EolParserRules.typeName_return retval = new Pinset_EolParserRules.typeName_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.pathName_return pathName32 = null;

        Pinset_EolParserRules.collectionType_return collectionType33 = null;

        Pinset_EolParserRules.specialType_return specialType34 = null;



        try {
            // EolParserRules.g:213:2: ( pathName | collectionType | specialType )
            int alt16=3;
            switch ( input.LA(1) ) {
            case NAME:
                {
                alt16=1;
                }
                break;
            case CollectionTypeName:
            case MapTypeName:
                {
                alt16=2;
                }
                break;
            case SpecialTypeName:
                {
                alt16=3;
                }
                break;
            default:
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 16, 0, input);

                throw nvae;
            }

            switch (alt16) {
                case 1 :
                    // EolParserRules.g:213:4: pathName
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_pathName_in_typeName721);
                    pathName32=pathName();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, pathName32.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:213:15: collectionType
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_collectionType_in_typeName725);
                    collectionType33=collectionType();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, collectionType33.getTree());

                    }
                    break;
                case 3 :
                    // EolParserRules.g:213:32: specialType
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_specialType_in_typeName729);
                    specialType34=specialType();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, specialType34.getTree());

                    }
                    break;

            }
            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getToken().setType(TYPE);
              	
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
    // $ANTLR end typeName

    public static class specialType_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start specialType
    // EolParserRules.g:216:1: specialType : SpecialTypeName s= '(' STRING e= ')' ;
    public final Pinset_EolParserRules.specialType_return specialType() throws RecognitionException {
        Pinset_EolParserRules.specialType_return retval = new Pinset_EolParserRules.specialType_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token s=null;
        Token e=null;
        Token SpecialTypeName35=null;
        Token STRING36=null;

        org.eclipse.epsilon.common.parse.AST s_tree=null;
        org.eclipse.epsilon.common.parse.AST e_tree=null;
        org.eclipse.epsilon.common.parse.AST SpecialTypeName35_tree=null;
        org.eclipse.epsilon.common.parse.AST STRING36_tree=null;

        try {
            // EolParserRules.g:222:2: ( SpecialTypeName s= '(' STRING e= ')' )
            // EolParserRules.g:222:4: SpecialTypeName s= '(' STRING e= ')'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            SpecialTypeName35=(Token)match(input,SpecialTypeName,FOLLOW_SpecialTypeName_in_specialType746); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            SpecialTypeName35_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(SpecialTypeName35);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(SpecialTypeName35_tree, root_0);
            }
            s=(Token)match(input,110,FOLLOW_110_in_specialType751); if (state.failed) return retval;
            STRING36=(Token)match(input,STRING,FOLLOW_STRING_in_specialType754); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            STRING36_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(STRING36);
            adaptor.addChild(root_0, STRING36_tree);
            }
            e=(Token)match(input,111,FOLLOW_111_in_specialType758); if (state.failed) return retval;

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(s); 
              		retval.tree.getExtraTokens().add(e);
              		retval.tree.getToken().setType(TYPE);
              	
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
    // $ANTLR end specialType

    public static class pathName_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start pathName
    // EolParserRules.g:225:1: pathName : (metamodel= NAME '!' )? head= packagedType ( '#' label= NAME )? ;
    public final Pinset_EolParserRules.pathName_return pathName() throws RecognitionException {
        Pinset_EolParserRules.pathName_return retval = new Pinset_EolParserRules.pathName_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token metamodel=null;
        Token label=null;
        Token char_literal37=null;
        Token char_literal38=null;
        Pinset_EolParserRules.packagedType_return head = null;


        org.eclipse.epsilon.common.parse.AST metamodel_tree=null;
        org.eclipse.epsilon.common.parse.AST label_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal37_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal38_tree=null;

        try {
            // EolParserRules.g:226:2: ( (metamodel= NAME '!' )? head= packagedType ( '#' label= NAME )? )
            // EolParserRules.g:226:4: (metamodel= NAME '!' )? head= packagedType ( '#' label= NAME )?
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            // EolParserRules.g:226:4: (metamodel= NAME '!' )?
            int alt17=2;
            int LA17_0 = input.LA(1);

            if ( (LA17_0==NAME) ) {
                int LA17_1 = input.LA(2);

                if ( (LA17_1==115) ) {
                    alt17=1;
                }
            }
            switch (alt17) {
                case 1 :
                    // EolParserRules.g:226:5: metamodel= NAME '!'
                    {
                    metamodel=(Token)match(input,NAME,FOLLOW_NAME_in_pathName773); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    metamodel_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(metamodel);
                    adaptor.addChild(root_0, metamodel_tree);
                    }
                    char_literal37=(Token)match(input,115,FOLLOW_115_in_pathName775); if (state.failed) return retval;

                    }
                    break;

            }

            pushFollow(FOLLOW_packagedType_in_pathName784);
            head=packagedType();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(head.getTree(), root_0);
            // EolParserRules.g:228:3: ( '#' label= NAME )?
            int alt18=2;
            int LA18_0 = input.LA(1);

            if ( (LA18_0==116) ) {
                alt18=1;
            }
            switch (alt18) {
                case 1 :
                    // EolParserRules.g:228:4: '#' label= NAME
                    {
                    char_literal38=(Token)match(input,116,FOLLOW_116_in_pathName790); if (state.failed) return retval;
                    label=(Token)match(input,NAME,FOLLOW_NAME_in_pathName795); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    label_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(label);
                    adaptor.addChild(root_0, label_tree);
                    }

                    }
                    break;

            }

            if ( state.backtracking==0 ) {

              			if (metamodel != null) {
              				(head!=null?((org.eclipse.epsilon.common.parse.AST)head.tree):null).token.setText(metamodel.getText() + "!" + (head!=null?((org.eclipse.epsilon.common.parse.AST)head.tree):null).token.getText());		
              			}
              			
              			if (label != null) {
              				(head!=null?((org.eclipse.epsilon.common.parse.AST)head.tree):null).token.setText((head!=null?((org.eclipse.epsilon.common.parse.AST)head.tree):null).token.getText() + "#" + label.getText());
              				(head!=null?((org.eclipse.epsilon.common.parse.AST)head.tree):null).token.setType(ENUMERATION_VALUE);
              			}	
              		
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
    // $ANTLR end pathName

    public static class packagedType_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start packagedType
    // EolParserRules.g:242:1: packagedType : head= NAME ( '::' field= NAME )* ;
    public final Pinset_EolParserRules.packagedType_return packagedType() throws RecognitionException {
        Pinset_EolParserRules.packagedType_return retval = new Pinset_EolParserRules.packagedType_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token head=null;
        Token field=null;
        Token string_literal39=null;

        org.eclipse.epsilon.common.parse.AST head_tree=null;
        org.eclipse.epsilon.common.parse.AST field_tree=null;
        org.eclipse.epsilon.common.parse.AST string_literal39_tree=null;

        try {
            // EolParserRules.g:243:2: (head= NAME ( '::' field= NAME )* )
            // EolParserRules.g:243:4: head= NAME ( '::' field= NAME )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            head=(Token)match(input,NAME,FOLLOW_NAME_in_packagedType816); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            head_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(head);
            adaptor.addChild(root_0, head_tree);
            }
            // EolParserRules.g:243:14: ( '::' field= NAME )*
            loop19:
            do {
                int alt19=2;
                int LA19_0 = input.LA(1);

                if ( (LA19_0==117) ) {
                    alt19=1;
                }


                switch (alt19) {
            	case 1 :
            	    // EolParserRules.g:243:15: '::' field= NAME
            	    {
            	    string_literal39=(Token)match(input,117,FOLLOW_117_in_packagedType819); if (state.failed) return retval;
            	    field=(Token)match(input,NAME,FOLLOW_NAME_in_packagedType824); if (state.failed) return retval;
            	    if ( state.backtracking==0 ) {
            	       
            	      				head.setText(head.getText() + "::" + field.getText()); 
            	      				((CommonToken) head).setStopIndex(((CommonToken)field).getStopIndex());
            	      			
            	    }

            	    }
            	    break;

            	default :
            	    break loop19;
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
    // $ANTLR end packagedType

    public static class collectionType_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start collectionType
    // EolParserRules.g:251:1: collectionType : ( CollectionTypeName | MapTypeName ) ( (op= '(' tn= typeName ( ',' tn= typeName )* cp= ')' ) | (op= '<' tn= typeName ( ',' tn= typeName )* cp= '>' ) )? ;
    public final Pinset_EolParserRules.collectionType_return collectionType() throws RecognitionException {
        Pinset_EolParserRules.collectionType_return retval = new Pinset_EolParserRules.collectionType_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Token cp=null;
        Token set40=null;
        Token char_literal41=null;
        Token char_literal42=null;
        Pinset_EolParserRules.typeName_return tn = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;
        org.eclipse.epsilon.common.parse.AST cp_tree=null;
        org.eclipse.epsilon.common.parse.AST set40_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal41_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal42_tree=null;

        try {
            // EolParserRules.g:257:2: ( ( CollectionTypeName | MapTypeName ) ( (op= '(' tn= typeName ( ',' tn= typeName )* cp= ')' ) | (op= '<' tn= typeName ( ',' tn= typeName )* cp= '>' ) )? )
            // EolParserRules.g:257:5: ( CollectionTypeName | MapTypeName ) ( (op= '(' tn= typeName ( ',' tn= typeName )* cp= ')' ) | (op= '<' tn= typeName ( ',' tn= typeName )* cp= '>' ) )?
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            set40=input.LT(1);
            set40=input.LT(1);
            if ( (input.LA(1)>=CollectionTypeName && input.LA(1)<=MapTypeName) ) {
                input.consume();
                if ( state.backtracking==0 ) root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(set40), root_0);
                state.errorRecovery=false;state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                throw mse;
            }

            // EolParserRules.g:258:3: ( (op= '(' tn= typeName ( ',' tn= typeName )* cp= ')' ) | (op= '<' tn= typeName ( ',' tn= typeName )* cp= '>' ) )?
            int alt22=3;
            alt22 = dfa22.predict(input);
            switch (alt22) {
                case 1 :
                    // EolParserRules.g:258:4: (op= '(' tn= typeName ( ',' tn= typeName )* cp= ')' )
                    {
                    // EolParserRules.g:258:4: (op= '(' tn= typeName ( ',' tn= typeName )* cp= ')' )
                    // EolParserRules.g:258:5: op= '(' tn= typeName ( ',' tn= typeName )* cp= ')'
                    {
                    op=(Token)match(input,110,FOLLOW_110_in_collectionType869); if (state.failed) return retval;
                    pushFollow(FOLLOW_typeName_in_collectionType874);
                    tn=typeName();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, tn.getTree());
                    if ( state.backtracking==0 ) {
                      setTokenType(tn,TYPE);
                    }
                    // EolParserRules.g:258:50: ( ',' tn= typeName )*
                    loop20:
                    do {
                        int alt20=2;
                        int LA20_0 = input.LA(1);

                        if ( (LA20_0==103) ) {
                            alt20=1;
                        }


                        switch (alt20) {
                    	case 1 :
                    	    // EolParserRules.g:258:51: ',' tn= typeName
                    	    {
                    	    char_literal41=(Token)match(input,103,FOLLOW_103_in_collectionType879); if (state.failed) return retval;
                    	    if ( state.backtracking==0 ) {
                    	    char_literal41_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(char_literal41);
                    	    adaptor.addChild(root_0, char_literal41_tree);
                    	    }
                    	    pushFollow(FOLLOW_typeName_in_collectionType883);
                    	    tn=typeName();

                    	    state._fsp--;
                    	    if (state.failed) return retval;
                    	    if ( state.backtracking==0 ) adaptor.addChild(root_0, tn.getTree());
                    	    if ( state.backtracking==0 ) {
                    	      setTokenType(tn,TYPE);
                    	    }

                    	    }
                    	    break;

                    	default :
                    	    break loop20;
                        }
                    } while (true);

                    cp=(Token)match(input,111,FOLLOW_111_in_collectionType891); if (state.failed) return retval;

                    }


                    }
                    break;
                case 2 :
                    // EolParserRules.g:259:4: (op= '<' tn= typeName ( ',' tn= typeName )* cp= '>' )
                    {
                    // EolParserRules.g:259:4: (op= '<' tn= typeName ( ',' tn= typeName )* cp= '>' )
                    // EolParserRules.g:259:5: op= '<' tn= typeName ( ',' tn= typeName )* cp= '>'
                    {
                    op=(Token)match(input,118,FOLLOW_118_in_collectionType903); if (state.failed) return retval;
                    pushFollow(FOLLOW_typeName_in_collectionType908);
                    tn=typeName();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, tn.getTree());
                    if ( state.backtracking==0 ) {
                      setTokenType(tn,TYPE);
                    }
                    // EolParserRules.g:259:50: ( ',' tn= typeName )*
                    loop21:
                    do {
                        int alt21=2;
                        int LA21_0 = input.LA(1);

                        if ( (LA21_0==103) ) {
                            alt21=1;
                        }


                        switch (alt21) {
                    	case 1 :
                    	    // EolParserRules.g:259:51: ',' tn= typeName
                    	    {
                    	    char_literal42=(Token)match(input,103,FOLLOW_103_in_collectionType913); if (state.failed) return retval;
                    	    if ( state.backtracking==0 ) {
                    	    char_literal42_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(char_literal42);
                    	    adaptor.addChild(root_0, char_literal42_tree);
                    	    }
                    	    pushFollow(FOLLOW_typeName_in_collectionType917);
                    	    tn=typeName();

                    	    state._fsp--;
                    	    if (state.failed) return retval;
                    	    if ( state.backtracking==0 ) adaptor.addChild(root_0, tn.getTree());
                    	    if ( state.backtracking==0 ) {
                    	      setTokenType(tn,TYPE);
                    	    }

                    	    }
                    	    break;

                    	default :
                    	    break loop21;
                        }
                    } while (true);

                    cp=(Token)match(input,119,FOLLOW_119_in_collectionType925); if (state.failed) return retval;

                    }


                    }
                    break;

            }


            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(op); 
              		retval.tree.getExtraTokens().add(cp);
              		retval.tree.getToken().setType(TYPE);
              	
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
    // $ANTLR end collectionType

    public static class statement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start statement
    // EolParserRules.g:263:1: statement : ( statementA | statementB );
    public final Pinset_EolParserRules.statement_return statement() throws RecognitionException {
        Pinset_EolParserRules.statement_return retval = new Pinset_EolParserRules.statement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.statementA_return statementA43 = null;

        Pinset_EolParserRules.statementB_return statementB44 = null;



        try {
            // EolParserRules.g:264:2: ( statementA | statementB )
            int alt23=2;
            alt23 = dfa23.predict(input);
            switch (alt23) {
                case 1 :
                    // EolParserRules.g:264:4: statementA
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_statementA_in_statement944);
                    statementA43=statementA();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, statementA43.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:264:17: statementB
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_statementB_in_statement948);
                    statementB44=statementB();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, statementB44.getTree());

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
    // $ANTLR end statement

    public static class statementA_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start statementA
    // EolParserRules.g:267:1: statementA : ( assignmentStatement | expressionStatement | forStatement | ifStatement | whileStatement | switchStatement | returnStatement | breakStatement );
    public final Pinset_EolParserRules.statementA_return statementA() throws RecognitionException {
        Pinset_EolParserRules.statementA_return retval = new Pinset_EolParserRules.statementA_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.assignmentStatement_return assignmentStatement45 = null;

        Pinset_EolParserRules.expressionStatement_return expressionStatement46 = null;

        Pinset_EolParserRules.forStatement_return forStatement47 = null;

        Pinset_EolParserRules.ifStatement_return ifStatement48 = null;

        Pinset_EolParserRules.whileStatement_return whileStatement49 = null;

        Pinset_EolParserRules.switchStatement_return switchStatement50 = null;

        Pinset_EolParserRules.returnStatement_return returnStatement51 = null;

        Pinset_EolParserRules.breakStatement_return breakStatement52 = null;



        try {
            // EolParserRules.g:268:2: ( assignmentStatement | expressionStatement | forStatement | ifStatement | whileStatement | switchStatement | returnStatement | breakStatement )
            int alt24=8;
            alt24 = dfa24.predict(input);
            switch (alt24) {
                case 1 :
                    // EolParserRules.g:268:4: assignmentStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_assignmentStatement_in_statementA959);
                    assignmentStatement45=assignmentStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, assignmentStatement45.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:268:26: expressionStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_expressionStatement_in_statementA963);
                    expressionStatement46=expressionStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionStatement46.getTree());

                    }
                    break;
                case 3 :
                    // EolParserRules.g:268:48: forStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_forStatement_in_statementA967);
                    forStatement47=forStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, forStatement47.getTree());

                    }
                    break;
                case 4 :
                    // EolParserRules.g:269:5: ifStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_ifStatement_in_statementA973);
                    ifStatement48=ifStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, ifStatement48.getTree());

                    }
                    break;
                case 5 :
                    // EolParserRules.g:269:19: whileStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_whileStatement_in_statementA977);
                    whileStatement49=whileStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, whileStatement49.getTree());

                    }
                    break;
                case 6 :
                    // EolParserRules.g:269:36: switchStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_switchStatement_in_statementA981);
                    switchStatement50=switchStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, switchStatement50.getTree());

                    }
                    break;
                case 7 :
                    // EolParserRules.g:269:54: returnStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_returnStatement_in_statementA985);
                    returnStatement51=returnStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, returnStatement51.getTree());

                    }
                    break;
                case 8 :
                    // EolParserRules.g:269:72: breakStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_breakStatement_in_statementA989);
                    breakStatement52=breakStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, breakStatement52.getTree());

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
    // $ANTLR end statementA

    public static class statementB_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start statementB
    // EolParserRules.g:272:1: statementB : ( breakAllStatement | returnStatement | transactionStatement | abortStatement | continueStatement | throwStatement | deleteStatement );
    public final Pinset_EolParserRules.statementB_return statementB() throws RecognitionException {
        Pinset_EolParserRules.statementB_return retval = new Pinset_EolParserRules.statementB_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.breakAllStatement_return breakAllStatement53 = null;

        Pinset_EolParserRules.returnStatement_return returnStatement54 = null;

        Pinset_EolParserRules.transactionStatement_return transactionStatement55 = null;

        Pinset_EolParserRules.abortStatement_return abortStatement56 = null;

        Pinset_EolParserRules.continueStatement_return continueStatement57 = null;

        Pinset_EolParserRules.throwStatement_return throwStatement58 = null;

        Pinset_EolParserRules.deleteStatement_return deleteStatement59 = null;



        try {
            // EolParserRules.g:273:2: ( breakAllStatement | returnStatement | transactionStatement | abortStatement | continueStatement | throwStatement | deleteStatement )
            int alt25=7;
            switch ( input.LA(1) ) {
            case 132:
                {
                alt25=1;
                }
                break;
            case 128:
                {
                alt25=2;
                }
                break;
            case 135:
                {
                alt25=3;
                }
                break;
            case 134:
                {
                alt25=4;
                }
                break;
            case 133:
                {
                alt25=5;
                }
                break;
            case 129:
                {
                alt25=6;
                }
                break;
            case 130:
                {
                alt25=7;
                }
                break;
            default:
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 25, 0, input);

                throw nvae;
            }

            switch (alt25) {
                case 1 :
                    // EolParserRules.g:273:4: breakAllStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_breakAllStatement_in_statementB1001);
                    breakAllStatement53=breakAllStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, breakAllStatement53.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:273:24: returnStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_returnStatement_in_statementB1005);
                    returnStatement54=returnStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, returnStatement54.getTree());

                    }
                    break;
                case 3 :
                    // EolParserRules.g:273:42: transactionStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_transactionStatement_in_statementB1009);
                    transactionStatement55=transactionStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, transactionStatement55.getTree());

                    }
                    break;
                case 4 :
                    // EolParserRules.g:274:5: abortStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_abortStatement_in_statementB1015);
                    abortStatement56=abortStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, abortStatement56.getTree());

                    }
                    break;
                case 5 :
                    // EolParserRules.g:274:22: continueStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_continueStatement_in_statementB1019);
                    continueStatement57=continueStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, continueStatement57.getTree());

                    }
                    break;
                case 6 :
                    // EolParserRules.g:274:42: throwStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_throwStatement_in_statementB1023);
                    throwStatement58=throwStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, throwStatement58.getTree());

                    }
                    break;
                case 7 :
                    // EolParserRules.g:275:5: deleteStatement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_deleteStatement_in_statementB1029);
                    deleteStatement59=deleteStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, deleteStatement59.getTree());

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
    // $ANTLR end statementB

    public static class statementOrStatementBlock_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start statementOrStatementBlock
    // EolParserRules.g:278:1: statementOrStatementBlock : ( statement | statementBlock );
    public final Pinset_EolParserRules.statementOrStatementBlock_return statementOrStatementBlock() throws RecognitionException {
        Pinset_EolParserRules.statementOrStatementBlock_return retval = new Pinset_EolParserRules.statementOrStatementBlock_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.statement_return statement60 = null;

        Pinset_EolParserRules.statementBlock_return statementBlock61 = null;



        try {
            // EolParserRules.g:279:2: ( statement | statementBlock )
            int alt26=2;
            int LA26_0 = input.LA(1);

            if ( (LA26_0==FLOAT||LA26_0==INT||LA26_0==BOOLEAN||LA26_0==STRING||(LA26_0>=CollectionTypeName && LA26_0<=SpecialTypeName)||LA26_0==NAME||LA26_0==110||LA26_0==120||LA26_0==122||LA26_0==125||(LA26_0>=127 && LA26_0<=135)||LA26_0==152||LA26_0==155||(LA26_0>=162 && LA26_0<=164)) ) {
                alt26=1;
            }
            else if ( (LA26_0==105) ) {
                alt26=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 26, 0, input);

                throw nvae;
            }
            switch (alt26) {
                case 1 :
                    // EolParserRules.g:279:4: statement
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_statement_in_statementOrStatementBlock1040);
                    statement60=statement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, statement60.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:279:16: statementBlock
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_statementBlock_in_statementOrStatementBlock1044);
                    statementBlock61=statementBlock();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, statementBlock61.getTree());

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
    // $ANTLR end statementOrStatementBlock

    public static class expressionOrStatementBlock_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start expressionOrStatementBlock
    // EolParserRules.g:281:1: expressionOrStatementBlock : ( ':' logicalExpression | statementBlock );
    public final Pinset_EolParserRules.expressionOrStatementBlock_return expressionOrStatementBlock() throws RecognitionException {
        Pinset_EolParserRules.expressionOrStatementBlock_return retval = new Pinset_EolParserRules.expressionOrStatementBlock_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token char_literal62=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression63 = null;

        Pinset_EolParserRules.statementBlock_return statementBlock64 = null;


        org.eclipse.epsilon.common.parse.AST char_literal62_tree=null;

        try {
            // EolParserRules.g:282:2: ( ':' logicalExpression | statementBlock )
            int alt27=2;
            int LA27_0 = input.LA(1);

            if ( (LA27_0==112) ) {
                alt27=1;
            }
            else if ( (LA27_0==105) ) {
                alt27=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 27, 0, input);

                throw nvae;
            }
            switch (alt27) {
                case 1 :
                    // EolParserRules.g:282:4: ':' logicalExpression
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    char_literal62=(Token)match(input,112,FOLLOW_112_in_expressionOrStatementBlock1053); if (state.failed) return retval;
                    pushFollow(FOLLOW_logicalExpression_in_expressionOrStatementBlock1056);
                    logicalExpression63=logicalExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression63.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:282:29: statementBlock
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_statementBlock_in_expressionOrStatementBlock1060);
                    statementBlock64=statementBlock();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, statementBlock64.getTree());

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
    // $ANTLR end expressionOrStatementBlock

    public static class ifStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start ifStatement
    // EolParserRules.g:285:1: ifStatement : i= 'if' '(' logicalExpression ')' statementOrStatementBlock ( elseStatement )? ;
    public final Pinset_EolParserRules.ifStatement_return ifStatement() throws RecognitionException {
        Pinset_EolParserRules.ifStatement_return retval = new Pinset_EolParserRules.ifStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token i=null;
        Token char_literal65=null;
        Token char_literal67=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression66 = null;

        Pinset_EolParserRules.statementOrStatementBlock_return statementOrStatementBlock68 = null;

        Pinset_EolParserRules.elseStatement_return elseStatement69 = null;


        org.eclipse.epsilon.common.parse.AST i_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal65_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal67_tree=null;

        try {
            // EolParserRules.g:286:2: (i= 'if' '(' logicalExpression ')' statementOrStatementBlock ( elseStatement )? )
            // EolParserRules.g:286:4: i= 'if' '(' logicalExpression ')' statementOrStatementBlock ( elseStatement )?
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            i=(Token)match(input,120,FOLLOW_120_in_ifStatement1073); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            i_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(i);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(i_tree, root_0);
            }
            char_literal65=(Token)match(input,110,FOLLOW_110_in_ifStatement1076); if (state.failed) return retval;
            pushFollow(FOLLOW_logicalExpression_in_ifStatement1079);
            logicalExpression66=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression66.getTree());
            char_literal67=(Token)match(input,111,FOLLOW_111_in_ifStatement1081); if (state.failed) return retval;
            pushFollow(FOLLOW_statementOrStatementBlock_in_ifStatement1084);
            statementOrStatementBlock68=statementOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, statementOrStatementBlock68.getTree());
            // EolParserRules.g:286:66: ( elseStatement )?
            int alt28=2;
            int LA28_0 = input.LA(1);

            if ( (LA28_0==121) ) {
                int LA28_1 = input.LA(2);

                if ( (synpred43_EolParserRules()) ) {
                    alt28=1;
                }
            }
            switch (alt28) {
                case 1 :
                    // EolParserRules.g:0:0: elseStatement
                    {
                    pushFollow(FOLLOW_elseStatement_in_ifStatement1086);
                    elseStatement69=elseStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, elseStatement69.getTree());

                    }
                    break;

            }

            if ( state.backtracking==0 ) {
              i.setType(IF);
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
    // $ANTLR end ifStatement

    public static class elseStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start elseStatement
    // EolParserRules.g:290:1: elseStatement : e= 'else' statementOrStatementBlock ;
    public final Pinset_EolParserRules.elseStatement_return elseStatement() throws RecognitionException {
        Pinset_EolParserRules.elseStatement_return retval = new Pinset_EolParserRules.elseStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token e=null;
        Pinset_EolParserRules.statementOrStatementBlock_return statementOrStatementBlock70 = null;


        org.eclipse.epsilon.common.parse.AST e_tree=null;

        try {
            // EolParserRules.g:294:2: (e= 'else' statementOrStatementBlock )
            // EolParserRules.g:294:4: e= 'else' statementOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            e=(Token)match(input,121,FOLLOW_121_in_elseStatement1109); if (state.failed) return retval;
            pushFollow(FOLLOW_statementOrStatementBlock_in_elseStatement1112);
            statementOrStatementBlock70=statementOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, statementOrStatementBlock70.getTree());

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(e);
              	
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
    // $ANTLR end elseStatement

    public static class switchStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start switchStatement
    // EolParserRules.g:297:1: switchStatement : s= 'switch' '(' logicalExpression ')' '{' ( caseStatement )* ( defaultStatement )? '}' ;
    public final Pinset_EolParserRules.switchStatement_return switchStatement() throws RecognitionException {
        Pinset_EolParserRules.switchStatement_return retval = new Pinset_EolParserRules.switchStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token s=null;
        Token char_literal71=null;
        Token char_literal73=null;
        Token char_literal74=null;
        Token char_literal77=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression72 = null;

        Pinset_EolParserRules.caseStatement_return caseStatement75 = null;

        Pinset_EolParserRules.defaultStatement_return defaultStatement76 = null;


        org.eclipse.epsilon.common.parse.AST s_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal71_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal73_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal74_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal77_tree=null;

        try {
            // EolParserRules.g:298:2: (s= 'switch' '(' logicalExpression ')' '{' ( caseStatement )* ( defaultStatement )? '}' )
            // EolParserRules.g:298:4: s= 'switch' '(' logicalExpression ')' '{' ( caseStatement )* ( defaultStatement )? '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            s=(Token)match(input,122,FOLLOW_122_in_switchStatement1126); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            s_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(s);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(s_tree, root_0);
            }
            char_literal71=(Token)match(input,110,FOLLOW_110_in_switchStatement1129); if (state.failed) return retval;
            pushFollow(FOLLOW_logicalExpression_in_switchStatement1132);
            logicalExpression72=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression72.getTree());
            char_literal73=(Token)match(input,111,FOLLOW_111_in_switchStatement1134); if (state.failed) return retval;
            char_literal74=(Token)match(input,105,FOLLOW_105_in_switchStatement1137); if (state.failed) return retval;
            // EolParserRules.g:298:49: ( caseStatement )*
            loop29:
            do {
                int alt29=2;
                int LA29_0 = input.LA(1);

                if ( (LA29_0==123) ) {
                    alt29=1;
                }


                switch (alt29) {
            	case 1 :
            	    // EolParserRules.g:0:0: caseStatement
            	    {
            	    pushFollow(FOLLOW_caseStatement_in_switchStatement1140);
            	    caseStatement75=caseStatement();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, caseStatement75.getTree());

            	    }
            	    break;

            	default :
            	    break loop29;
                }
            } while (true);

            // EolParserRules.g:298:64: ( defaultStatement )?
            int alt30=2;
            int LA30_0 = input.LA(1);

            if ( (LA30_0==124) ) {
                alt30=1;
            }
            switch (alt30) {
                case 1 :
                    // EolParserRules.g:0:0: defaultStatement
                    {
                    pushFollow(FOLLOW_defaultStatement_in_switchStatement1143);
                    defaultStatement76=defaultStatement();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, defaultStatement76.getTree());

                    }
                    break;

            }

            char_literal77=(Token)match(input,106,FOLLOW_106_in_switchStatement1146); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              s.setType(SWITCH);
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
    // $ANTLR end switchStatement

    public static class caseStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start caseStatement
    // EolParserRules.g:302:1: caseStatement : c= 'case' logicalExpression ':' ( block | statementBlock ) ;
    public final Pinset_EolParserRules.caseStatement_return caseStatement() throws RecognitionException {
        Pinset_EolParserRules.caseStatement_return retval = new Pinset_EolParserRules.caseStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token c=null;
        Token char_literal79=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression78 = null;

        Pinset_EolParserRules.block_return block80 = null;

        Pinset_EolParserRules.statementBlock_return statementBlock81 = null;


        org.eclipse.epsilon.common.parse.AST c_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal79_tree=null;

        try {
            // EolParserRules.g:303:2: (c= 'case' logicalExpression ':' ( block | statementBlock ) )
            // EolParserRules.g:303:4: c= 'case' logicalExpression ':' ( block | statementBlock )
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            c=(Token)match(input,123,FOLLOW_123_in_caseStatement1165); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            c_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(c);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(c_tree, root_0);
            }
            pushFollow(FOLLOW_logicalExpression_in_caseStatement1168);
            logicalExpression78=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression78.getTree());
            char_literal79=(Token)match(input,112,FOLLOW_112_in_caseStatement1170); if (state.failed) return retval;
            // EolParserRules.g:303:37: ( block | statementBlock )
            int alt31=2;
            int LA31_0 = input.LA(1);

            if ( (LA31_0==EOF||LA31_0==FLOAT||LA31_0==INT||LA31_0==BOOLEAN||LA31_0==STRING||(LA31_0>=CollectionTypeName && LA31_0<=SpecialTypeName)||LA31_0==NAME||LA31_0==106||LA31_0==110||LA31_0==120||(LA31_0>=122 && LA31_0<=125)||(LA31_0>=127 && LA31_0<=135)||LA31_0==152||LA31_0==155||(LA31_0>=162 && LA31_0<=164)) ) {
                alt31=1;
            }
            else if ( (LA31_0==105) ) {
                alt31=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 31, 0, input);

                throw nvae;
            }
            switch (alt31) {
                case 1 :
                    // EolParserRules.g:303:38: block
                    {
                    pushFollow(FOLLOW_block_in_caseStatement1174);
                    block80=block();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, block80.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:303:46: statementBlock
                    {
                    pushFollow(FOLLOW_statementBlock_in_caseStatement1178);
                    statementBlock81=statementBlock();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, statementBlock81.getTree());

                    }
                    break;

            }

            if ( state.backtracking==0 ) {
              c.setType(CASE);
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
    // $ANTLR end caseStatement

    public static class defaultStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start defaultStatement
    // EolParserRules.g:307:1: defaultStatement : d= 'default' ':' ( block | statementBlock ) ;
    public final Pinset_EolParserRules.defaultStatement_return defaultStatement() throws RecognitionException {
        Pinset_EolParserRules.defaultStatement_return retval = new Pinset_EolParserRules.defaultStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token d=null;
        Token char_literal82=null;
        Pinset_EolParserRules.block_return block83 = null;

        Pinset_EolParserRules.statementBlock_return statementBlock84 = null;


        org.eclipse.epsilon.common.parse.AST d_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal82_tree=null;

        try {
            // EolParserRules.g:308:2: (d= 'default' ':' ( block | statementBlock ) )
            // EolParserRules.g:308:4: d= 'default' ':' ( block | statementBlock )
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            d=(Token)match(input,124,FOLLOW_124_in_defaultStatement1197); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            d_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(d);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(d_tree, root_0);
            }
            char_literal82=(Token)match(input,112,FOLLOW_112_in_defaultStatement1200); if (state.failed) return retval;
            // EolParserRules.g:308:22: ( block | statementBlock )
            int alt32=2;
            int LA32_0 = input.LA(1);

            if ( (LA32_0==EOF||LA32_0==FLOAT||LA32_0==INT||LA32_0==BOOLEAN||LA32_0==STRING||(LA32_0>=CollectionTypeName && LA32_0<=SpecialTypeName)||LA32_0==NAME||LA32_0==106||LA32_0==110||LA32_0==120||(LA32_0>=122 && LA32_0<=125)||(LA32_0>=127 && LA32_0<=135)||LA32_0==152||LA32_0==155||(LA32_0>=162 && LA32_0<=164)) ) {
                alt32=1;
            }
            else if ( (LA32_0==105) ) {
                alt32=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 32, 0, input);

                throw nvae;
            }
            switch (alt32) {
                case 1 :
                    // EolParserRules.g:308:23: block
                    {
                    pushFollow(FOLLOW_block_in_defaultStatement1204);
                    block83=block();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, block83.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:308:31: statementBlock
                    {
                    pushFollow(FOLLOW_statementBlock_in_defaultStatement1208);
                    statementBlock84=statementBlock();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, statementBlock84.getTree());

                    }
                    break;

            }

            if ( state.backtracking==0 ) {
              d.setType(DEFAULT);
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
    // $ANTLR end defaultStatement

    public static class forStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start forStatement
    // EolParserRules.g:312:1: forStatement : f= 'for' '(' formalParameter 'in' logicalExpression ')' statementOrStatementBlock ;
    public final Pinset_EolParserRules.forStatement_return forStatement() throws RecognitionException {
        Pinset_EolParserRules.forStatement_return retval = new Pinset_EolParserRules.forStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token f=null;
        Token char_literal85=null;
        Token string_literal87=null;
        Token char_literal89=null;
        Pinset_EolParserRules.formalParameter_return formalParameter86 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression88 = null;

        Pinset_EolParserRules.statementOrStatementBlock_return statementOrStatementBlock90 = null;


        org.eclipse.epsilon.common.parse.AST f_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal85_tree=null;
        org.eclipse.epsilon.common.parse.AST string_literal87_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal89_tree=null;

        try {
            // EolParserRules.g:313:2: (f= 'for' '(' formalParameter 'in' logicalExpression ')' statementOrStatementBlock )
            // EolParserRules.g:313:4: f= 'for' '(' formalParameter 'in' logicalExpression ')' statementOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            f=(Token)match(input,125,FOLLOW_125_in_forStatement1226); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            f_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(f);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(f_tree, root_0);
            }
            char_literal85=(Token)match(input,110,FOLLOW_110_in_forStatement1229); if (state.failed) return retval;
            pushFollow(FOLLOW_formalParameter_in_forStatement1232);
            formalParameter86=formalParameter();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, formalParameter86.getTree());
            string_literal87=(Token)match(input,126,FOLLOW_126_in_forStatement1234); if (state.failed) return retval;
            pushFollow(FOLLOW_logicalExpression_in_forStatement1237);
            logicalExpression88=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression88.getTree());
            char_literal89=(Token)match(input,111,FOLLOW_111_in_forStatement1239); if (state.failed) return retval;
            pushFollow(FOLLOW_statementOrStatementBlock_in_forStatement1242);
            statementOrStatementBlock90=statementOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, statementOrStatementBlock90.getTree());
            if ( state.backtracking==0 ) {
              f.setType(FOR);
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
    // $ANTLR end forStatement

    public static class whileStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start whileStatement
    // EolParserRules.g:317:1: whileStatement : w= 'while' '(' logicalExpression ')' statementOrStatementBlock ;
    public final Pinset_EolParserRules.whileStatement_return whileStatement() throws RecognitionException {
        Pinset_EolParserRules.whileStatement_return retval = new Pinset_EolParserRules.whileStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token w=null;
        Token char_literal91=null;
        Token char_literal93=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression92 = null;

        Pinset_EolParserRules.statementOrStatementBlock_return statementOrStatementBlock94 = null;


        org.eclipse.epsilon.common.parse.AST w_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal91_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal93_tree=null;

        try {
            // EolParserRules.g:318:2: (w= 'while' '(' logicalExpression ')' statementOrStatementBlock )
            // EolParserRules.g:318:4: w= 'while' '(' logicalExpression ')' statementOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            w=(Token)match(input,127,FOLLOW_127_in_whileStatement1258); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            w_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(w);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(w_tree, root_0);
            }
            char_literal91=(Token)match(input,110,FOLLOW_110_in_whileStatement1261); if (state.failed) return retval;
            pushFollow(FOLLOW_logicalExpression_in_whileStatement1264);
            logicalExpression92=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression92.getTree());
            char_literal93=(Token)match(input,111,FOLLOW_111_in_whileStatement1266); if (state.failed) return retval;
            pushFollow(FOLLOW_statementOrStatementBlock_in_whileStatement1269);
            statementOrStatementBlock94=statementOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, statementOrStatementBlock94.getTree());
            if ( state.backtracking==0 ) {
              w.setType(WHILE);
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
    // $ANTLR end whileStatement

    public static class returnStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start returnStatement
    // EolParserRules.g:322:1: returnStatement : r= 'return' ( logicalExpression )? sem= ';' ;
    public final Pinset_EolParserRules.returnStatement_return returnStatement() throws RecognitionException {
        Pinset_EolParserRules.returnStatement_return retval = new Pinset_EolParserRules.returnStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token r=null;
        Token sem=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression95 = null;


        org.eclipse.epsilon.common.parse.AST r_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:326:2: (r= 'return' ( logicalExpression )? sem= ';' )
            // EolParserRules.g:326:4: r= 'return' ( logicalExpression )? sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            r=(Token)match(input,128,FOLLOW_128_in_returnStatement1291); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            r_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(r);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(r_tree, root_0);
            }
            // EolParserRules.g:326:16: ( logicalExpression )?
            int alt33=2;
            int LA33_0 = input.LA(1);

            if ( (LA33_0==FLOAT||LA33_0==INT||LA33_0==BOOLEAN||LA33_0==STRING||(LA33_0>=CollectionTypeName && LA33_0<=SpecialTypeName)||LA33_0==NAME||LA33_0==110||LA33_0==152||LA33_0==155||(LA33_0>=162 && LA33_0<=164)) ) {
                alt33=1;
            }
            switch (alt33) {
                case 1 :
                    // EolParserRules.g:0:0: logicalExpression
                    {
                    pushFollow(FOLLOW_logicalExpression_in_returnStatement1294);
                    logicalExpression95=logicalExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression95.getTree());

                    }
                    break;

            }

            sem=(Token)match(input,101,FOLLOW_101_in_returnStatement1299); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              r.setType(RETURN);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end returnStatement

    public static class throwStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start throwStatement
    // EolParserRules.g:330:1: throwStatement : t= 'throw' ( logicalExpression )? sem= ';' ;
    public final Pinset_EolParserRules.throwStatement_return throwStatement() throws RecognitionException {
        Pinset_EolParserRules.throwStatement_return retval = new Pinset_EolParserRules.throwStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token t=null;
        Token sem=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression96 = null;


        org.eclipse.epsilon.common.parse.AST t_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:334:2: (t= 'throw' ( logicalExpression )? sem= ';' )
            // EolParserRules.g:334:4: t= 'throw' ( logicalExpression )? sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            t=(Token)match(input,129,FOLLOW_129_in_throwStatement1322); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            t_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(t);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(t_tree, root_0);
            }
            // EolParserRules.g:334:15: ( logicalExpression )?
            int alt34=2;
            int LA34_0 = input.LA(1);

            if ( (LA34_0==FLOAT||LA34_0==INT||LA34_0==BOOLEAN||LA34_0==STRING||(LA34_0>=CollectionTypeName && LA34_0<=SpecialTypeName)||LA34_0==NAME||LA34_0==110||LA34_0==152||LA34_0==155||(LA34_0>=162 && LA34_0<=164)) ) {
                alt34=1;
            }
            switch (alt34) {
                case 1 :
                    // EolParserRules.g:0:0: logicalExpression
                    {
                    pushFollow(FOLLOW_logicalExpression_in_throwStatement1325);
                    logicalExpression96=logicalExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression96.getTree());

                    }
                    break;

            }

            sem=(Token)match(input,101,FOLLOW_101_in_throwStatement1330); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              t.setType(THROW);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end throwStatement

    public static class deleteStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start deleteStatement
    // EolParserRules.g:338:1: deleteStatement : d= 'delete' ( logicalExpression )? sem= ';' ;
    public final Pinset_EolParserRules.deleteStatement_return deleteStatement() throws RecognitionException {
        Pinset_EolParserRules.deleteStatement_return retval = new Pinset_EolParserRules.deleteStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token d=null;
        Token sem=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression97 = null;


        org.eclipse.epsilon.common.parse.AST d_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:342:2: (d= 'delete' ( logicalExpression )? sem= ';' )
            // EolParserRules.g:342:4: d= 'delete' ( logicalExpression )? sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            d=(Token)match(input,130,FOLLOW_130_in_deleteStatement1353); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            d_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(d);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(d_tree, root_0);
            }
            // EolParserRules.g:342:16: ( logicalExpression )?
            int alt35=2;
            int LA35_0 = input.LA(1);

            if ( (LA35_0==FLOAT||LA35_0==INT||LA35_0==BOOLEAN||LA35_0==STRING||(LA35_0>=CollectionTypeName && LA35_0<=SpecialTypeName)||LA35_0==NAME||LA35_0==110||LA35_0==152||LA35_0==155||(LA35_0>=162 && LA35_0<=164)) ) {
                alt35=1;
            }
            switch (alt35) {
                case 1 :
                    // EolParserRules.g:0:0: logicalExpression
                    {
                    pushFollow(FOLLOW_logicalExpression_in_deleteStatement1356);
                    logicalExpression97=logicalExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression97.getTree());

                    }
                    break;

            }

            sem=(Token)match(input,101,FOLLOW_101_in_deleteStatement1361); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              d.setType(DELETE);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end deleteStatement

    public static class breakStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start breakStatement
    // EolParserRules.g:346:1: breakStatement : b= 'break' sem= ';' ;
    public final Pinset_EolParserRules.breakStatement_return breakStatement() throws RecognitionException {
        Pinset_EolParserRules.breakStatement_return retval = new Pinset_EolParserRules.breakStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token b=null;
        Token sem=null;

        org.eclipse.epsilon.common.parse.AST b_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:350:2: (b= 'break' sem= ';' )
            // EolParserRules.g:350:4: b= 'break' sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            b=(Token)match(input,131,FOLLOW_131_in_breakStatement1387); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            b_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(b);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(b_tree, root_0);
            }
            sem=(Token)match(input,101,FOLLOW_101_in_breakStatement1392); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              b.setType(BREAK);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end breakStatement

    public static class breakAllStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start breakAllStatement
    // EolParserRules.g:354:1: breakAllStatement : b= 'breakAll' sem= ';' ;
    public final Pinset_EolParserRules.breakAllStatement_return breakAllStatement() throws RecognitionException {
        Pinset_EolParserRules.breakAllStatement_return retval = new Pinset_EolParserRules.breakAllStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token b=null;
        Token sem=null;

        org.eclipse.epsilon.common.parse.AST b_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:358:2: (b= 'breakAll' sem= ';' )
            // EolParserRules.g:358:4: b= 'breakAll' sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            b=(Token)match(input,132,FOLLOW_132_in_breakAllStatement1415); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            b_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(b);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(b_tree, root_0);
            }
            sem=(Token)match(input,101,FOLLOW_101_in_breakAllStatement1420); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              b.setType(BREAKALL);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end breakAllStatement

    public static class continueStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start continueStatement
    // EolParserRules.g:362:1: continueStatement : c= 'continue' sem= ';' ;
    public final Pinset_EolParserRules.continueStatement_return continueStatement() throws RecognitionException {
        Pinset_EolParserRules.continueStatement_return retval = new Pinset_EolParserRules.continueStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token c=null;
        Token sem=null;

        org.eclipse.epsilon.common.parse.AST c_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:366:2: (c= 'continue' sem= ';' )
            // EolParserRules.g:366:4: c= 'continue' sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            c=(Token)match(input,133,FOLLOW_133_in_continueStatement1443); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            c_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(c);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(c_tree, root_0);
            }
            sem=(Token)match(input,101,FOLLOW_101_in_continueStatement1448); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              c.setType(CONTINUE);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end continueStatement

    public static class abortStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start abortStatement
    // EolParserRules.g:370:1: abortStatement : a= 'abort' sem= ';' ;
    public final Pinset_EolParserRules.abortStatement_return abortStatement() throws RecognitionException {
        Pinset_EolParserRules.abortStatement_return retval = new Pinset_EolParserRules.abortStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token a=null;
        Token sem=null;

        org.eclipse.epsilon.common.parse.AST a_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:374:2: (a= 'abort' sem= ';' )
            // EolParserRules.g:374:4: a= 'abort' sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            a=(Token)match(input,134,FOLLOW_134_in_abortStatement1471); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            a_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(a);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(a_tree, root_0);
            }
            sem=(Token)match(input,101,FOLLOW_101_in_abortStatement1476); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              a.setType(ABORT);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end abortStatement

    public static class transactionStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start transactionStatement
    // EolParserRules.g:378:1: transactionStatement : t= 'transaction' ( NAME ( ',' NAME )* )? statementOrStatementBlock ;
    public final Pinset_EolParserRules.transactionStatement_return transactionStatement() throws RecognitionException {
        Pinset_EolParserRules.transactionStatement_return retval = new Pinset_EolParserRules.transactionStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token t=null;
        Token NAME98=null;
        Token char_literal99=null;
        Token NAME100=null;
        Pinset_EolParserRules.statementOrStatementBlock_return statementOrStatementBlock101 = null;


        org.eclipse.epsilon.common.parse.AST t_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME98_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal99_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME100_tree=null;

        try {
            // EolParserRules.g:379:2: (t= 'transaction' ( NAME ( ',' NAME )* )? statementOrStatementBlock )
            // EolParserRules.g:379:4: t= 'transaction' ( NAME ( ',' NAME )* )? statementOrStatementBlock
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            t=(Token)match(input,135,FOLLOW_135_in_transactionStatement1493); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            t_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(t);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(t_tree, root_0);
            }
            // EolParserRules.g:379:21: ( NAME ( ',' NAME )* )?
            int alt37=2;
            alt37 = dfa37.predict(input);
            switch (alt37) {
                case 1 :
                    // EolParserRules.g:379:22: NAME ( ',' NAME )*
                    {
                    NAME98=(Token)match(input,NAME,FOLLOW_NAME_in_transactionStatement1497); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    NAME98_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME98);
                    adaptor.addChild(root_0, NAME98_tree);
                    }
                    // EolParserRules.g:379:27: ( ',' NAME )*
                    loop36:
                    do {
                        int alt36=2;
                        int LA36_0 = input.LA(1);

                        if ( (LA36_0==103) ) {
                            alt36=1;
                        }


                        switch (alt36) {
                    	case 1 :
                    	    // EolParserRules.g:379:28: ',' NAME
                    	    {
                    	    char_literal99=(Token)match(input,103,FOLLOW_103_in_transactionStatement1500); if (state.failed) return retval;
                    	    if ( state.backtracking==0 ) {
                    	    char_literal99_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(char_literal99);
                    	    adaptor.addChild(root_0, char_literal99_tree);
                    	    }
                    	    NAME100=(Token)match(input,NAME,FOLLOW_NAME_in_transactionStatement1502); if (state.failed) return retval;
                    	    if ( state.backtracking==0 ) {
                    	    NAME100_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME100);
                    	    adaptor.addChild(root_0, NAME100_tree);
                    	    }

                    	    }
                    	    break;

                    	default :
                    	    break loop36;
                        }
                    } while (true);


                    }
                    break;

            }

            pushFollow(FOLLOW_statementOrStatementBlock_in_transactionStatement1508);
            statementOrStatementBlock101=statementOrStatementBlock();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, statementOrStatementBlock101.getTree());
            if ( state.backtracking==0 ) {
              t.setType(TRANSACTION);
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
    // $ANTLR end transactionStatement

    public static class assignmentStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start assignmentStatement
    // EolParserRules.g:383:1: assignmentStatement : logicalExpression ( (normal= ':=' | normal= '+=' | normal= '-=' | normal= '*=' | normal= '/=' ) | special= '::=' ) logicalExpression sem= ';' ;
    public final Pinset_EolParserRules.assignmentStatement_return assignmentStatement() throws RecognitionException {
        Pinset_EolParserRules.assignmentStatement_return retval = new Pinset_EolParserRules.assignmentStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token normal=null;
        Token special=null;
        Token sem=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression102 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression103 = null;


        org.eclipse.epsilon.common.parse.AST normal_tree=null;
        org.eclipse.epsilon.common.parse.AST special_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:387:2: ( logicalExpression ( (normal= ':=' | normal= '+=' | normal= '-=' | normal= '*=' | normal= '/=' ) | special= '::=' ) logicalExpression sem= ';' )
            // EolParserRules.g:387:4: logicalExpression ( (normal= ':=' | normal= '+=' | normal= '-=' | normal= '*=' | normal= '/=' ) | special= '::=' ) logicalExpression sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_logicalExpression_in_assignmentStatement1528);
            logicalExpression102=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression102.getTree());
            // EolParserRules.g:387:22: ( (normal= ':=' | normal= '+=' | normal= '-=' | normal= '*=' | normal= '/=' ) | special= '::=' )
            int alt39=2;
            int LA39_0 = input.LA(1);

            if ( ((LA39_0>=136 && LA39_0<=140)) ) {
                alt39=1;
            }
            else if ( (LA39_0==141) ) {
                alt39=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 39, 0, input);

                throw nvae;
            }
            switch (alt39) {
                case 1 :
                    // EolParserRules.g:387:23: (normal= ':=' | normal= '+=' | normal= '-=' | normal= '*=' | normal= '/=' )
                    {
                    // EolParserRules.g:387:23: (normal= ':=' | normal= '+=' | normal= '-=' | normal= '*=' | normal= '/=' )
                    int alt38=5;
                    switch ( input.LA(1) ) {
                    case 136:
                        {
                        alt38=1;
                        }
                        break;
                    case 137:
                        {
                        alt38=2;
                        }
                        break;
                    case 138:
                        {
                        alt38=3;
                        }
                        break;
                    case 139:
                        {
                        alt38=4;
                        }
                        break;
                    case 140:
                        {
                        alt38=5;
                        }
                        break;
                    default:
                        if (state.backtracking>0) {state.failed=true; return retval;}
                        NoViableAltException nvae =
                            new NoViableAltException("", 38, 0, input);

                        throw nvae;
                    }

                    switch (alt38) {
                        case 1 :
                            // EolParserRules.g:387:24: normal= ':='
                            {
                            normal=(Token)match(input,136,FOLLOW_136_in_assignmentStatement1534); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            normal_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(normal);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(normal_tree, root_0);
                            }

                            }
                            break;
                        case 2 :
                            // EolParserRules.g:387:37: normal= '+='
                            {
                            normal=(Token)match(input,137,FOLLOW_137_in_assignmentStatement1539); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            normal_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(normal);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(normal_tree, root_0);
                            }

                            }
                            break;
                        case 3 :
                            // EolParserRules.g:387:50: normal= '-='
                            {
                            normal=(Token)match(input,138,FOLLOW_138_in_assignmentStatement1544); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            normal_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(normal);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(normal_tree, root_0);
                            }

                            }
                            break;
                        case 4 :
                            // EolParserRules.g:387:63: normal= '*='
                            {
                            normal=(Token)match(input,139,FOLLOW_139_in_assignmentStatement1549); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            normal_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(normal);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(normal_tree, root_0);
                            }

                            }
                            break;
                        case 5 :
                            // EolParserRules.g:387:76: normal= '/='
                            {
                            normal=(Token)match(input,140,FOLLOW_140_in_assignmentStatement1554); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            normal_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(normal);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(normal_tree, root_0);
                            }

                            }
                            break;

                    }

                    if ( state.backtracking==0 ) {
                      normal.setType(ASSIGNMENT);
                    }

                    }
                    break;
                case 2 :
                    // EolParserRules.g:388:35: special= '::='
                    {
                    special=(Token)match(input,141,FOLLOW_141_in_assignmentStatement1566); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    special_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(special);
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(special_tree, root_0);
                    }
                    if ( state.backtracking==0 ) {
                      special.setType(SPECIAL_ASSIGNMENT);
                    }

                    }
                    break;

            }

            pushFollow(FOLLOW_logicalExpression_in_assignmentStatement1574);
            logicalExpression103=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression103.getTree());
            sem=(Token)match(input,101,FOLLOW_101_in_assignmentStatement1578); if (state.failed) return retval;

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end assignmentStatement

    public static class expressionStatement_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start expressionStatement
    // EolParserRules.g:392:1: expressionStatement : ( ( postfixExpression op= '=' logicalExpression ) | logicalExpression ) sem= ';' ;
    public final Pinset_EolParserRules.expressionStatement_return expressionStatement() throws RecognitionException {
        Pinset_EolParserRules.expressionStatement_return retval = new Pinset_EolParserRules.expressionStatement_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Token sem=null;
        Pinset_EolParserRules.postfixExpression_return postfixExpression104 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression105 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression106 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;
        org.eclipse.epsilon.common.parse.AST sem_tree=null;

        try {
            // EolParserRules.g:396:2: ( ( ( postfixExpression op= '=' logicalExpression ) | logicalExpression ) sem= ';' )
            // EolParserRules.g:396:4: ( ( postfixExpression op= '=' logicalExpression ) | logicalExpression ) sem= ';'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            // EolParserRules.g:396:4: ( ( postfixExpression op= '=' logicalExpression ) | logicalExpression )
            int alt40=2;
            alt40 = dfa40.predict(input);
            switch (alt40) {
                case 1 :
                    // EolParserRules.g:396:5: ( postfixExpression op= '=' logicalExpression )
                    {
                    // EolParserRules.g:396:5: ( postfixExpression op= '=' logicalExpression )
                    // EolParserRules.g:396:6: postfixExpression op= '=' logicalExpression
                    {
                    pushFollow(FOLLOW_postfixExpression_in_expressionStatement1598);
                    postfixExpression104=postfixExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, postfixExpression104.getTree());
                    op=(Token)match(input,107,FOLLOW_107_in_expressionStatement1602); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
                    }
                    pushFollow(FOLLOW_logicalExpression_in_expressionStatement1605);
                    logicalExpression105=logicalExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression105.getTree());
                    if ( state.backtracking==0 ) {
                      op.setType(OPERATOR);
                    }

                    }


                    }
                    break;
                case 2 :
                    // EolParserRules.g:396:78: logicalExpression
                    {
                    pushFollow(FOLLOW_logicalExpression_in_expressionStatement1612);
                    logicalExpression106=logicalExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression106.getTree());

                    }
                    break;

            }

            sem=(Token)match(input,101,FOLLOW_101_in_expressionStatement1617); if (state.failed) return retval;

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(sem);
              	
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
    // $ANTLR end expressionStatement

    public static class logicalExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start logicalExpression
    // EolParserRules.g:399:1: logicalExpression : relationalExpression ( ( (op= 'or' | op= 'and' | op= 'xor' | op= 'implies' ) | (op= '?' relationalExpression ( 'else' | ':' ) ) ) relationalExpression )* ;
    public final Pinset_EolParserRules.logicalExpression_return logicalExpression() throws RecognitionException {
        Pinset_EolParserRules.logicalExpression_return retval = new Pinset_EolParserRules.logicalExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Token set109=null;
        Pinset_EolParserRules.relationalExpression_return relationalExpression107 = null;

        Pinset_EolParserRules.relationalExpression_return relationalExpression108 = null;

        Pinset_EolParserRules.relationalExpression_return relationalExpression110 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;
        org.eclipse.epsilon.common.parse.AST set109_tree=null;

        try {
            // EolParserRules.g:400:2: ( relationalExpression ( ( (op= 'or' | op= 'and' | op= 'xor' | op= 'implies' ) | (op= '?' relationalExpression ( 'else' | ':' ) ) ) relationalExpression )* )
            // EolParserRules.g:400:4: relationalExpression ( ( (op= 'or' | op= 'and' | op= 'xor' | op= 'implies' ) | (op= '?' relationalExpression ( 'else' | ':' ) ) ) relationalExpression )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_relationalExpression_in_logicalExpression1629);
            relationalExpression107=relationalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, relationalExpression107.getTree());
            // EolParserRules.g:400:25: ( ( (op= 'or' | op= 'and' | op= 'xor' | op= 'implies' ) | (op= '?' relationalExpression ( 'else' | ':' ) ) ) relationalExpression )*
            loop43:
            do {
                int alt43=2;
                int LA43_0 = input.LA(1);

                if ( ((LA43_0>=142 && LA43_0<=146)) ) {
                    alt43=1;
                }


                switch (alt43) {
            	case 1 :
            	    // EolParserRules.g:401:4: ( (op= 'or' | op= 'and' | op= 'xor' | op= 'implies' ) | (op= '?' relationalExpression ( 'else' | ':' ) ) ) relationalExpression
            	    {
            	    // EolParserRules.g:401:4: ( (op= 'or' | op= 'and' | op= 'xor' | op= 'implies' ) | (op= '?' relationalExpression ( 'else' | ':' ) ) )
            	    int alt42=2;
            	    int LA42_0 = input.LA(1);

            	    if ( ((LA42_0>=142 && LA42_0<=145)) ) {
            	        alt42=1;
            	    }
            	    else if ( (LA42_0==146) ) {
            	        alt42=2;
            	    }
            	    else {
            	        if (state.backtracking>0) {state.failed=true; return retval;}
            	        NoViableAltException nvae =
            	            new NoViableAltException("", 42, 0, input);

            	        throw nvae;
            	    }
            	    switch (alt42) {
            	        case 1 :
            	            // EolParserRules.g:401:5: (op= 'or' | op= 'and' | op= 'xor' | op= 'implies' )
            	            {
            	            // EolParserRules.g:401:5: (op= 'or' | op= 'and' | op= 'xor' | op= 'implies' )
            	            int alt41=4;
            	            switch ( input.LA(1) ) {
            	            case 142:
            	                {
            	                alt41=1;
            	                }
            	                break;
            	            case 143:
            	                {
            	                alt41=2;
            	                }
            	                break;
            	            case 144:
            	                {
            	                alt41=3;
            	                }
            	                break;
            	            case 145:
            	                {
            	                alt41=4;
            	                }
            	                break;
            	            default:
            	                if (state.backtracking>0) {state.failed=true; return retval;}
            	                NoViableAltException nvae =
            	                    new NoViableAltException("", 41, 0, input);

            	                throw nvae;
            	            }

            	            switch (alt41) {
            	                case 1 :
            	                    // EolParserRules.g:401:6: op= 'or'
            	                    {
            	                    op=(Token)match(input,142,FOLLOW_142_in_logicalExpression1640); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;
            	                case 2 :
            	                    // EolParserRules.g:401:15: op= 'and'
            	                    {
            	                    op=(Token)match(input,143,FOLLOW_143_in_logicalExpression1645); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;
            	                case 3 :
            	                    // EolParserRules.g:401:25: op= 'xor'
            	                    {
            	                    op=(Token)match(input,144,FOLLOW_144_in_logicalExpression1650); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;
            	                case 4 :
            	                    // EolParserRules.g:401:35: op= 'implies'
            	                    {
            	                    op=(Token)match(input,145,FOLLOW_145_in_logicalExpression1655); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;

            	            }

            	            if ( state.backtracking==0 ) {
            	              op.setType(OPERATOR);
            	            }

            	            }
            	            break;
            	        case 2 :
            	            // EolParserRules.g:402:4: (op= '?' relationalExpression ( 'else' | ':' ) )
            	            {
            	            // EolParserRules.g:402:4: (op= '?' relationalExpression ( 'else' | ':' ) )
            	            // EolParserRules.g:402:5: op= '?' relationalExpression ( 'else' | ':' )
            	            {
            	            op=(Token)match(input,146,FOLLOW_146_in_logicalExpression1669); if (state.failed) return retval;
            	            if ( state.backtracking==0 ) {
            	            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	            }
            	            pushFollow(FOLLOW_relationalExpression_in_logicalExpression1672);
            	            relationalExpression108=relationalExpression();

            	            state._fsp--;
            	            if (state.failed) return retval;
            	            if ( state.backtracking==0 ) adaptor.addChild(root_0, relationalExpression108.getTree());
            	            set109=input.LT(1);
            	            if ( input.LA(1)==112||input.LA(1)==121 ) {
            	                input.consume();
            	                if ( state.backtracking==0 ) adaptor.addChild(root_0, adaptor.create(set109));
            	                state.errorRecovery=false;state.failed=false;
            	            }
            	            else {
            	                if (state.backtracking>0) {state.failed=true; return retval;}
            	                MismatchedSetException mse = new MismatchedSetException(null,input);
            	                throw mse;
            	            }

            	            if ( state.backtracking==0 ) {
            	              op.setType(TERNARY);
            	            }

            	            }


            	            }
            	            break;

            	    }

            	    pushFollow(FOLLOW_relationalExpression_in_logicalExpression1690);
            	    relationalExpression110=relationalExpression();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, relationalExpression110.getTree());

            	    }
            	    break;

            	default :
            	    break loop43;
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
    // $ANTLR end logicalExpression

    public static class relationalExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start relationalExpression
    // EolParserRules.g:407:1: relationalExpression : additiveExpression ( (op= '==' relationalExpression | op= '=' relationalExpression | (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' ) additiveExpression ) )* ;
    public final Pinset_EolParserRules.relationalExpression_return relationalExpression() throws RecognitionException {
        Pinset_EolParserRules.relationalExpression_return retval = new Pinset_EolParserRules.relationalExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Pinset_EolParserRules.additiveExpression_return additiveExpression111 = null;

        Pinset_EolParserRules.relationalExpression_return relationalExpression112 = null;

        Pinset_EolParserRules.relationalExpression_return relationalExpression113 = null;

        Pinset_EolParserRules.additiveExpression_return additiveExpression114 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;

        try {
            // EolParserRules.g:408:2: ( additiveExpression ( (op= '==' relationalExpression | op= '=' relationalExpression | (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' ) additiveExpression ) )* )
            // EolParserRules.g:408:4: additiveExpression ( (op= '==' relationalExpression | op= '=' relationalExpression | (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' ) additiveExpression ) )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_additiveExpression_in_relationalExpression1706);
            additiveExpression111=additiveExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, additiveExpression111.getTree());
            // EolParserRules.g:408:23: ( (op= '==' relationalExpression | op= '=' relationalExpression | (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' ) additiveExpression ) )*
            loop46:
            do {
                int alt46=2;
                alt46 = dfa46.predict(input);
                switch (alt46) {
            	case 1 :
            	    // EolParserRules.g:408:24: (op= '==' relationalExpression | op= '=' relationalExpression | (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' ) additiveExpression )
            	    {
            	    // EolParserRules.g:408:24: (op= '==' relationalExpression | op= '=' relationalExpression | (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' ) additiveExpression )
            	    int alt45=3;
            	    switch ( input.LA(1) ) {
            	    case 147:
            	        {
            	        alt45=1;
            	        }
            	        break;
            	    case 107:
            	        {
            	        alt45=2;
            	        }
            	        break;
            	    case 118:
            	    case 119:
            	    case 148:
            	    case 149:
            	    case 150:
            	        {
            	        alt45=3;
            	        }
            	        break;
            	    default:
            	        if (state.backtracking>0) {state.failed=true; return retval;}
            	        NoViableAltException nvae =
            	            new NoViableAltException("", 45, 0, input);

            	        throw nvae;
            	    }

            	    switch (alt45) {
            	        case 1 :
            	            // EolParserRules.g:408:25: op= '==' relationalExpression
            	            {
            	            op=(Token)match(input,147,FOLLOW_147_in_relationalExpression1712); if (state.failed) return retval;
            	            if ( state.backtracking==0 ) {
            	            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	            }
            	            pushFollow(FOLLOW_relationalExpression_in_relationalExpression1715);
            	            relationalExpression112=relationalExpression();

            	            state._fsp--;
            	            if (state.failed) return retval;
            	            if ( state.backtracking==0 ) adaptor.addChild(root_0, relationalExpression112.getTree());

            	            }
            	            break;
            	        case 2 :
            	            // EolParserRules.g:408:57: op= '=' relationalExpression
            	            {
            	            op=(Token)match(input,107,FOLLOW_107_in_relationalExpression1721); if (state.failed) return retval;
            	            if ( state.backtracking==0 ) {
            	            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	            }
            	            pushFollow(FOLLOW_relationalExpression_in_relationalExpression1724);
            	            relationalExpression113=relationalExpression();

            	            state._fsp--;
            	            if (state.failed) return retval;
            	            if ( state.backtracking==0 ) adaptor.addChild(root_0, relationalExpression113.getTree());

            	            }
            	            break;
            	        case 3 :
            	            // EolParserRules.g:409:24: (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' ) additiveExpression
            	            {
            	            // EolParserRules.g:409:24: (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' )
            	            int alt44=5;
            	            switch ( input.LA(1) ) {
            	            case 119:
            	                {
            	                alt44=1;
            	                }
            	                break;
            	            case 118:
            	                {
            	                alt44=2;
            	                }
            	                break;
            	            case 148:
            	                {
            	                alt44=3;
            	                }
            	                break;
            	            case 149:
            	                {
            	                alt44=4;
            	                }
            	                break;
            	            case 150:
            	                {
            	                alt44=5;
            	                }
            	                break;
            	            default:
            	                if (state.backtracking>0) {state.failed=true; return retval;}
            	                NoViableAltException nvae =
            	                    new NoViableAltException("", 44, 0, input);

            	                throw nvae;
            	            }

            	            switch (alt44) {
            	                case 1 :
            	                    // EolParserRules.g:409:25: op= '>'
            	                    {
            	                    op=(Token)match(input,119,FOLLOW_119_in_relationalExpression1754); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;
            	                case 2 :
            	                    // EolParserRules.g:409:33: op= '<'
            	                    {
            	                    op=(Token)match(input,118,FOLLOW_118_in_relationalExpression1759); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;
            	                case 3 :
            	                    // EolParserRules.g:409:41: op= '>='
            	                    {
            	                    op=(Token)match(input,148,FOLLOW_148_in_relationalExpression1764); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;
            	                case 4 :
            	                    // EolParserRules.g:409:50: op= '<='
            	                    {
            	                    op=(Token)match(input,149,FOLLOW_149_in_relationalExpression1769); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;
            	                case 5 :
            	                    // EolParserRules.g:409:59: op= '<>'
            	                    {
            	                    op=(Token)match(input,150,FOLLOW_150_in_relationalExpression1774); if (state.failed) return retval;
            	                    if ( state.backtracking==0 ) {
            	                    op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	                    }

            	                    }
            	                    break;

            	            }

            	            pushFollow(FOLLOW_additiveExpression_in_relationalExpression1778);
            	            additiveExpression114=additiveExpression();

            	            state._fsp--;
            	            if (state.failed) return retval;
            	            if ( state.backtracking==0 ) adaptor.addChild(root_0, additiveExpression114.getTree());

            	            }
            	            break;

            	    }

            	    if ( state.backtracking==0 ) {
            	      op.setType(OPERATOR);
            	    }

            	    }
            	    break;

            	default :
            	    break loop46;
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
    // $ANTLR end relationalExpression

    public static class additiveExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start additiveExpression
    // EolParserRules.g:413:1: additiveExpression : multiplicativeExpression ( (op= '+' | op= '-' ) multiplicativeExpression )* ;
    public final Pinset_EolParserRules.additiveExpression_return additiveExpression() throws RecognitionException {
        Pinset_EolParserRules.additiveExpression_return retval = new Pinset_EolParserRules.additiveExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Pinset_EolParserRules.multiplicativeExpression_return multiplicativeExpression115 = null;

        Pinset_EolParserRules.multiplicativeExpression_return multiplicativeExpression116 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;

        try {
            // EolParserRules.g:414:2: ( multiplicativeExpression ( (op= '+' | op= '-' ) multiplicativeExpression )* )
            // EolParserRules.g:414:4: multiplicativeExpression ( (op= '+' | op= '-' ) multiplicativeExpression )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_multiplicativeExpression_in_additiveExpression1796);
            multiplicativeExpression115=multiplicativeExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, multiplicativeExpression115.getTree());
            // EolParserRules.g:414:29: ( (op= '+' | op= '-' ) multiplicativeExpression )*
            loop48:
            do {
                int alt48=2;
                int LA48_0 = input.LA(1);

                if ( ((LA48_0>=151 && LA48_0<=152)) ) {
                    alt48=1;
                }


                switch (alt48) {
            	case 1 :
            	    // EolParserRules.g:414:30: (op= '+' | op= '-' ) multiplicativeExpression
            	    {
            	    // EolParserRules.g:414:30: (op= '+' | op= '-' )
            	    int alt47=2;
            	    int LA47_0 = input.LA(1);

            	    if ( (LA47_0==151) ) {
            	        alt47=1;
            	    }
            	    else if ( (LA47_0==152) ) {
            	        alt47=2;
            	    }
            	    else {
            	        if (state.backtracking>0) {state.failed=true; return retval;}
            	        NoViableAltException nvae =
            	            new NoViableAltException("", 47, 0, input);

            	        throw nvae;
            	    }
            	    switch (alt47) {
            	        case 1 :
            	            // EolParserRules.g:414:31: op= '+'
            	            {
            	            op=(Token)match(input,151,FOLLOW_151_in_additiveExpression1802); if (state.failed) return retval;
            	            if ( state.backtracking==0 ) {
            	            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	            }

            	            }
            	            break;
            	        case 2 :
            	            // EolParserRules.g:414:39: op= '-'
            	            {
            	            op=(Token)match(input,152,FOLLOW_152_in_additiveExpression1807); if (state.failed) return retval;
            	            if ( state.backtracking==0 ) {
            	            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	            }

            	            }
            	            break;

            	    }

            	    pushFollow(FOLLOW_multiplicativeExpression_in_additiveExpression1811);
            	    multiplicativeExpression116=multiplicativeExpression();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, multiplicativeExpression116.getTree());
            	    if ( state.backtracking==0 ) {
            	      op.setType(OPERATOR);
            	    }

            	    }
            	    break;

            	default :
            	    break loop48;
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
    // $ANTLR end additiveExpression

    public static class multiplicativeExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start multiplicativeExpression
    // EolParserRules.g:418:1: multiplicativeExpression : unaryExpression ( (op= '*' | op= '/' ) unaryExpression )* ;
    public final Pinset_EolParserRules.multiplicativeExpression_return multiplicativeExpression() throws RecognitionException {
        Pinset_EolParserRules.multiplicativeExpression_return retval = new Pinset_EolParserRules.multiplicativeExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Pinset_EolParserRules.unaryExpression_return unaryExpression117 = null;

        Pinset_EolParserRules.unaryExpression_return unaryExpression118 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;

        try {
            // EolParserRules.g:419:2: ( unaryExpression ( (op= '*' | op= '/' ) unaryExpression )* )
            // EolParserRules.g:419:4: unaryExpression ( (op= '*' | op= '/' ) unaryExpression )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_unaryExpression_in_multiplicativeExpression1829);
            unaryExpression117=unaryExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, unaryExpression117.getTree());
            // EolParserRules.g:419:20: ( (op= '*' | op= '/' ) unaryExpression )*
            loop50:
            do {
                int alt50=2;
                int LA50_0 = input.LA(1);

                if ( ((LA50_0>=153 && LA50_0<=154)) ) {
                    alt50=1;
                }


                switch (alt50) {
            	case 1 :
            	    // EolParserRules.g:419:21: (op= '*' | op= '/' ) unaryExpression
            	    {
            	    // EolParserRules.g:419:21: (op= '*' | op= '/' )
            	    int alt49=2;
            	    int LA49_0 = input.LA(1);

            	    if ( (LA49_0==153) ) {
            	        alt49=1;
            	    }
            	    else if ( (LA49_0==154) ) {
            	        alt49=2;
            	    }
            	    else {
            	        if (state.backtracking>0) {state.failed=true; return retval;}
            	        NoViableAltException nvae =
            	            new NoViableAltException("", 49, 0, input);

            	        throw nvae;
            	    }
            	    switch (alt49) {
            	        case 1 :
            	            // EolParserRules.g:419:22: op= '*'
            	            {
            	            op=(Token)match(input,153,FOLLOW_153_in_multiplicativeExpression1835); if (state.failed) return retval;
            	            if ( state.backtracking==0 ) {
            	            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	            }

            	            }
            	            break;
            	        case 2 :
            	            // EolParserRules.g:419:30: op= '/'
            	            {
            	            op=(Token)match(input,154,FOLLOW_154_in_multiplicativeExpression1840); if (state.failed) return retval;
            	            if ( state.backtracking==0 ) {
            	            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
            	            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
            	            }

            	            }
            	            break;

            	    }

            	    pushFollow(FOLLOW_unaryExpression_in_multiplicativeExpression1844);
            	    unaryExpression118=unaryExpression();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, unaryExpression118.getTree());
            	    if ( state.backtracking==0 ) {
            	      op.setType(OPERATOR);
            	    }

            	    }
            	    break;

            	default :
            	    break loop50;
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
    // $ANTLR end multiplicativeExpression

    public static class unaryExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start unaryExpression
    // EolParserRules.g:423:1: unaryExpression : ( (op= 'not' | op= '-' ) )? shortcutOperatorExpression ;
    public final Pinset_EolParserRules.unaryExpression_return unaryExpression() throws RecognitionException {
        Pinset_EolParserRules.unaryExpression_return retval = new Pinset_EolParserRules.unaryExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Pinset_EolParserRules.shortcutOperatorExpression_return shortcutOperatorExpression119 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;

        try {
            // EolParserRules.g:424:2: ( ( (op= 'not' | op= '-' ) )? shortcutOperatorExpression )
            // EolParserRules.g:424:4: ( (op= 'not' | op= '-' ) )? shortcutOperatorExpression
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            // EolParserRules.g:424:4: ( (op= 'not' | op= '-' ) )?
            int alt52=2;
            int LA52_0 = input.LA(1);

            if ( (LA52_0==152||LA52_0==155) ) {
                alt52=1;
            }
            switch (alt52) {
                case 1 :
                    // EolParserRules.g:424:5: (op= 'not' | op= '-' )
                    {
                    // EolParserRules.g:424:5: (op= 'not' | op= '-' )
                    int alt51=2;
                    int LA51_0 = input.LA(1);

                    if ( (LA51_0==155) ) {
                        alt51=1;
                    }
                    else if ( (LA51_0==152) ) {
                        alt51=2;
                    }
                    else {
                        if (state.backtracking>0) {state.failed=true; return retval;}
                        NoViableAltException nvae =
                            new NoViableAltException("", 51, 0, input);

                        throw nvae;
                    }
                    switch (alt51) {
                        case 1 :
                            // EolParserRules.g:424:6: op= 'not'
                            {
                            op=(Token)match(input,155,FOLLOW_155_in_unaryExpression1865); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
                            }

                            }
                            break;
                        case 2 :
                            // EolParserRules.g:424:16: op= '-'
                            {
                            op=(Token)match(input,152,FOLLOW_152_in_unaryExpression1870); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
                            }

                            }
                            break;

                    }

                    if ( state.backtracking==0 ) {
                      op.setType(OPERATOR);
                    }

                    }
                    break;

            }

            pushFollow(FOLLOW_shortcutOperatorExpression_in_unaryExpression1878);
            shortcutOperatorExpression119=shortcutOperatorExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, shortcutOperatorExpression119.getTree());

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
    // $ANTLR end unaryExpression

    public static class shortcutOperatorExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start shortcutOperatorExpression
    // EolParserRules.g:427:1: shortcutOperatorExpression : postfixExpression ( (op= '++' | op= '--' ) )? ;
    public final Pinset_EolParserRules.shortcutOperatorExpression_return shortcutOperatorExpression() throws RecognitionException {
        Pinset_EolParserRules.shortcutOperatorExpression_return retval = new Pinset_EolParserRules.shortcutOperatorExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Pinset_EolParserRules.postfixExpression_return postfixExpression120 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;

        try {
            // EolParserRules.g:428:2: ( postfixExpression ( (op= '++' | op= '--' ) )? )
            // EolParserRules.g:428:4: postfixExpression ( (op= '++' | op= '--' ) )?
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_postfixExpression_in_shortcutOperatorExpression1890);
            postfixExpression120=postfixExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, postfixExpression120.getTree());
            // EolParserRules.g:428:22: ( (op= '++' | op= '--' ) )?
            int alt54=2;
            int LA54_0 = input.LA(1);

            if ( ((LA54_0>=156 && LA54_0<=157)) ) {
                alt54=1;
            }
            switch (alt54) {
                case 1 :
                    // EolParserRules.g:428:23: (op= '++' | op= '--' )
                    {
                    // EolParserRules.g:428:23: (op= '++' | op= '--' )
                    int alt53=2;
                    int LA53_0 = input.LA(1);

                    if ( (LA53_0==156) ) {
                        alt53=1;
                    }
                    else if ( (LA53_0==157) ) {
                        alt53=2;
                    }
                    else {
                        if (state.backtracking>0) {state.failed=true; return retval;}
                        NoViableAltException nvae =
                            new NoViableAltException("", 53, 0, input);

                        throw nvae;
                    }
                    switch (alt53) {
                        case 1 :
                            // EolParserRules.g:428:24: op= '++'
                            {
                            op=(Token)match(input,156,FOLLOW_156_in_shortcutOperatorExpression1896); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
                            }

                            }
                            break;
                        case 2 :
                            // EolParserRules.g:428:35: op= '--'
                            {
                            op=(Token)match(input,157,FOLLOW_157_in_shortcutOperatorExpression1903); if (state.failed) return retval;
                            if ( state.backtracking==0 ) {
                            op_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(op);
                            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(op_tree, root_0);
                            }

                            }
                            break;

                    }

                    if ( state.backtracking==0 ) {
                      op.setType(OPERATOR);
                    }

                    }
                    break;

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
    // $ANTLR end shortcutOperatorExpression

    public static class postfixExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start postfixExpression
    // EolParserRules.g:431:1: postfixExpression : itemSelectorExpression ( ( NAVIGATION | POINT | ARROW ) fc= featureCall (is= '[' logicalExpression ']' )* )* ;
    public final Pinset_EolParserRules.postfixExpression_return postfixExpression() throws RecognitionException {
        Pinset_EolParserRules.postfixExpression_return retval = new Pinset_EolParserRules.postfixExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token is=null;
        Token set122=null;
        Token char_literal124=null;
        Pinset_EolParserRules.featureCall_return fc = null;

        Pinset_EolParserRules.itemSelectorExpression_return itemSelectorExpression121 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression123 = null;


        org.eclipse.epsilon.common.parse.AST is_tree=null;
        org.eclipse.epsilon.common.parse.AST set122_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal124_tree=null;

        try {
            // EolParserRules.g:432:2: ( itemSelectorExpression ( ( NAVIGATION | POINT | ARROW ) fc= featureCall (is= '[' logicalExpression ']' )* )* )
            // EolParserRules.g:432:4: itemSelectorExpression ( ( NAVIGATION | POINT | ARROW ) fc= featureCall (is= '[' logicalExpression ']' )* )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_itemSelectorExpression_in_postfixExpression1921);
            itemSelectorExpression121=itemSelectorExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, itemSelectorExpression121.getTree());
            // EolParserRules.g:432:27: ( ( NAVIGATION | POINT | ARROW ) fc= featureCall (is= '[' logicalExpression ']' )* )*
            loop56:
            do {
                int alt56=2;
                int LA56_0 = input.LA(1);

                if ( (LA56_0==POINT||(LA56_0>=ARROW && LA56_0<=NAVIGATION)) ) {
                    alt56=1;
                }


                switch (alt56) {
            	case 1 :
            	    // EolParserRules.g:432:28: ( NAVIGATION | POINT | ARROW ) fc= featureCall (is= '[' logicalExpression ']' )*
            	    {
            	    set122=input.LT(1);
            	    set122=input.LT(1);
            	    if ( input.LA(1)==POINT||(input.LA(1)>=ARROW && input.LA(1)<=NAVIGATION) ) {
            	        input.consume();
            	        if ( state.backtracking==0 ) root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(set122), root_0);
            	        state.errorRecovery=false;state.failed=false;
            	    }
            	    else {
            	        if (state.backtracking>0) {state.failed=true; return retval;}
            	        MismatchedSetException mse = new MismatchedSetException(null,input);
            	        throw mse;
            	    }

            	    pushFollow(FOLLOW_featureCall_in_postfixExpression1935);
            	    fc=featureCall();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, fc.getTree());
            	    if ( state.backtracking==0 ) {
            	      setTokenType(fc,FEATURECALL);
            	    }
            	    // EolParserRules.g:433:35: (is= '[' logicalExpression ']' )*
            	    loop55:
            	    do {
            	        int alt55=2;
            	        int LA55_0 = input.LA(1);

            	        if ( (LA55_0==158) ) {
            	            alt55=1;
            	        }


            	        switch (alt55) {
            	    	case 1 :
            	    	    // EolParserRules.g:433:36: is= '[' logicalExpression ']'
            	    	    {
            	    	    is=(Token)match(input,158,FOLLOW_158_in_postfixExpression1944); if (state.failed) return retval;
            	    	    if ( state.backtracking==0 ) {
            	    	    is_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(is);
            	    	    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(is_tree, root_0);
            	    	    }
            	    	    pushFollow(FOLLOW_logicalExpression_in_postfixExpression1947);
            	    	    logicalExpression123=logicalExpression();

            	    	    state._fsp--;
            	    	    if (state.failed) return retval;
            	    	    if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression123.getTree());
            	    	    char_literal124=(Token)match(input,159,FOLLOW_159_in_postfixExpression1949); if (state.failed) return retval;
            	    	    if ( state.backtracking==0 ) {
            	    	      is.setType(ITEMSELECTOR);
            	    	    }

            	    	    }
            	    	    break;

            	    	default :
            	    	    break loop55;
            	        }
            	    } while (true);


            	    }
            	    break;

            	default :
            	    break loop56;
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
    // $ANTLR end postfixExpression

    public static class itemSelectorExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start itemSelectorExpression
    // EolParserRules.g:437:1: itemSelectorExpression : primitiveExpression (is= '[' primitiveExpression ']' )* ;
    public final Pinset_EolParserRules.itemSelectorExpression_return itemSelectorExpression() throws RecognitionException {
        Pinset_EolParserRules.itemSelectorExpression_return retval = new Pinset_EolParserRules.itemSelectorExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token is=null;
        Token char_literal127=null;
        Pinset_EolParserRules.primitiveExpression_return primitiveExpression125 = null;

        Pinset_EolParserRules.primitiveExpression_return primitiveExpression126 = null;


        org.eclipse.epsilon.common.parse.AST is_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal127_tree=null;

        try {
            // EolParserRules.g:438:2: ( primitiveExpression (is= '[' primitiveExpression ']' )* )
            // EolParserRules.g:438:4: primitiveExpression (is= '[' primitiveExpression ']' )*
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_primitiveExpression_in_itemSelectorExpression1971);
            primitiveExpression125=primitiveExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, primitiveExpression125.getTree());
            // EolParserRules.g:438:24: (is= '[' primitiveExpression ']' )*
            loop57:
            do {
                int alt57=2;
                int LA57_0 = input.LA(1);

                if ( (LA57_0==158) ) {
                    alt57=1;
                }


                switch (alt57) {
            	case 1 :
            	    // EolParserRules.g:438:25: is= '[' primitiveExpression ']'
            	    {
            	    is=(Token)match(input,158,FOLLOW_158_in_itemSelectorExpression1976); if (state.failed) return retval;
            	    if ( state.backtracking==0 ) {
            	    is_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(is);
            	    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(is_tree, root_0);
            	    }
            	    pushFollow(FOLLOW_primitiveExpression_in_itemSelectorExpression1979);
            	    primitiveExpression126=primitiveExpression();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) adaptor.addChild(root_0, primitiveExpression126.getTree());
            	    char_literal127=(Token)match(input,159,FOLLOW_159_in_itemSelectorExpression1981); if (state.failed) return retval;
            	    if ( state.backtracking==0 ) {
            	      is.setType(ITEMSELECTOR);
            	    }

            	    }
            	    break;

            	default :
            	    break loop57;
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
    // $ANTLR end itemSelectorExpression

    public static class featureCall_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start featureCall
    // EolParserRules.g:442:1: featureCall : ( simpleFeatureCall | complexFeatureCall );
    public final Pinset_EolParserRules.featureCall_return featureCall() throws RecognitionException {
        Pinset_EolParserRules.featureCall_return retval = new Pinset_EolParserRules.featureCall_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.simpleFeatureCall_return simpleFeatureCall128 = null;

        Pinset_EolParserRules.complexFeatureCall_return complexFeatureCall129 = null;



        try {
            // EolParserRules.g:443:2: ( simpleFeatureCall | complexFeatureCall )
            int alt58=2;
            alt58 = dfa58.predict(input);
            switch (alt58) {
                case 1 :
                    // EolParserRules.g:443:4: simpleFeatureCall
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_simpleFeatureCall_in_featureCall1999);
                    simpleFeatureCall128=simpleFeatureCall();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, simpleFeatureCall128.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:443:24: complexFeatureCall
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_complexFeatureCall_in_featureCall2003);
                    complexFeatureCall129=complexFeatureCall();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, complexFeatureCall129.getTree());

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
    // $ANTLR end featureCall

    public static class simpleFeatureCall_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start simpleFeatureCall
    // EolParserRules.g:446:1: simpleFeatureCall : n= NAME ( parameterList )? ;
    public final Pinset_EolParserRules.simpleFeatureCall_return simpleFeatureCall() throws RecognitionException {
        Pinset_EolParserRules.simpleFeatureCall_return retval = new Pinset_EolParserRules.simpleFeatureCall_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token n=null;
        Pinset_EolParserRules.parameterList_return parameterList130 = null;


        org.eclipse.epsilon.common.parse.AST n_tree=null;

        try {
            // EolParserRules.g:447:2: (n= NAME ( parameterList )? )
            // EolParserRules.g:447:5: n= NAME ( parameterList )?
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            n=(Token)match(input,NAME,FOLLOW_NAME_in_simpleFeatureCall2017); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            n_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(n);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(n_tree, root_0);
            }
            // EolParserRules.g:447:13: ( parameterList )?
            int alt59=2;
            int LA59_0 = input.LA(1);

            if ( (LA59_0==110) ) {
                alt59=1;
            }
            switch (alt59) {
                case 1 :
                    // EolParserRules.g:0:0: parameterList
                    {
                    pushFollow(FOLLOW_parameterList_in_simpleFeatureCall2020);
                    parameterList130=parameterList();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, parameterList130.getTree());

                    }
                    break;

            }

            if ( state.backtracking==0 ) {
              n.setType(FEATURECALL);
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
    // $ANTLR end simpleFeatureCall

    public static class parameterList_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start parameterList
    // EolParserRules.g:451:1: parameterList : op= '(' ( logicalExpression ( ',' logicalExpression )* )? cp= ')' -> ^( PARAMETERS ( logicalExpression )* ) ;
    public final Pinset_EolParserRules.parameterList_return parameterList() throws RecognitionException {
        Pinset_EolParserRules.parameterList_return retval = new Pinset_EolParserRules.parameterList_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Token cp=null;
        Token char_literal132=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression131 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression133 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;
        org.eclipse.epsilon.common.parse.AST cp_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal132_tree=null;
        RewriteRuleTokenStream stream_110=new RewriteRuleTokenStream(adaptor,"token 110");
        RewriteRuleTokenStream stream_111=new RewriteRuleTokenStream(adaptor,"token 111");
        RewriteRuleTokenStream stream_103=new RewriteRuleTokenStream(adaptor,"token 103");
        RewriteRuleSubtreeStream stream_logicalExpression=new RewriteRuleSubtreeStream(adaptor,"rule logicalExpression");
        try {
            // EolParserRules.g:457:2: (op= '(' ( logicalExpression ( ',' logicalExpression )* )? cp= ')' -> ^( PARAMETERS ( logicalExpression )* ) )
            // EolParserRules.g:457:4: op= '(' ( logicalExpression ( ',' logicalExpression )* )? cp= ')'
            {
            op=(Token)match(input,110,FOLLOW_110_in_parameterList2043); if (state.failed) return retval; 
            if ( state.backtracking==0 ) stream_110.add(op);

            // EolParserRules.g:457:11: ( logicalExpression ( ',' logicalExpression )* )?
            int alt61=2;
            int LA61_0 = input.LA(1);

            if ( (LA61_0==FLOAT||LA61_0==INT||LA61_0==BOOLEAN||LA61_0==STRING||(LA61_0>=CollectionTypeName && LA61_0<=SpecialTypeName)||LA61_0==NAME||LA61_0==110||LA61_0==152||LA61_0==155||(LA61_0>=162 && LA61_0<=164)) ) {
                alt61=1;
            }
            switch (alt61) {
                case 1 :
                    // EolParserRules.g:457:12: logicalExpression ( ',' logicalExpression )*
                    {
                    pushFollow(FOLLOW_logicalExpression_in_parameterList2046);
                    logicalExpression131=logicalExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) stream_logicalExpression.add(logicalExpression131.getTree());
                    // EolParserRules.g:457:30: ( ',' logicalExpression )*
                    loop60:
                    do {
                        int alt60=2;
                        int LA60_0 = input.LA(1);

                        if ( (LA60_0==103) ) {
                            alt60=1;
                        }


                        switch (alt60) {
                    	case 1 :
                    	    // EolParserRules.g:457:31: ',' logicalExpression
                    	    {
                    	    char_literal132=(Token)match(input,103,FOLLOW_103_in_parameterList2049); if (state.failed) return retval; 
                    	    if ( state.backtracking==0 ) stream_103.add(char_literal132);

                    	    pushFollow(FOLLOW_logicalExpression_in_parameterList2051);
                    	    logicalExpression133=logicalExpression();

                    	    state._fsp--;
                    	    if (state.failed) return retval;
                    	    if ( state.backtracking==0 ) stream_logicalExpression.add(logicalExpression133.getTree());

                    	    }
                    	    break;

                    	default :
                    	    break loop60;
                        }
                    } while (true);


                    }
                    break;

            }

            cp=(Token)match(input,111,FOLLOW_111_in_parameterList2059); if (state.failed) return retval; 
            if ( state.backtracking==0 ) stream_111.add(cp);



            // AST REWRITE
            // elements: logicalExpression
            // token labels: 
            // rule labels: retval
            // token list labels: 
            // rule list labels: 
            if ( state.backtracking==0 ) {
            retval.tree = root_0;
            RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"token retval",retval!=null?retval.tree:null);

            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
            // 458:3: -> ^( PARAMETERS ( logicalExpression )* )
            {
                // EolParserRules.g:458:6: ^( PARAMETERS ( logicalExpression )* )
                {
                org.eclipse.epsilon.common.parse.AST root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
                root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(PARAMETERS, "PARAMETERS"), root_1);

                // EolParserRules.g:458:19: ( logicalExpression )*
                while ( stream_logicalExpression.hasNext() ) {
                    adaptor.addChild(root_1, stream_logicalExpression.nextTree());

                }
                stream_logicalExpression.reset();

                adaptor.addChild(root_0, root_1);
                }

            }

            retval.tree = root_0;retval.tree = root_0;}
            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.setImaginary(true);
              		retval.tree.getExtraTokens().add(op);
              		retval.tree.getExtraTokens().add(cp);
              	
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
    // $ANTLR end parameterList

    public static class complexFeatureCall_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start complexFeatureCall
    // EolParserRules.g:461:1: complexFeatureCall : NAME op= '(' ( lambdaExpression | lambdaExpressionInBrackets ) ( ',' ( logicalExpression | lambdaExpressionInBrackets ) )* cp= ')' ;
    public final Pinset_EolParserRules.complexFeatureCall_return complexFeatureCall() throws RecognitionException {
        Pinset_EolParserRules.complexFeatureCall_return retval = new Pinset_EolParserRules.complexFeatureCall_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token op=null;
        Token cp=null;
        Token NAME134=null;
        Token char_literal137=null;
        Pinset_EolParserRules.lambdaExpression_return lambdaExpression135 = null;

        Pinset_EolParserRules.lambdaExpressionInBrackets_return lambdaExpressionInBrackets136 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression138 = null;

        Pinset_EolParserRules.lambdaExpressionInBrackets_return lambdaExpressionInBrackets139 = null;


        org.eclipse.epsilon.common.parse.AST op_tree=null;
        org.eclipse.epsilon.common.parse.AST cp_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME134_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal137_tree=null;

        try {
            // EolParserRules.g:466:2: ( NAME op= '(' ( lambdaExpression | lambdaExpressionInBrackets ) ( ',' ( logicalExpression | lambdaExpressionInBrackets ) )* cp= ')' )
            // EolParserRules.g:466:4: NAME op= '(' ( lambdaExpression | lambdaExpressionInBrackets ) ( ',' ( logicalExpression | lambdaExpressionInBrackets ) )* cp= ')'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            NAME134=(Token)match(input,NAME,FOLLOW_NAME_in_complexFeatureCall2087); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME134_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME134);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(NAME134_tree, root_0);
            }
            op=(Token)match(input,110,FOLLOW_110_in_complexFeatureCall2092); if (state.failed) return retval;
            // EolParserRules.g:466:18: ( lambdaExpression | lambdaExpressionInBrackets )
            int alt62=2;
            int LA62_0 = input.LA(1);

            if ( (LA62_0==NAME||(LA62_0>=160 && LA62_0<=161)) ) {
                alt62=1;
            }
            else if ( (LA62_0==110||LA62_0==158) ) {
                alt62=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 62, 0, input);

                throw nvae;
            }
            switch (alt62) {
                case 1 :
                    // EolParserRules.g:466:19: lambdaExpression
                    {
                    pushFollow(FOLLOW_lambdaExpression_in_complexFeatureCall2096);
                    lambdaExpression135=lambdaExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, lambdaExpression135.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:466:38: lambdaExpressionInBrackets
                    {
                    pushFollow(FOLLOW_lambdaExpressionInBrackets_in_complexFeatureCall2100);
                    lambdaExpressionInBrackets136=lambdaExpressionInBrackets();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, lambdaExpressionInBrackets136.getTree());

                    }
                    break;

            }

            // EolParserRules.g:467:3: ( ',' ( logicalExpression | lambdaExpressionInBrackets ) )*
            loop64:
            do {
                int alt64=2;
                int LA64_0 = input.LA(1);

                if ( (LA64_0==103) ) {
                    alt64=1;
                }


                switch (alt64) {
            	case 1 :
            	    // EolParserRules.g:467:4: ',' ( logicalExpression | lambdaExpressionInBrackets )
            	    {
            	    char_literal137=(Token)match(input,103,FOLLOW_103_in_complexFeatureCall2106); if (state.failed) return retval;
            	    // EolParserRules.g:467:9: ( logicalExpression | lambdaExpressionInBrackets )
            	    int alt63=2;
            	    switch ( input.LA(1) ) {
            	    case FLOAT:
            	    case INT:
            	    case BOOLEAN:
            	    case STRING:
            	    case CollectionTypeName:
            	    case MapTypeName:
            	    case SpecialTypeName:
            	    case NAME:
            	    case 152:
            	    case 155:
            	    case 162:
            	    case 163:
            	    case 164:
            	        {
            	        alt63=1;
            	        }
            	        break;
            	    case 110:
            	        {
            	        switch ( input.LA(2) ) {
            	        case NAME:
            	            {
            	            int LA63_4 = input.LA(3);

            	            if ( (LA63_4==POINT||(LA63_4>=ARROW && LA63_4<=NAVIGATION)||LA63_4==107||(LA63_4>=110 && LA63_4<=111)||(LA63_4>=115 && LA63_4<=119)||(LA63_4>=142 && LA63_4<=154)||(LA63_4>=156 && LA63_4<=158)) ) {
            	                alt63=1;
            	            }
            	            else if ( (LA63_4==103||LA63_4==112||(LA63_4>=160 && LA63_4<=161)) ) {
            	                alt63=2;
            	            }
            	            else {
            	                if (state.backtracking>0) {state.failed=true; return retval;}
            	                NoViableAltException nvae =
            	                    new NoViableAltException("", 63, 4, input);

            	                throw nvae;
            	            }
            	            }
            	            break;
            	        case 160:
            	        case 161:
            	            {
            	            alt63=2;
            	            }
            	            break;
            	        case FLOAT:
            	        case INT:
            	        case BOOLEAN:
            	        case STRING:
            	        case CollectionTypeName:
            	        case MapTypeName:
            	        case SpecialTypeName:
            	        case 110:
            	        case 152:
            	        case 155:
            	        case 162:
            	        case 163:
            	        case 164:
            	            {
            	            alt63=1;
            	            }
            	            break;
            	        default:
            	            if (state.backtracking>0) {state.failed=true; return retval;}
            	            NoViableAltException nvae =
            	                new NoViableAltException("", 63, 2, input);

            	            throw nvae;
            	        }

            	        }
            	        break;
            	    case 158:
            	        {
            	        alt63=2;
            	        }
            	        break;
            	    default:
            	        if (state.backtracking>0) {state.failed=true; return retval;}
            	        NoViableAltException nvae =
            	            new NoViableAltException("", 63, 0, input);

            	        throw nvae;
            	    }

            	    switch (alt63) {
            	        case 1 :
            	            // EolParserRules.g:467:10: logicalExpression
            	            {
            	            pushFollow(FOLLOW_logicalExpression_in_complexFeatureCall2110);
            	            logicalExpression138=logicalExpression();

            	            state._fsp--;
            	            if (state.failed) return retval;
            	            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression138.getTree());

            	            }
            	            break;
            	        case 2 :
            	            // EolParserRules.g:467:30: lambdaExpressionInBrackets
            	            {
            	            pushFollow(FOLLOW_lambdaExpressionInBrackets_in_complexFeatureCall2114);
            	            lambdaExpressionInBrackets139=lambdaExpressionInBrackets();

            	            state._fsp--;
            	            if (state.failed) return retval;
            	            if ( state.backtracking==0 ) adaptor.addChild(root_0, lambdaExpressionInBrackets139.getTree());

            	            }
            	            break;

            	    }


            	    }
            	    break;

            	default :
            	    break loop64;
                }
            } while (true);

            cp=(Token)match(input,111,FOLLOW_111_in_complexFeatureCall2121); if (state.failed) return retval;

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(op);
              		retval.tree.getExtraTokens().add(cp);
              	
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
    // $ANTLR end complexFeatureCall

    public static class lambdaExpressionInBrackets_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start lambdaExpressionInBrackets
    // EolParserRules.g:470:1: lambdaExpressionInBrackets : ( (lop= '(' lambdaExpression lcp= ')' ) | (lop= '[' lambdaExpression lcp= ']' ) );
    public final Pinset_EolParserRules.lambdaExpressionInBrackets_return lambdaExpressionInBrackets() throws RecognitionException {
        Pinset_EolParserRules.lambdaExpressionInBrackets_return retval = new Pinset_EolParserRules.lambdaExpressionInBrackets_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token lop=null;
        Token lcp=null;
        Pinset_EolParserRules.lambdaExpression_return lambdaExpression140 = null;

        Pinset_EolParserRules.lambdaExpression_return lambdaExpression141 = null;


        org.eclipse.epsilon.common.parse.AST lop_tree=null;
        org.eclipse.epsilon.common.parse.AST lcp_tree=null;

        try {
            // EolParserRules.g:476:2: ( (lop= '(' lambdaExpression lcp= ')' ) | (lop= '[' lambdaExpression lcp= ']' ) )
            int alt65=2;
            int LA65_0 = input.LA(1);

            if ( (LA65_0==110) ) {
                alt65=1;
            }
            else if ( (LA65_0==158) ) {
                alt65=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 65, 0, input);

                throw nvae;
            }
            switch (alt65) {
                case 1 :
                    // EolParserRules.g:476:4: (lop= '(' lambdaExpression lcp= ')' )
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    // EolParserRules.g:476:4: (lop= '(' lambdaExpression lcp= ')' )
                    // EolParserRules.g:476:5: lop= '(' lambdaExpression lcp= ')'
                    {
                    lop=(Token)match(input,110,FOLLOW_110_in_lambdaExpressionInBrackets2142); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    lop_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(lop);
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(lop_tree, root_0);
                    }
                    pushFollow(FOLLOW_lambdaExpression_in_lambdaExpressionInBrackets2145);
                    lambdaExpression140=lambdaExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, lambdaExpression140.getTree());
                    lcp=(Token)match(input,111,FOLLOW_111_in_lambdaExpressionInBrackets2149); if (state.failed) return retval;

                    }


                    }
                    break;
                case 2 :
                    // EolParserRules.g:477:3: (lop= '[' lambdaExpression lcp= ']' )
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    // EolParserRules.g:477:3: (lop= '[' lambdaExpression lcp= ']' )
                    // EolParserRules.g:477:4: lop= '[' lambdaExpression lcp= ']'
                    {
                    lop=(Token)match(input,158,FOLLOW_158_in_lambdaExpressionInBrackets2160); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    lop_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(lop);
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(lop_tree, root_0);
                    }
                    pushFollow(FOLLOW_lambdaExpression_in_lambdaExpressionInBrackets2163);
                    lambdaExpression141=lambdaExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, lambdaExpression141.getTree());
                    lcp=(Token)match(input,159,FOLLOW_159_in_lambdaExpressionInBrackets2167); if (state.failed) return retval;

                    }


                    }
                    break;

            }
            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		lop.setType(LAMBDAEXPR);
              		retval.tree.getExtraTokens().add(lop);
              		retval.tree.getExtraTokens().add(lcp);
              	
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
    // $ANTLR end lambdaExpressionInBrackets

    public static class lambdaExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start lambdaExpression
    // EolParserRules.g:480:1: lambdaExpression : ( formalParameterList )? (lt= '|' | lt= '=>' ) logicalExpression ;
    public final Pinset_EolParserRules.lambdaExpression_return lambdaExpression() throws RecognitionException {
        Pinset_EolParserRules.lambdaExpression_return retval = new Pinset_EolParserRules.lambdaExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token lt=null;
        Pinset_EolParserRules.formalParameterList_return formalParameterList142 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression143 = null;


        org.eclipse.epsilon.common.parse.AST lt_tree=null;

        try {
            // EolParserRules.g:484:2: ( ( formalParameterList )? (lt= '|' | lt= '=>' ) logicalExpression )
            // EolParserRules.g:484:4: ( formalParameterList )? (lt= '|' | lt= '=>' ) logicalExpression
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            // EolParserRules.g:484:4: ( formalParameterList )?
            int alt66=2;
            int LA66_0 = input.LA(1);

            if ( (LA66_0==NAME) ) {
                alt66=1;
            }
            switch (alt66) {
                case 1 :
                    // EolParserRules.g:0:0: formalParameterList
                    {
                    pushFollow(FOLLOW_formalParameterList_in_lambdaExpression2186);
                    formalParameterList142=formalParameterList();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, formalParameterList142.getTree());

                    }
                    break;

            }

            // EolParserRules.g:484:25: (lt= '|' | lt= '=>' )
            int alt67=2;
            int LA67_0 = input.LA(1);

            if ( (LA67_0==160) ) {
                alt67=1;
            }
            else if ( (LA67_0==161) ) {
                alt67=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 67, 0, input);

                throw nvae;
            }
            switch (alt67) {
                case 1 :
                    // EolParserRules.g:484:26: lt= '|'
                    {
                    lt=(Token)match(input,160,FOLLOW_160_in_lambdaExpression2192); if (state.failed) return retval;

                    }
                    break;
                case 2 :
                    // EolParserRules.g:484:36: lt= '=>'
                    {
                    lt=(Token)match(input,161,FOLLOW_161_in_lambdaExpression2199); if (state.failed) return retval;

                    }
                    break;

            }

            pushFollow(FOLLOW_logicalExpression_in_lambdaExpression2203);
            logicalExpression143=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression143.getTree());

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(lt);
              	
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
    // $ANTLR end lambdaExpression

    public static class newExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start newExpression
    // EolParserRules.g:487:1: newExpression : n= 'new' tn= typeName ( parameterList )? ;
    public final Pinset_EolParserRules.newExpression_return newExpression() throws RecognitionException {
        Pinset_EolParserRules.newExpression_return retval = new Pinset_EolParserRules.newExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token n=null;
        Pinset_EolParserRules.typeName_return tn = null;

        Pinset_EolParserRules.parameterList_return parameterList144 = null;


        org.eclipse.epsilon.common.parse.AST n_tree=null;

        try {
            // EolParserRules.g:488:2: (n= 'new' tn= typeName ( parameterList )? )
            // EolParserRules.g:488:4: n= 'new' tn= typeName ( parameterList )?
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            n=(Token)match(input,162,FOLLOW_162_in_newExpression2216); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            n_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(n);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(n_tree, root_0);
            }
            pushFollow(FOLLOW_typeName_in_newExpression2221);
            tn=typeName();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, tn.getTree());
            if ( state.backtracking==0 ) {
              setTokenType(tn,TYPE);
            }
            // EolParserRules.g:488:50: ( parameterList )?
            int alt68=2;
            int LA68_0 = input.LA(1);

            if ( (LA68_0==110) ) {
                alt68=1;
            }
            switch (alt68) {
                case 1 :
                    // EolParserRules.g:0:0: parameterList
                    {
                    pushFollow(FOLLOW_parameterList_in_newExpression2225);
                    parameterList144=parameterList();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, parameterList144.getTree());

                    }
                    break;

            }

            if ( state.backtracking==0 ) {
              n.setType(NEW);
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
    // $ANTLR end newExpression

    public static class variableDeclarationExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start variableDeclarationExpression
    // EolParserRules.g:492:1: variableDeclarationExpression : (v= 'var' | v= 'ext' ) NAME ( ':' (n= 'new' )? t= typeName ( parameterList )? )? ;
    public final Pinset_EolParserRules.variableDeclarationExpression_return variableDeclarationExpression() throws RecognitionException {
        Pinset_EolParserRules.variableDeclarationExpression_return retval = new Pinset_EolParserRules.variableDeclarationExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token v=null;
        Token n=null;
        Token NAME145=null;
        Token char_literal146=null;
        Pinset_EolParserRules.typeName_return t = null;

        Pinset_EolParserRules.parameterList_return parameterList147 = null;


        org.eclipse.epsilon.common.parse.AST v_tree=null;
        org.eclipse.epsilon.common.parse.AST n_tree=null;
        org.eclipse.epsilon.common.parse.AST NAME145_tree=null;
        org.eclipse.epsilon.common.parse.AST char_literal146_tree=null;

        try {
            // EolParserRules.g:498:2: ( (v= 'var' | v= 'ext' ) NAME ( ':' (n= 'new' )? t= typeName ( parameterList )? )? )
            // EolParserRules.g:498:4: (v= 'var' | v= 'ext' ) NAME ( ':' (n= 'new' )? t= typeName ( parameterList )? )?
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            // EolParserRules.g:498:4: (v= 'var' | v= 'ext' )
            int alt69=2;
            int LA69_0 = input.LA(1);

            if ( (LA69_0==163) ) {
                alt69=1;
            }
            else if ( (LA69_0==164) ) {
                alt69=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                NoViableAltException nvae =
                    new NoViableAltException("", 69, 0, input);

                throw nvae;
            }
            switch (alt69) {
                case 1 :
                    // EolParserRules.g:498:5: v= 'var'
                    {
                    v=(Token)match(input,163,FOLLOW_163_in_variableDeclarationExpression2249); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    v_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(v);
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(v_tree, root_0);
                    }

                    }
                    break;
                case 2 :
                    // EolParserRules.g:498:14: v= 'ext'
                    {
                    v=(Token)match(input,164,FOLLOW_164_in_variableDeclarationExpression2254); if (state.failed) return retval;
                    if ( state.backtracking==0 ) {
                    v_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(v);
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(v_tree, root_0);
                    }

                    }
                    break;

            }

            NAME145=(Token)match(input,NAME,FOLLOW_NAME_in_variableDeclarationExpression2258); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            NAME145_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(NAME145);
            adaptor.addChild(root_0, NAME145_tree);
            }
            // EolParserRules.g:498:29: ( ':' (n= 'new' )? t= typeName ( parameterList )? )?
            int alt72=2;
            alt72 = dfa72.predict(input);
            switch (alt72) {
                case 1 :
                    // EolParserRules.g:498:30: ':' (n= 'new' )? t= typeName ( parameterList )?
                    {
                    char_literal146=(Token)match(input,112,FOLLOW_112_in_variableDeclarationExpression2261); if (state.failed) return retval;
                    // EolParserRules.g:498:36: (n= 'new' )?
                    int alt70=2;
                    int LA70_0 = input.LA(1);

                    if ( (LA70_0==162) ) {
                        alt70=1;
                    }
                    switch (alt70) {
                        case 1 :
                            // EolParserRules.g:0:0: n= 'new'
                            {
                            n=(Token)match(input,162,FOLLOW_162_in_variableDeclarationExpression2266); if (state.failed) return retval;

                            }
                            break;

                    }

                    pushFollow(FOLLOW_typeName_in_variableDeclarationExpression2272);
                    t=typeName();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, t.getTree());
                    if ( state.backtracking==0 ) {
                      setTokenType(t, TYPE);
                    }
                    // EolParserRules.g:498:81: ( parameterList )?
                    int alt71=2;
                    int LA71_0 = input.LA(1);

                    if ( (LA71_0==110) ) {
                        alt71=1;
                    }
                    switch (alt71) {
                        case 1 :
                            // EolParserRules.g:0:0: parameterList
                            {
                            pushFollow(FOLLOW_parameterList_in_variableDeclarationExpression2276);
                            parameterList147=parameterList();

                            state._fsp--;
                            if (state.failed) return retval;
                            if ( state.backtracking==0 ) adaptor.addChild(root_0, parameterList147.getTree());

                            }
                            break;

                    }


                    }
                    break;

            }


            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		String txt = n != null ? "new" : v.getText();
              		retval.tree.getToken().setText(txt);
              		retval.tree.getToken().setType(VAR);
              	
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
    // $ANTLR end variableDeclarationExpression

    public static class literalSequentialCollection_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start literalSequentialCollection
    // EolParserRules.g:501:1: literalSequentialCollection : l= CollectionTypeName ob= '{' ( expressionListOrRange )? cb= '}' ;
    public final Pinset_EolParserRules.literalSequentialCollection_return literalSequentialCollection() throws RecognitionException {
        Pinset_EolParserRules.literalSequentialCollection_return retval = new Pinset_EolParserRules.literalSequentialCollection_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token l=null;
        Token ob=null;
        Token cb=null;
        Pinset_EolParserRules.expressionListOrRange_return expressionListOrRange148 = null;


        org.eclipse.epsilon.common.parse.AST l_tree=null;
        org.eclipse.epsilon.common.parse.AST ob_tree=null;
        org.eclipse.epsilon.common.parse.AST cb_tree=null;

        try {
            // EolParserRules.g:506:2: (l= CollectionTypeName ob= '{' ( expressionListOrRange )? cb= '}' )
            // EolParserRules.g:506:4: l= CollectionTypeName ob= '{' ( expressionListOrRange )? cb= '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            l=(Token)match(input,CollectionTypeName,FOLLOW_CollectionTypeName_in_literalSequentialCollection2299); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            l_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(l);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(l_tree, root_0);
            }
            ob=(Token)match(input,105,FOLLOW_105_in_literalSequentialCollection2306); if (state.failed) return retval;
            // EolParserRules.g:507:11: ( expressionListOrRange )?
            int alt73=2;
            int LA73_0 = input.LA(1);

            if ( (LA73_0==FLOAT||LA73_0==INT||LA73_0==BOOLEAN||LA73_0==STRING||(LA73_0>=CollectionTypeName && LA73_0<=SpecialTypeName)||LA73_0==NAME||LA73_0==110||LA73_0==152||LA73_0==155||(LA73_0>=162 && LA73_0<=164)) ) {
                alt73=1;
            }
            switch (alt73) {
                case 1 :
                    // EolParserRules.g:0:0: expressionListOrRange
                    {
                    pushFollow(FOLLOW_expressionListOrRange_in_literalSequentialCollection2309);
                    expressionListOrRange148=expressionListOrRange();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionListOrRange148.getTree());

                    }
                    break;

            }

            cb=(Token)match(input,106,FOLLOW_106_in_literalSequentialCollection2314); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              l.setType(COLLECTION);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(ob);
              		retval.tree.getExtraTokens().add(cb);
              	
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
    // $ANTLR end literalSequentialCollection

    public static class expressionRange_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start expressionRange
    // EolParserRules.g:511:1: expressionRange : logicalExpression exp= '..' logicalExpression ;
    public final Pinset_EolParserRules.expressionRange_return expressionRange() throws RecognitionException {
        Pinset_EolParserRules.expressionRange_return retval = new Pinset_EolParserRules.expressionRange_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token exp=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression149 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression150 = null;


        org.eclipse.epsilon.common.parse.AST exp_tree=null;

        try {
            // EolParserRules.g:512:2: ( logicalExpression exp= '..' logicalExpression )
            // EolParserRules.g:512:4: logicalExpression exp= '..' logicalExpression
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_logicalExpression_in_expressionRange2329);
            logicalExpression149=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression149.getTree());
            exp=(Token)match(input,POINT_POINT,FOLLOW_POINT_POINT_in_expressionRange2333); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            exp_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(exp);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(exp_tree, root_0);
            }
            pushFollow(FOLLOW_logicalExpression_in_expressionRange2336);
            logicalExpression150=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression150.getTree());
            if ( state.backtracking==0 ) {
              exp.setType(EXPRRANGE);
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
    // $ANTLR end expressionRange

    public static class expressionList_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start expressionList
    // EolParserRules.g:516:1: expressionList : logicalExpression ( ',' logicalExpression )* -> ^( EXPRLIST ( logicalExpression )+ ) ;
    public final Pinset_EolParserRules.expressionList_return expressionList() throws RecognitionException {
        Pinset_EolParserRules.expressionList_return retval = new Pinset_EolParserRules.expressionList_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token char_literal152=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression151 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression153 = null;


        org.eclipse.epsilon.common.parse.AST char_literal152_tree=null;
        RewriteRuleTokenStream stream_103=new RewriteRuleTokenStream(adaptor,"token 103");
        RewriteRuleSubtreeStream stream_logicalExpression=new RewriteRuleSubtreeStream(adaptor,"rule logicalExpression");
        try {
            // EolParserRules.g:520:2: ( logicalExpression ( ',' logicalExpression )* -> ^( EXPRLIST ( logicalExpression )+ ) )
            // EolParserRules.g:520:4: logicalExpression ( ',' logicalExpression )*
            {
            pushFollow(FOLLOW_logicalExpression_in_expressionList2357);
            logicalExpression151=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) stream_logicalExpression.add(logicalExpression151.getTree());
            // EolParserRules.g:520:22: ( ',' logicalExpression )*
            loop74:
            do {
                int alt74=2;
                int LA74_0 = input.LA(1);

                if ( (LA74_0==103) ) {
                    alt74=1;
                }


                switch (alt74) {
            	case 1 :
            	    // EolParserRules.g:520:23: ',' logicalExpression
            	    {
            	    char_literal152=(Token)match(input,103,FOLLOW_103_in_expressionList2360); if (state.failed) return retval; 
            	    if ( state.backtracking==0 ) stream_103.add(char_literal152);

            	    pushFollow(FOLLOW_logicalExpression_in_expressionList2362);
            	    logicalExpression153=logicalExpression();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) stream_logicalExpression.add(logicalExpression153.getTree());

            	    }
            	    break;

            	default :
            	    break loop74;
                }
            } while (true);



            // AST REWRITE
            // elements: logicalExpression
            // token labels: 
            // rule labels: retval
            // token list labels: 
            // rule list labels: 
            if ( state.backtracking==0 ) {
            retval.tree = root_0;
            RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"token retval",retval!=null?retval.tree:null);

            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
            // 521:2: -> ^( EXPRLIST ( logicalExpression )+ )
            {
                // EolParserRules.g:521:5: ^( EXPRLIST ( logicalExpression )+ )
                {
                org.eclipse.epsilon.common.parse.AST root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
                root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(EXPRLIST, "EXPRLIST"), root_1);

                if ( !(stream_logicalExpression.hasNext()) ) {
                    throw new RewriteEarlyExitException();
                }
                while ( stream_logicalExpression.hasNext() ) {
                    adaptor.addChild(root_1, stream_logicalExpression.nextTree());

                }
                stream_logicalExpression.reset();

                adaptor.addChild(root_0, root_1);
                }

            }

            retval.tree = root_0;retval.tree = root_0;}
            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.setImaginary(true);
              	
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
    // $ANTLR end expressionList

    public static class expressionListOrRange_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start expressionListOrRange
    // EolParserRules.g:524:1: expressionListOrRange : ( expressionRange | expressionList );
    public final Pinset_EolParserRules.expressionListOrRange_return expressionListOrRange() throws RecognitionException {
        Pinset_EolParserRules.expressionListOrRange_return retval = new Pinset_EolParserRules.expressionListOrRange_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.expressionRange_return expressionRange154 = null;

        Pinset_EolParserRules.expressionList_return expressionList155 = null;



        try {
            // EolParserRules.g:525:2: ( expressionRange | expressionList )
            int alt75=2;
            alt75 = dfa75.predict(input);
            switch (alt75) {
                case 1 :
                    // EolParserRules.g:525:4: expressionRange
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_expressionRange_in_expressionListOrRange2386);
                    expressionRange154=expressionRange();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionRange154.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:525:22: expressionList
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_expressionList_in_expressionListOrRange2390);
                    expressionList155=expressionList();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, expressionList155.getTree());

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
    // $ANTLR end expressionListOrRange

    public static class literalMapCollection_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start literalMapCollection
    // EolParserRules.g:528:1: literalMapCollection : m= MapTypeName ob= '{' ( keyvalExpressionList )? cb= '}' ;
    public final Pinset_EolParserRules.literalMapCollection_return literalMapCollection() throws RecognitionException {
        Pinset_EolParserRules.literalMapCollection_return retval = new Pinset_EolParserRules.literalMapCollection_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token m=null;
        Token ob=null;
        Token cb=null;
        Pinset_EolParserRules.keyvalExpressionList_return keyvalExpressionList156 = null;


        org.eclipse.epsilon.common.parse.AST m_tree=null;
        org.eclipse.epsilon.common.parse.AST ob_tree=null;
        org.eclipse.epsilon.common.parse.AST cb_tree=null;

        try {
            // EolParserRules.g:533:2: (m= MapTypeName ob= '{' ( keyvalExpressionList )? cb= '}' )
            // EolParserRules.g:533:4: m= MapTypeName ob= '{' ( keyvalExpressionList )? cb= '}'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            m=(Token)match(input,MapTypeName,FOLLOW_MapTypeName_in_literalMapCollection2409); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            m_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(m);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(m_tree, root_0);
            }
            ob=(Token)match(input,105,FOLLOW_105_in_literalMapCollection2414); if (state.failed) return retval;
            // EolParserRules.g:533:27: ( keyvalExpressionList )?
            int alt76=2;
            int LA76_0 = input.LA(1);

            if ( (LA76_0==FLOAT||LA76_0==INT||LA76_0==BOOLEAN||LA76_0==STRING||(LA76_0>=CollectionTypeName && LA76_0<=SpecialTypeName)||LA76_0==NAME||LA76_0==110||LA76_0==152||LA76_0==155||(LA76_0>=162 && LA76_0<=164)) ) {
                alt76=1;
            }
            switch (alt76) {
                case 1 :
                    // EolParserRules.g:0:0: keyvalExpressionList
                    {
                    pushFollow(FOLLOW_keyvalExpressionList_in_literalMapCollection2417);
                    keyvalExpressionList156=keyvalExpressionList();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, keyvalExpressionList156.getTree());

                    }
                    break;

            }

            cb=(Token)match(input,106,FOLLOW_106_in_literalMapCollection2422); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              m.setType(MAP);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(ob);
              		retval.tree.getExtraTokens().add(cb);
              	
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
    // $ANTLR end literalMapCollection

    public static class keyvalExpressionList_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start keyvalExpressionList
    // EolParserRules.g:537:1: keyvalExpressionList : keyvalExpression ( ',' keyvalExpression )* -> ^( KEYVALLIST ( keyvalExpression )+ ) ;
    public final Pinset_EolParserRules.keyvalExpressionList_return keyvalExpressionList() throws RecognitionException {
        Pinset_EolParserRules.keyvalExpressionList_return retval = new Pinset_EolParserRules.keyvalExpressionList_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token char_literal158=null;
        Pinset_EolParserRules.keyvalExpression_return keyvalExpression157 = null;

        Pinset_EolParserRules.keyvalExpression_return keyvalExpression159 = null;


        org.eclipse.epsilon.common.parse.AST char_literal158_tree=null;
        RewriteRuleTokenStream stream_103=new RewriteRuleTokenStream(adaptor,"token 103");
        RewriteRuleSubtreeStream stream_keyvalExpression=new RewriteRuleSubtreeStream(adaptor,"rule keyvalExpression");
        try {
            // EolParserRules.g:541:2: ( keyvalExpression ( ',' keyvalExpression )* -> ^( KEYVALLIST ( keyvalExpression )+ ) )
            // EolParserRules.g:541:4: keyvalExpression ( ',' keyvalExpression )*
            {
            pushFollow(FOLLOW_keyvalExpression_in_keyvalExpressionList2443);
            keyvalExpression157=keyvalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) stream_keyvalExpression.add(keyvalExpression157.getTree());
            // EolParserRules.g:541:21: ( ',' keyvalExpression )*
            loop77:
            do {
                int alt77=2;
                int LA77_0 = input.LA(1);

                if ( (LA77_0==103) ) {
                    alt77=1;
                }


                switch (alt77) {
            	case 1 :
            	    // EolParserRules.g:541:22: ',' keyvalExpression
            	    {
            	    char_literal158=(Token)match(input,103,FOLLOW_103_in_keyvalExpressionList2446); if (state.failed) return retval; 
            	    if ( state.backtracking==0 ) stream_103.add(char_literal158);

            	    pushFollow(FOLLOW_keyvalExpression_in_keyvalExpressionList2448);
            	    keyvalExpression159=keyvalExpression();

            	    state._fsp--;
            	    if (state.failed) return retval;
            	    if ( state.backtracking==0 ) stream_keyvalExpression.add(keyvalExpression159.getTree());

            	    }
            	    break;

            	default :
            	    break loop77;
                }
            } while (true);



            // AST REWRITE
            // elements: keyvalExpression
            // token labels: 
            // rule labels: retval
            // token list labels: 
            // rule list labels: 
            if ( state.backtracking==0 ) {
            retval.tree = root_0;
            RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"token retval",retval!=null?retval.tree:null);

            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
            // 542:2: -> ^( KEYVALLIST ( keyvalExpression )+ )
            {
                // EolParserRules.g:542:5: ^( KEYVALLIST ( keyvalExpression )+ )
                {
                org.eclipse.epsilon.common.parse.AST root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();
                root_1 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(adaptor.create(KEYVALLIST, "KEYVALLIST"), root_1);

                if ( !(stream_keyvalExpression.hasNext()) ) {
                    throw new RewriteEarlyExitException();
                }
                while ( stream_keyvalExpression.hasNext() ) {
                    adaptor.addChild(root_1, stream_keyvalExpression.nextTree());

                }
                stream_keyvalExpression.reset();

                adaptor.addChild(root_0, root_1);
                }

            }

            retval.tree = root_0;retval.tree = root_0;}
            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.setImaginary(true);
              	
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
    // $ANTLR end keyvalExpressionList

    public static class keyvalExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start keyvalExpression
    // EolParserRules.g:545:1: keyvalExpression : additiveExpression eq= '=' logicalExpression ;
    public final Pinset_EolParserRules.keyvalExpression_return keyvalExpression() throws RecognitionException {
        Pinset_EolParserRules.keyvalExpression_return retval = new Pinset_EolParserRules.keyvalExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token eq=null;
        Pinset_EolParserRules.additiveExpression_return additiveExpression160 = null;

        Pinset_EolParserRules.logicalExpression_return logicalExpression161 = null;


        org.eclipse.epsilon.common.parse.AST eq_tree=null;

        try {
            // EolParserRules.g:547:2: ( additiveExpression eq= '=' logicalExpression )
            // EolParserRules.g:547:4: additiveExpression eq= '=' logicalExpression
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            pushFollow(FOLLOW_additiveExpression_in_keyvalExpression2473);
            additiveExpression160=additiveExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, additiveExpression160.getTree());
            eq=(Token)match(input,107,FOLLOW_107_in_keyvalExpression2477); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            eq_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(eq);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(eq_tree, root_0);
            }
            pushFollow(FOLLOW_logicalExpression_in_keyvalExpression2480);
            logicalExpression161=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression161.getTree());
            if ( state.backtracking==0 ) {
              eq.setType(KEYVAL);
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
    // $ANTLR end keyvalExpression

    public static class primitiveExpression_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start primitiveExpression
    // EolParserRules.g:551:1: primitiveExpression : ( literalSequentialCollection | literalMapCollection | literal | featureCall | collectionType | pathName | specialType | logicalExpressionInBrackets | newExpression | variableDeclarationExpression );
    public final Pinset_EolParserRules.primitiveExpression_return primitiveExpression() throws RecognitionException {
        Pinset_EolParserRules.primitiveExpression_return retval = new Pinset_EolParserRules.primitiveExpression_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Pinset_EolParserRules.literalSequentialCollection_return literalSequentialCollection162 = null;

        Pinset_EolParserRules.literalMapCollection_return literalMapCollection163 = null;

        Pinset_EolParserRules.literal_return literal164 = null;

        Pinset_EolParserRules.featureCall_return featureCall165 = null;

        Pinset_EolParserRules.collectionType_return collectionType166 = null;

        Pinset_EolParserRules.pathName_return pathName167 = null;

        Pinset_EolParserRules.specialType_return specialType168 = null;

        Pinset_EolParserRules.logicalExpressionInBrackets_return logicalExpressionInBrackets169 = null;

        Pinset_EolParserRules.newExpression_return newExpression170 = null;

        Pinset_EolParserRules.variableDeclarationExpression_return variableDeclarationExpression171 = null;



        try {
            // EolParserRules.g:552:2: ( literalSequentialCollection | literalMapCollection | literal | featureCall | collectionType | pathName | specialType | logicalExpressionInBrackets | newExpression | variableDeclarationExpression )
            int alt78=10;
            alt78 = dfa78.predict(input);
            switch (alt78) {
                case 1 :
                    // EolParserRules.g:552:4: literalSequentialCollection
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_literalSequentialCollection_in_primitiveExpression2495);
                    literalSequentialCollection162=literalSequentialCollection();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, literalSequentialCollection162.getTree());

                    }
                    break;
                case 2 :
                    // EolParserRules.g:552:34: literalMapCollection
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_literalMapCollection_in_primitiveExpression2499);
                    literalMapCollection163=literalMapCollection();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, literalMapCollection163.getTree());

                    }
                    break;
                case 3 :
                    // EolParserRules.g:552:57: literal
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_literal_in_primitiveExpression2503);
                    literal164=literal();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, literal164.getTree());

                    }
                    break;
                case 4 :
                    // EolParserRules.g:552:67: featureCall
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_featureCall_in_primitiveExpression2507);
                    featureCall165=featureCall();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, featureCall165.getTree());

                    }
                    break;
                case 5 :
                    // EolParserRules.g:552:81: collectionType
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_collectionType_in_primitiveExpression2511);
                    collectionType166=collectionType();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, collectionType166.getTree());

                    }
                    break;
                case 6 :
                    // EolParserRules.g:553:3: pathName
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_pathName_in_primitiveExpression2517);
                    pathName167=pathName();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, pathName167.getTree());

                    }
                    break;
                case 7 :
                    // EolParserRules.g:553:14: specialType
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_specialType_in_primitiveExpression2521);
                    specialType168=specialType();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, specialType168.getTree());

                    }
                    break;
                case 8 :
                    // EolParserRules.g:553:28: logicalExpressionInBrackets
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_logicalExpressionInBrackets_in_primitiveExpression2525);
                    logicalExpressionInBrackets169=logicalExpressionInBrackets();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpressionInBrackets169.getTree());

                    }
                    break;
                case 9 :
                    // EolParserRules.g:553:58: newExpression
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_newExpression_in_primitiveExpression2529);
                    newExpression170=newExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, newExpression170.getTree());

                    }
                    break;
                case 10 :
                    // EolParserRules.g:553:74: variableDeclarationExpression
                    {
                    root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

                    pushFollow(FOLLOW_variableDeclarationExpression_in_primitiveExpression2533);
                    variableDeclarationExpression171=variableDeclarationExpression();

                    state._fsp--;
                    if (state.failed) return retval;
                    if ( state.backtracking==0 ) adaptor.addChild(root_0, variableDeclarationExpression171.getTree());

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
    // $ANTLR end primitiveExpression

    public static class logicalExpressionInBrackets_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start logicalExpressionInBrackets
    // EolParserRules.g:556:1: logicalExpressionInBrackets : ob= '(' logicalExpression cb= ')' ;
    public final Pinset_EolParserRules.logicalExpressionInBrackets_return logicalExpressionInBrackets() throws RecognitionException {
        Pinset_EolParserRules.logicalExpressionInBrackets_return retval = new Pinset_EolParserRules.logicalExpressionInBrackets_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token ob=null;
        Token cb=null;
        Pinset_EolParserRules.logicalExpression_return logicalExpression172 = null;


        org.eclipse.epsilon.common.parse.AST ob_tree=null;
        org.eclipse.epsilon.common.parse.AST cb_tree=null;

        try {
            // EolParserRules.g:562:2: (ob= '(' logicalExpression cb= ')' )
            // EolParserRules.g:562:4: ob= '(' logicalExpression cb= ')'
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            ob=(Token)match(input,110,FOLLOW_110_in_logicalExpressionInBrackets2552); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
            ob_tree = (org.eclipse.epsilon.common.parse.AST)adaptor.create(ob);
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.becomeRoot(ob_tree, root_0);
            }
            pushFollow(FOLLOW_logicalExpression_in_logicalExpressionInBrackets2555);
            logicalExpression172=logicalExpression();

            state._fsp--;
            if (state.failed) return retval;
            if ( state.backtracking==0 ) adaptor.addChild(root_0, logicalExpression172.getTree());
            cb=(Token)match(input,111,FOLLOW_111_in_logicalExpressionInBrackets2559); if (state.failed) return retval;
            if ( state.backtracking==0 ) {
              ob.setType(EXPRESSIONINBRACKETS);
            }

            }

            retval.stop = input.LT(-1);

            if ( state.backtracking==0 ) {

            retval.tree = (org.eclipse.epsilon.common.parse.AST)adaptor.rulePostProcessing(root_0);
            adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop);
            }
            if ( state.backtracking==0 ) {

              		retval.tree.getExtraTokens().add(ob);
              		retval.tree.getExtraTokens().add(cb);
              		retval.tree.setImaginary(true);
              	
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
    // $ANTLR end logicalExpressionInBrackets

    public static class literal_return extends ParserRuleReturnScope {
        org.eclipse.epsilon.common.parse.AST tree;
        @Override
		public Object getTree() { return tree; }
    };

    // $ANTLR start literal
    // EolParserRules.g:566:1: literal : ( STRING | INT | FLOAT | BOOLEAN );
    public final Pinset_EolParserRules.literal_return literal() throws RecognitionException {
        Pinset_EolParserRules.literal_return retval = new Pinset_EolParserRules.literal_return();
        retval.start = input.LT(1);

        org.eclipse.epsilon.common.parse.AST root_0 = null;

        Token set173=null;

        org.eclipse.epsilon.common.parse.AST set173_tree=null;

        try {
            // EolParserRules.g:567:2: ( STRING | INT | FLOAT | BOOLEAN )
            // EolParserRules.g:
            {
            root_0 = (org.eclipse.epsilon.common.parse.AST)adaptor.nil();

            set173=input.LT(1);
            if ( input.LA(1)==FLOAT||input.LA(1)==INT||input.LA(1)==BOOLEAN||input.LA(1)==STRING ) {
                input.consume();
                if ( state.backtracking==0 ) adaptor.addChild(root_0, adaptor.create(set173));
                state.errorRecovery=false;state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return retval;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                throw mse;
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
    // $ANTLR end literal

    // $ANTLR start synpred16_EolParserRules
    public final void synpred16_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:205:4: ( annotation )
        // EolParserRules.g:205:4: annotation
        {
        pushFollow(FOLLOW_annotation_in_synpred16_EolParserRules692);
        annotation();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred16_EolParserRules

    // $ANTLR start synpred24_EolParserRules
    public final void synpred24_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:258:4: ( ( '(' typeName ( ',' typeName )* ')' ) )
        // EolParserRules.g:258:4: ( '(' typeName ( ',' typeName )* ')' )
        {
        // EolParserRules.g:258:4: ( '(' typeName ( ',' typeName )* ')' )
        // EolParserRules.g:258:5: '(' typeName ( ',' typeName )* ')'
        {
        match(input,110,FOLLOW_110_in_synpred24_EolParserRules869); if (state.failed) return ;
        pushFollow(FOLLOW_typeName_in_synpred24_EolParserRules874);
        typeName();

        state._fsp--;
        if (state.failed) return ;
        // EolParserRules.g:258:50: ( ',' typeName )*
        loop79:
        do {
            int alt79=2;
            int LA79_0 = input.LA(1);

            if ( (LA79_0==103) ) {
                alt79=1;
            }


            switch (alt79) {
        	case 1 :
        	    // EolParserRules.g:258:51: ',' typeName
        	    {
        	    match(input,103,FOLLOW_103_in_synpred24_EolParserRules879); if (state.failed) return ;
        	    pushFollow(FOLLOW_typeName_in_synpred24_EolParserRules883);
        	    typeName();

        	    state._fsp--;
        	    if (state.failed) return ;

        	    }
        	    break;

        	default :
        	    break loop79;
            }
        } while (true);

        match(input,111,FOLLOW_111_in_synpred24_EolParserRules891); if (state.failed) return ;

        }


        }
    }
    // $ANTLR end synpred24_EolParserRules

    // $ANTLR start synpred26_EolParserRules
    public final void synpred26_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:259:4: ( ( '<' typeName ( ',' typeName )* '>' ) )
        // EolParserRules.g:259:4: ( '<' typeName ( ',' typeName )* '>' )
        {
        // EolParserRules.g:259:4: ( '<' typeName ( ',' typeName )* '>' )
        // EolParserRules.g:259:5: '<' typeName ( ',' typeName )* '>'
        {
        match(input,118,FOLLOW_118_in_synpred26_EolParserRules903); if (state.failed) return ;
        pushFollow(FOLLOW_typeName_in_synpred26_EolParserRules908);
        typeName();

        state._fsp--;
        if (state.failed) return ;
        // EolParserRules.g:259:50: ( ',' typeName )*
        loop80:
        do {
            int alt80=2;
            int LA80_0 = input.LA(1);

            if ( (LA80_0==103) ) {
                alt80=1;
            }


            switch (alt80) {
        	case 1 :
        	    // EolParserRules.g:259:51: ',' typeName
        	    {
        	    match(input,103,FOLLOW_103_in_synpred26_EolParserRules913); if (state.failed) return ;
        	    pushFollow(FOLLOW_typeName_in_synpred26_EolParserRules917);
        	    typeName();

        	    state._fsp--;
        	    if (state.failed) return ;

        	    }
        	    break;

        	default :
        	    break loop80;
            }
        } while (true);

        match(input,119,FOLLOW_119_in_synpred26_EolParserRules925); if (state.failed) return ;

        }


        }
    }
    // $ANTLR end synpred26_EolParserRules

    // $ANTLR start synpred27_EolParserRules
    public final void synpred27_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:264:4: ( statementA )
        // EolParserRules.g:264:4: statementA
        {
        pushFollow(FOLLOW_statementA_in_synpred27_EolParserRules944);
        statementA();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred27_EolParserRules

    // $ANTLR start synpred28_EolParserRules
    public final void synpred28_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:268:4: ( assignmentStatement )
        // EolParserRules.g:268:4: assignmentStatement
        {
        pushFollow(FOLLOW_assignmentStatement_in_synpred28_EolParserRules959);
        assignmentStatement();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred28_EolParserRules

    // $ANTLR start synpred29_EolParserRules
    public final void synpred29_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:268:26: ( expressionStatement )
        // EolParserRules.g:268:26: expressionStatement
        {
        pushFollow(FOLLOW_expressionStatement_in_synpred29_EolParserRules963);
        expressionStatement();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred29_EolParserRules

    // $ANTLR start synpred43_EolParserRules
    public final void synpred43_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:286:66: ( elseStatement )
        // EolParserRules.g:286:66: elseStatement
        {
        pushFollow(FOLLOW_elseStatement_in_synpred43_EolParserRules1086);
        elseStatement();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred43_EolParserRules

    // $ANTLR start synpred52_EolParserRules
    public final void synpred52_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:379:22: ( NAME ( ',' NAME )* )
        // EolParserRules.g:379:22: NAME ( ',' NAME )*
        {
        match(input,NAME,FOLLOW_NAME_in_synpred52_EolParserRules1497); if (state.failed) return ;
        // EolParserRules.g:379:27: ( ',' NAME )*
        loop81:
        do {
            int alt81=2;
            int LA81_0 = input.LA(1);

            if ( (LA81_0==103) ) {
                alt81=1;
            }


            switch (alt81) {
        	case 1 :
        	    // EolParserRules.g:379:28: ',' NAME
        	    {
        	    match(input,103,FOLLOW_103_in_synpred52_EolParserRules1500); if (state.failed) return ;
        	    match(input,NAME,FOLLOW_NAME_in_synpred52_EolParserRules1502); if (state.failed) return ;

        	    }
        	    break;

        	default :
        	    break loop81;
            }
        } while (true);


        }
    }
    // $ANTLR end synpred52_EolParserRules

    // $ANTLR start synpred58_EolParserRules
    public final void synpred58_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:396:5: ( ( postfixExpression '=' logicalExpression ) )
        // EolParserRules.g:396:5: ( postfixExpression '=' logicalExpression )
        {
        // EolParserRules.g:396:5: ( postfixExpression '=' logicalExpression )
        // EolParserRules.g:396:6: postfixExpression '=' logicalExpression
        {
        pushFollow(FOLLOW_postfixExpression_in_synpred58_EolParserRules1598);
        postfixExpression();

        state._fsp--;
        if (state.failed) return ;
        match(input,107,FOLLOW_107_in_synpred58_EolParserRules1602); if (state.failed) return ;
        pushFollow(FOLLOW_logicalExpression_in_synpred58_EolParserRules1605);
        logicalExpression();

        state._fsp--;
        if (state.failed) return ;

        }


        }
    }
    // $ANTLR end synpred58_EolParserRules

    // $ANTLR start synpred71_EolParserRules
    public final void synpred71_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:408:24: ( ( '==' relationalExpression | '=' relationalExpression | ( '>' | '<' | '>=' | '<=' | '<>' ) additiveExpression ) )
        // EolParserRules.g:408:24: ( '==' relationalExpression | '=' relationalExpression | ( '>' | '<' | '>=' | '<=' | '<>' ) additiveExpression )
        {
        // EolParserRules.g:408:24: ( '==' relationalExpression | '=' relationalExpression | ( '>' | '<' | '>=' | '<=' | '<>' ) additiveExpression )
        int alt83=3;
        switch ( input.LA(1) ) {
        case 147:
            {
            alt83=1;
            }
            break;
        case 107:
            {
            alt83=2;
            }
            break;
        case 118:
        case 119:
        case 148:
        case 149:
        case 150:
            {
            alt83=3;
            }
            break;
        default:
            if (state.backtracking>0) {state.failed=true; return ;}
            NoViableAltException nvae =
                new NoViableAltException("", 83, 0, input);

            throw nvae;
        }

        switch (alt83) {
            case 1 :
                // EolParserRules.g:408:25: '==' relationalExpression
                {
                match(input,147,FOLLOW_147_in_synpred71_EolParserRules1712); if (state.failed) return ;
                pushFollow(FOLLOW_relationalExpression_in_synpred71_EolParserRules1715);
                relationalExpression();

                state._fsp--;
                if (state.failed) return ;

                }
                break;
            case 2 :
                // EolParserRules.g:408:57: '=' relationalExpression
                {
                match(input,107,FOLLOW_107_in_synpred71_EolParserRules1721); if (state.failed) return ;
                pushFollow(FOLLOW_relationalExpression_in_synpred71_EolParserRules1724);
                relationalExpression();

                state._fsp--;
                if (state.failed) return ;

                }
                break;
            case 3 :
                // EolParserRules.g:409:24: ( '>' | '<' | '>=' | '<=' | '<>' ) additiveExpression
                {
                if ( (input.LA(1)>=118 && input.LA(1)<=119)||(input.LA(1)>=148 && input.LA(1)<=150) ) {
                    input.consume();
                    state.errorRecovery=false;state.failed=false;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    MismatchedSetException mse = new MismatchedSetException(null,input);
                    throw mse;
                }

                pushFollow(FOLLOW_additiveExpression_in_synpred71_EolParserRules1778);
                additiveExpression();

                state._fsp--;
                if (state.failed) return ;

                }
                break;

        }


        }
    }
    // $ANTLR end synpred71_EolParserRules

    // $ANTLR start synpred99_EolParserRules
    public final void synpred99_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:498:30: ( ':' ( 'new' )? typeName ( parameterList )? )
        // EolParserRules.g:498:30: ':' ( 'new' )? typeName ( parameterList )?
        {
        match(input,112,FOLLOW_112_in_synpred99_EolParserRules2261); if (state.failed) return ;
        // EolParserRules.g:498:36: ( 'new' )?
        int alt87=2;
        int LA87_0 = input.LA(1);

        if ( (LA87_0==162) ) {
            alt87=1;
        }
        switch (alt87) {
            case 1 :
                // EolParserRules.g:0:0: 'new'
                {
                match(input,162,FOLLOW_162_in_synpred99_EolParserRules2266); if (state.failed) return ;

                }
                break;

        }

        pushFollow(FOLLOW_typeName_in_synpred99_EolParserRules2272);
        typeName();

        state._fsp--;
        if (state.failed) return ;
        // EolParserRules.g:498:81: ( parameterList )?
        int alt88=2;
        int LA88_0 = input.LA(1);

        if ( (LA88_0==110) ) {
            alt88=1;
        }
        switch (alt88) {
            case 1 :
                // EolParserRules.g:0:0: parameterList
                {
                pushFollow(FOLLOW_parameterList_in_synpred99_EolParserRules2276);
                parameterList();

                state._fsp--;
                if (state.failed) return ;

                }
                break;

        }


        }
    }
    // $ANTLR end synpred99_EolParserRules

    // $ANTLR start synpred102_EolParserRules
    public final void synpred102_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:525:4: ( expressionRange )
        // EolParserRules.g:525:4: expressionRange
        {
        pushFollow(FOLLOW_expressionRange_in_synpred102_EolParserRules2386);
        expressionRange();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred102_EolParserRules

    // $ANTLR start synpred105_EolParserRules
    public final void synpred105_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:552:4: ( literalSequentialCollection )
        // EolParserRules.g:552:4: literalSequentialCollection
        {
        pushFollow(FOLLOW_literalSequentialCollection_in_synpred105_EolParserRules2495);
        literalSequentialCollection();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred105_EolParserRules

    // $ANTLR start synpred106_EolParserRules
    public final void synpred106_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:552:34: ( literalMapCollection )
        // EolParserRules.g:552:34: literalMapCollection
        {
        pushFollow(FOLLOW_literalMapCollection_in_synpred106_EolParserRules2499);
        literalMapCollection();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred106_EolParserRules

    // $ANTLR start synpred108_EolParserRules
    public final void synpred108_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:552:67: ( featureCall )
        // EolParserRules.g:552:67: featureCall
        {
        pushFollow(FOLLOW_featureCall_in_synpred108_EolParserRules2507);
        featureCall();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred108_EolParserRules

    // $ANTLR start synpred109_EolParserRules
    public final void synpred109_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:552:81: ( collectionType )
        // EolParserRules.g:552:81: collectionType
        {
        pushFollow(FOLLOW_collectionType_in_synpred109_EolParserRules2511);
        collectionType();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred109_EolParserRules

    // $ANTLR start synpred110_EolParserRules
    public final void synpred110_EolParserRules_fragment() throws RecognitionException {   
        // EolParserRules.g:553:3: ( pathName )
        // EolParserRules.g:553:3: pathName
        {
        pushFollow(FOLLOW_pathName_in_synpred110_EolParserRules2517);
        pathName();

        state._fsp--;
        if (state.failed) return ;

        }
    }
    // $ANTLR end synpred110_EolParserRules

    // Delegated rules

    public final boolean synpred52_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred52_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred71_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred71_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred110_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred110_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred16_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred16_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred102_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred102_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred29_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred29_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred108_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred108_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred27_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred27_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred106_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred106_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred26_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred26_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred58_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred58_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred105_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred105_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred28_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred28_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred43_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred43_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred109_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred109_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred24_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred24_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred99_EolParserRules() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred99_EolParserRules_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }


    protected DFA22 dfa22 = new DFA22(this);
    protected DFA23 dfa23 = new DFA23(this);
    protected DFA24 dfa24 = new DFA24(this);
    protected DFA37 dfa37 = new DFA37(this);
    protected DFA40 dfa40 = new DFA40(this);
    protected DFA46 dfa46 = new DFA46(this);
    protected DFA58 dfa58 = new DFA58(this);
    protected DFA72 dfa72 = new DFA72(this);
    protected DFA75 dfa75 = new DFA75(this);
    protected DFA78 dfa78 = new DFA78(this);
    static final String DFA22_eotS =
        "\71\uffff";
    static final String DFA22_eofS =
        "\1\3\70\uffff";
    static final String DFA22_minS =
        "\1\11\2\0\66\uffff";
    static final String DFA22_maxS =
        "\1\u00b3\2\0\66\uffff";
    static final String DFA22_acceptS =
        "\3\uffff\1\3\63\uffff\1\1\1\2";
    static final String DFA22_specialS =
        "\1\uffff\1\0\1\1\66\uffff}>";
    static final String[] DFA22_transitionS = {
            "\4\3\12\uffff\1\3\3\uffff\1\3\111\uffff\1\3\1\uffff\1\3\1\uffff"+
            "\5\3\1\1\2\3\1\uffff\1\3\3\uffff\1\2\1\3\1\uffff\1\3\4\uffff"+
            "\1\3\11\uffff\23\3\1\uffff\6\3\3\uffff\2\3\2\uffff\1\3\1\uffff"+
            "\1\3\1\uffff\4\3\1\uffff\2\3",
            "\1\uffff",
            "\1\uffff",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
    };

    static final short[] DFA22_eot = DFA.unpackEncodedString(DFA22_eotS);
    static final short[] DFA22_eof = DFA.unpackEncodedString(DFA22_eofS);
    static final char[] DFA22_min = DFA.unpackEncodedStringToUnsignedChars(DFA22_minS);
    static final char[] DFA22_max = DFA.unpackEncodedStringToUnsignedChars(DFA22_maxS);
    static final short[] DFA22_accept = DFA.unpackEncodedString(DFA22_acceptS);
    static final short[] DFA22_special = DFA.unpackEncodedString(DFA22_specialS);
    static final short[][] DFA22_transition;

    static {
        int numStates = DFA22_transitionS.length;
        DFA22_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA22_transition[i] = DFA.unpackEncodedString(DFA22_transitionS[i]);
        }
    }

    class DFA22 extends DFA {

        public DFA22(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 22;
            this.eot = DFA22_eot;
            this.eof = DFA22_eof;
            this.min = DFA22_min;
            this.max = DFA22_max;
            this.accept = DFA22_accept;
            this.special = DFA22_special;
            this.transition = DFA22_transition;
        }
        @Override
		public String getDescription() {
            return "258:3: ( (op= '(' tn= typeName ( ',' tn= typeName )* cp= ')' ) | (op= '<' tn= typeName ( ',' tn= typeName )* cp= '>' ) )?";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA22_1 = input.LA(1);

                         
                        int index22_1 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred24_EolParserRules()) ) {s = 55;}

                        else if ( (true) ) {s = 3;}

                         
                        input.seek(index22_1);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA22_2 = input.LA(1);

                         
                        int index22_2 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred26_EolParserRules()) ) {s = 56;}

                        else if ( (true) ) {s = 3;}

                         
                        input.seek(index22_2);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 22, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA23_eotS =
        "\30\uffff";
    static final String DFA23_eofS =
        "\30\uffff";
    static final String DFA23_minS =
        "\1\4\17\uffff\1\0\7\uffff";
    static final String DFA23_maxS =
        "\1\u00a4\17\uffff\1\0\7\uffff";
    static final String DFA23_acceptS =
        "\1\uffff\1\1\20\uffff\1\2\5\uffff";
    static final String DFA23_specialS =
        "\20\uffff\1\0\7\uffff}>";
    static final String[] DFA23_transitionS = {
            "\1\1\3\uffff\1\1\4\uffff\1\1\1\uffff\1\1\1\uffff\3\1\3\uffff"+
            "\1\1\126\uffff\1\1\11\uffff\1\1\1\uffff\1\1\2\uffff\1\1\1\uffff"+
            "\1\1\1\20\2\22\1\1\4\22\20\uffff\1\1\2\uffff\1\1\6\uffff\3\1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "\1\uffff",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
    };

    static final short[] DFA23_eot = DFA.unpackEncodedString(DFA23_eotS);
    static final short[] DFA23_eof = DFA.unpackEncodedString(DFA23_eofS);
    static final char[] DFA23_min = DFA.unpackEncodedStringToUnsignedChars(DFA23_minS);
    static final char[] DFA23_max = DFA.unpackEncodedStringToUnsignedChars(DFA23_maxS);
    static final short[] DFA23_accept = DFA.unpackEncodedString(DFA23_acceptS);
    static final short[] DFA23_special = DFA.unpackEncodedString(DFA23_specialS);
    static final short[][] DFA23_transition;

    static {
        int numStates = DFA23_transitionS.length;
        DFA23_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA23_transition[i] = DFA.unpackEncodedString(DFA23_transitionS[i]);
        }
    }

    class DFA23 extends DFA {

        public DFA23(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 23;
            this.eot = DFA23_eot;
            this.eof = DFA23_eof;
            this.min = DFA23_min;
            this.max = DFA23_max;
            this.accept = DFA23_accept;
            this.special = DFA23_special;
            this.transition = DFA23_transition;
        }
        @Override
		public String getDescription() {
            return "263:1: statement : ( statementA | statementB );";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA23_16 = input.LA(1);

                         
                        int index23_16 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred27_EolParserRules()) ) {s = 1;}

                        else if ( (true) ) {s = 18;}

                         
                        input.seek(index23_16);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 23, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA24_eotS =
        "\24\uffff";
    static final String DFA24_eofS =
        "\24\uffff";
    static final String DFA24_minS =
        "\1\4\13\0\10\uffff";
    static final String DFA24_maxS =
        "\1\u00a4\13\0\10\uffff";
    static final String DFA24_acceptS =
        "\14\uffff\1\3\1\4\1\5\1\6\1\7\1\10\1\1\1\2";
    static final String DFA24_specialS =
        "\1\uffff\1\0\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\12\10\uffff}>";
    static final String[] DFA24_transitionS = {
            "\1\5\3\uffff\1\5\4\uffff\1\5\1\uffff\1\5\1\uffff\1\3\1\4\1\7"+
            "\3\uffff\1\6\126\uffff\1\10\11\uffff\1\15\1\uffff\1\17\2\uffff"+
            "\1\14\1\uffff\1\16\1\20\2\uffff\1\21\24\uffff\1\2\2\uffff\1"+
            "\1\6\uffff\1\11\1\12\1\13",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
    };

    static final short[] DFA24_eot = DFA.unpackEncodedString(DFA24_eotS);
    static final short[] DFA24_eof = DFA.unpackEncodedString(DFA24_eofS);
    static final char[] DFA24_min = DFA.unpackEncodedStringToUnsignedChars(DFA24_minS);
    static final char[] DFA24_max = DFA.unpackEncodedStringToUnsignedChars(DFA24_maxS);
    static final short[] DFA24_accept = DFA.unpackEncodedString(DFA24_acceptS);
    static final short[] DFA24_special = DFA.unpackEncodedString(DFA24_specialS);
    static final short[][] DFA24_transition;

    static {
        int numStates = DFA24_transitionS.length;
        DFA24_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA24_transition[i] = DFA.unpackEncodedString(DFA24_transitionS[i]);
        }
    }

    class DFA24 extends DFA {

        public DFA24(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 24;
            this.eot = DFA24_eot;
            this.eof = DFA24_eof;
            this.min = DFA24_min;
            this.max = DFA24_max;
            this.accept = DFA24_accept;
            this.special = DFA24_special;
            this.transition = DFA24_transition;
        }
        @Override
		public String getDescription() {
            return "267:1: statementA : ( assignmentStatement | expressionStatement | forStatement | ifStatement | whileStatement | switchStatement | returnStatement | breakStatement );";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA24_1 = input.LA(1);

                         
                        int index24_1 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_1);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA24_2 = input.LA(1);

                         
                        int index24_2 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_2);
                        if ( s>=0 ) return s;
                        break;
                    case 2 : 
                        int LA24_3 = input.LA(1);

                         
                        int index24_3 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_3);
                        if ( s>=0 ) return s;
                        break;
                    case 3 : 
                        int LA24_4 = input.LA(1);

                         
                        int index24_4 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_4);
                        if ( s>=0 ) return s;
                        break;
                    case 4 : 
                        int LA24_5 = input.LA(1);

                         
                        int index24_5 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_5);
                        if ( s>=0 ) return s;
                        break;
                    case 5 : 
                        int LA24_6 = input.LA(1);

                         
                        int index24_6 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_6);
                        if ( s>=0 ) return s;
                        break;
                    case 6 : 
                        int LA24_7 = input.LA(1);

                         
                        int index24_7 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_7);
                        if ( s>=0 ) return s;
                        break;
                    case 7 : 
                        int LA24_8 = input.LA(1);

                         
                        int index24_8 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_8);
                        if ( s>=0 ) return s;
                        break;
                    case 8 : 
                        int LA24_9 = input.LA(1);

                         
                        int index24_9 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_9);
                        if ( s>=0 ) return s;
                        break;
                    case 9 : 
                        int LA24_10 = input.LA(1);

                         
                        int index24_10 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_10);
                        if ( s>=0 ) return s;
                        break;
                    case 10 : 
                        int LA24_11 = input.LA(1);

                         
                        int index24_11 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred28_EolParserRules()) ) {s = 18;}

                        else if ( (synpred29_EolParserRules()) ) {s = 19;}

                         
                        input.seek(index24_11);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 24, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA37_eotS =
        "\32\uffff";
    static final String DFA37_eofS =
        "\32\uffff";
    static final String DFA37_minS =
        "\1\4\1\0\30\uffff";
    static final String DFA37_maxS =
        "\1\u00a4\1\0\30\uffff";
    static final String DFA37_acceptS =
        "\2\uffff\1\2\26\uffff\1\1";
    static final String DFA37_specialS =
        "\1\uffff\1\0\30\uffff}>";
    static final String[] DFA37_transitionS = {
            "\1\2\3\uffff\1\2\4\uffff\1\2\1\uffff\1\2\1\uffff\3\2\3\uffff"+
            "\1\1\121\uffff\1\2\4\uffff\1\2\11\uffff\1\2\1\uffff\1\2\2\uffff"+
            "\1\2\1\uffff\11\2\20\uffff\1\2\2\uffff\1\2\6\uffff\3\2",
            "\1\uffff",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
    };

    static final short[] DFA37_eot = DFA.unpackEncodedString(DFA37_eotS);
    static final short[] DFA37_eof = DFA.unpackEncodedString(DFA37_eofS);
    static final char[] DFA37_min = DFA.unpackEncodedStringToUnsignedChars(DFA37_minS);
    static final char[] DFA37_max = DFA.unpackEncodedStringToUnsignedChars(DFA37_maxS);
    static final short[] DFA37_accept = DFA.unpackEncodedString(DFA37_acceptS);
    static final short[] DFA37_special = DFA.unpackEncodedString(DFA37_specialS);
    static final short[][] DFA37_transition;

    static {
        int numStates = DFA37_transitionS.length;
        DFA37_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA37_transition[i] = DFA.unpackEncodedString(DFA37_transitionS[i]);
        }
    }

    class DFA37 extends DFA {

        public DFA37(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 37;
            this.eot = DFA37_eot;
            this.eof = DFA37_eof;
            this.min = DFA37_min;
            this.max = DFA37_max;
            this.accept = DFA37_accept;
            this.special = DFA37_special;
            this.transition = DFA37_transition;
        }
        @Override
		public String getDescription() {
            return "379:21: ( NAME ( ',' NAME )* )?";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA37_1 = input.LA(1);

                         
                        int index37_1 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred52_EolParserRules()) ) {s = 25;}

                        else if ( (true) ) {s = 2;}

                         
                        input.seek(index37_1);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 37, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA40_eotS =
        "\15\uffff";
    static final String DFA40_eofS =
        "\15\uffff";
    static final String DFA40_minS =
        "\1\4\11\0\3\uffff";
    static final String DFA40_maxS =
        "\1\u00a4\11\0\3\uffff";
    static final String DFA40_acceptS =
        "\12\uffff\1\2\1\uffff\1\1";
    static final String DFA40_specialS =
        "\1\uffff\1\0\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\3\uffff}>";
    static final String[] DFA40_transitionS = {
            "\1\3\3\uffff\1\3\4\uffff\1\3\1\uffff\1\3\1\uffff\1\1\1\2\1\5"+
            "\3\uffff\1\4\126\uffff\1\6\51\uffff\1\12\2\uffff\1\12\6\uffff"+
            "\1\7\1\10\1\11",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "",
            "",
            ""
    };

    static final short[] DFA40_eot = DFA.unpackEncodedString(DFA40_eotS);
    static final short[] DFA40_eof = DFA.unpackEncodedString(DFA40_eofS);
    static final char[] DFA40_min = DFA.unpackEncodedStringToUnsignedChars(DFA40_minS);
    static final char[] DFA40_max = DFA.unpackEncodedStringToUnsignedChars(DFA40_maxS);
    static final short[] DFA40_accept = DFA.unpackEncodedString(DFA40_acceptS);
    static final short[] DFA40_special = DFA.unpackEncodedString(DFA40_specialS);
    static final short[][] DFA40_transition;

    static {
        int numStates = DFA40_transitionS.length;
        DFA40_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA40_transition[i] = DFA.unpackEncodedString(DFA40_transitionS[i]);
        }
    }

    class DFA40 extends DFA {

        public DFA40(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 40;
            this.eot = DFA40_eot;
            this.eof = DFA40_eof;
            this.min = DFA40_min;
            this.max = DFA40_max;
            this.accept = DFA40_accept;
            this.special = DFA40_special;
            this.transition = DFA40_transition;
        }
        @Override
		public String getDescription() {
            return "396:4: ( ( postfixExpression op= '=' logicalExpression ) | logicalExpression )";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA40_1 = input.LA(1);

                         
                        int index40_1 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_1);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA40_2 = input.LA(1);

                         
                        int index40_2 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_2);
                        if ( s>=0 ) return s;
                        break;
                    case 2 : 
                        int LA40_3 = input.LA(1);

                         
                        int index40_3 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_3);
                        if ( s>=0 ) return s;
                        break;
                    case 3 : 
                        int LA40_4 = input.LA(1);

                         
                        int index40_4 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_4);
                        if ( s>=0 ) return s;
                        break;
                    case 4 : 
                        int LA40_5 = input.LA(1);

                         
                        int index40_5 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_5);
                        if ( s>=0 ) return s;
                        break;
                    case 5 : 
                        int LA40_6 = input.LA(1);

                         
                        int index40_6 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_6);
                        if ( s>=0 ) return s;
                        break;
                    case 6 : 
                        int LA40_7 = input.LA(1);

                         
                        int index40_7 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_7);
                        if ( s>=0 ) return s;
                        break;
                    case 7 : 
                        int LA40_8 = input.LA(1);

                         
                        int index40_8 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_8);
                        if ( s>=0 ) return s;
                        break;
                    case 8 : 
                        int LA40_9 = input.LA(1);

                         
                        int index40_9 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred58_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 10;}

                         
                        input.seek(index40_9);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 40, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA46_eotS =
        "\12\uffff";
    static final String DFA46_eofS =
        "\1\1\11\uffff";
    static final String DFA46_minS =
        "\1\12\1\uffff\7\0\1\uffff";
    static final String DFA46_maxS =
        "\1\u00b3\1\uffff\7\0\1\uffff";
    static final String DFA46_acceptS =
        "\1\uffff\1\2\7\uffff\1\1";
    static final String DFA46_specialS =
        "\2\uffff\1\4\1\3\1\1\1\5\1\2\1\6\1\0\1\uffff}>";
    static final String[] DFA46_transitionS = {
            "\1\1\20\uffff\1\1\111\uffff\1\1\1\uffff\1\1\1\uffff\2\1\1\3"+
            "\2\1\1\uffff\2\1\1\uffff\1\1\3\uffff\1\5\1\4\1\uffff\1\1\16"+
            "\uffff\13\1\1\2\1\6\1\7\1\10\10\uffff\1\1\5\uffff\2\1\2\uffff"+
            "\1\1\1\uffff\1\1\1\uffff\4\1\1\uffff\2\1",
            "",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            ""
    };

    static final short[] DFA46_eot = DFA.unpackEncodedString(DFA46_eotS);
    static final short[] DFA46_eof = DFA.unpackEncodedString(DFA46_eofS);
    static final char[] DFA46_min = DFA.unpackEncodedStringToUnsignedChars(DFA46_minS);
    static final char[] DFA46_max = DFA.unpackEncodedStringToUnsignedChars(DFA46_maxS);
    static final short[] DFA46_accept = DFA.unpackEncodedString(DFA46_acceptS);
    static final short[] DFA46_special = DFA.unpackEncodedString(DFA46_specialS);
    static final short[][] DFA46_transition;

    static {
        int numStates = DFA46_transitionS.length;
        DFA46_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA46_transition[i] = DFA.unpackEncodedString(DFA46_transitionS[i]);
        }
    }

    class DFA46 extends DFA {

        public DFA46(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 46;
            this.eot = DFA46_eot;
            this.eof = DFA46_eof;
            this.min = DFA46_min;
            this.max = DFA46_max;
            this.accept = DFA46_accept;
            this.special = DFA46_special;
            this.transition = DFA46_transition;
        }
        @Override
		public String getDescription() {
            return "()* loopback of 408:23: ( (op= '==' relationalExpression | op= '=' relationalExpression | (op= '>' | op= '<' | op= '>=' | op= '<=' | op= '<>' ) additiveExpression ) )*";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA46_8 = input.LA(1);

                         
                        int index46_8 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred71_EolParserRules()) ) {s = 9;}

                        else if ( (true) ) {s = 1;}

                         
                        input.seek(index46_8);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA46_4 = input.LA(1);

                         
                        int index46_4 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred71_EolParserRules()) ) {s = 9;}

                        else if ( (true) ) {s = 1;}

                         
                        input.seek(index46_4);
                        if ( s>=0 ) return s;
                        break;
                    case 2 : 
                        int LA46_6 = input.LA(1);

                         
                        int index46_6 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred71_EolParserRules()) ) {s = 9;}

                        else if ( (true) ) {s = 1;}

                         
                        input.seek(index46_6);
                        if ( s>=0 ) return s;
                        break;
                    case 3 : 
                        int LA46_3 = input.LA(1);

                         
                        int index46_3 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred71_EolParserRules()) ) {s = 9;}

                        else if ( (true) ) {s = 1;}

                         
                        input.seek(index46_3);
                        if ( s>=0 ) return s;
                        break;
                    case 4 : 
                        int LA46_2 = input.LA(1);

                         
                        int index46_2 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred71_EolParserRules()) ) {s = 9;}

                        else if ( (true) ) {s = 1;}

                         
                        input.seek(index46_2);
                        if ( s>=0 ) return s;
                        break;
                    case 5 : 
                        int LA46_5 = input.LA(1);

                         
                        int index46_5 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred71_EolParserRules()) ) {s = 9;}

                        else if ( (true) ) {s = 1;}

                         
                        input.seek(index46_5);
                        if ( s>=0 ) return s;
                        break;
                    case 6 : 
                        int LA46_7 = input.LA(1);

                         
                        int index46_7 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred71_EolParserRules()) ) {s = 9;}

                        else if ( (true) ) {s = 1;}

                         
                        input.seek(index46_7);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 46, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA58_eotS =
        "\12\uffff";
    static final String DFA58_eofS =
        "\1\uffff\1\3\10\uffff";
    static final String DFA58_minS =
        "\1\27\1\11\1\4\1\uffff\1\11\1\4\1\uffff\1\4\2\11";
    static final String DFA58_maxS =
        "\1\27\1\u00b3\1\u00a4\1\uffff\1\u00a1\1\u00a4\1\uffff\1\u00a4\2"+
        "\u00a1";
    static final String DFA58_acceptS =
        "\3\uffff\1\1\2\uffff\1\2\3\uffff";
    static final String DFA58_specialS =
        "\12\uffff}>";
    static final String[] DFA58_transitionS = {
            "\1\1",
            "\4\3\16\uffff\1\3\111\uffff\1\3\1\uffff\1\3\1\uffff\5\3\1\2"+
            "\2\3\1\uffff\1\3\3\uffff\2\3\1\uffff\1\3\16\uffff\23\3\1\uffff"+
            "\4\3\5\uffff\2\3\2\uffff\1\3\1\uffff\1\3\1\uffff\4\3\1\uffff"+
            "\2\3",
            "\1\3\3\uffff\1\3\4\uffff\1\3\1\uffff\1\3\1\uffff\3\3\3\uffff"+
            "\1\4\126\uffff\1\5\1\3\50\uffff\1\3\2\uffff\1\3\2\uffff\1\6"+
            "\1\uffff\2\6\3\3",
            "",
            "\1\3\1\uffff\2\3\132\uffff\1\7\3\uffff\1\3\2\uffff\2\3\1\6"+
            "\2\uffff\5\3\26\uffff\15\3\1\uffff\3\3\1\uffff\2\6",
            "\1\3\3\uffff\1\3\4\uffff\1\3\1\uffff\1\3\1\uffff\3\3\3\uffff"+
            "\1\10\126\uffff\1\3\51\uffff\1\3\2\uffff\1\3\4\uffff\2\6\3\3",
            "",
            "\1\3\3\uffff\1\3\4\uffff\1\3\1\uffff\1\3\1\uffff\3\3\3\uffff"+
            "\1\11\126\uffff\1\3\51\uffff\1\3\2\uffff\1\3\6\uffff\3\3",
            "\1\3\1\uffff\2\3\132\uffff\1\6\3\uffff\1\3\2\uffff\2\3\1\6"+
            "\2\uffff\5\3\26\uffff\15\3\1\uffff\3\3\1\uffff\2\6",
            "\1\3\1\uffff\2\3\132\uffff\1\7\3\uffff\1\3\2\uffff\2\3\1\6"+
            "\2\uffff\5\3\26\uffff\15\3\1\uffff\3\3\1\uffff\2\6"
    };

    static final short[] DFA58_eot = DFA.unpackEncodedString(DFA58_eotS);
    static final short[] DFA58_eof = DFA.unpackEncodedString(DFA58_eofS);
    static final char[] DFA58_min = DFA.unpackEncodedStringToUnsignedChars(DFA58_minS);
    static final char[] DFA58_max = DFA.unpackEncodedStringToUnsignedChars(DFA58_maxS);
    static final short[] DFA58_accept = DFA.unpackEncodedString(DFA58_acceptS);
    static final short[] DFA58_special = DFA.unpackEncodedString(DFA58_specialS);
    static final short[][] DFA58_transition;

    static {
        int numStates = DFA58_transitionS.length;
        DFA58_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA58_transition[i] = DFA.unpackEncodedString(DFA58_transitionS[i]);
        }
    }

    class DFA58 extends DFA {

        public DFA58(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 58;
            this.eot = DFA58_eot;
            this.eof = DFA58_eof;
            this.min = DFA58_min;
            this.max = DFA58_max;
            this.accept = DFA58_accept;
            this.special = DFA58_special;
            this.transition = DFA58_transition;
        }
        @Override
		public String getDescription() {
            return "442:1: featureCall : ( simpleFeatureCall | complexFeatureCall );";
        }
    }
    static final String DFA72_eotS =
        "\22\uffff";
    static final String DFA72_eofS =
        "\2\2\20\uffff";
    static final String DFA72_minS =
        "\1\11\1\4\1\uffff\3\0\1\156\1\21\1\uffff\1\17\2\0\1\156\1\157\1"+
        "\17\1\0\1\157\1\0";
    static final String DFA72_maxS =
        "\1\u00b3\1\u00a4\1\uffff\3\0\1\156\1\27\1\uffff\1\17\2\0\1\156\1"+
        "\157\1\17\1\0\1\157\1\0";
    static final String DFA72_acceptS =
        "\2\uffff\1\2\5\uffff\1\1\11\uffff";
    static final String DFA72_specialS =
        "\3\uffff\1\5\1\6\1\2\4\uffff\1\1\1\3\3\uffff\1\0\1\uffff\1\4}>";
    static final String[] DFA72_transitionS = {
            "\4\2\16\uffff\1\2\111\uffff\1\2\1\uffff\1\2\1\uffff\5\2\1\uffff"+
            "\1\2\1\1\1\uffff\1\2\3\uffff\2\2\1\uffff\1\2\16\uffff\23\2\1"+
            "\uffff\4\2\5\uffff\2\2\2\uffff\1\2\1\uffff\1\2\1\uffff\4\2\1"+
            "\uffff\2\2",
            "\1\2\3\uffff\1\2\4\uffff\1\2\1\uffff\1\2\1\uffff\1\3\1\4\1"+
            "\6\3\uffff\1\5\121\uffff\2\2\3\uffff\1\2\11\uffff\1\2\1\uffff"+
            "\4\2\1\uffff\11\2\20\uffff\1\2\2\uffff\1\2\6\uffff\1\7\2\2",
            "",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\11",
            "\2\13\1\14\3\uffff\1\12",
            "",
            "\1\15",
            "\1\uffff",
            "\1\uffff",
            "\1\16",
            "\1\17",
            "\1\20",
            "\1\uffff",
            "\1\21",
            "\1\uffff"
    };

    static final short[] DFA72_eot = DFA.unpackEncodedString(DFA72_eotS);
    static final short[] DFA72_eof = DFA.unpackEncodedString(DFA72_eofS);
    static final char[] DFA72_min = DFA.unpackEncodedStringToUnsignedChars(DFA72_minS);
    static final char[] DFA72_max = DFA.unpackEncodedStringToUnsignedChars(DFA72_maxS);
    static final short[] DFA72_accept = DFA.unpackEncodedString(DFA72_acceptS);
    static final short[] DFA72_special = DFA.unpackEncodedString(DFA72_specialS);
    static final short[][] DFA72_transition;

    static {
        int numStates = DFA72_transitionS.length;
        DFA72_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA72_transition[i] = DFA.unpackEncodedString(DFA72_transitionS[i]);
        }
    }

    class DFA72 extends DFA {

        public DFA72(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 72;
            this.eot = DFA72_eot;
            this.eof = DFA72_eof;
            this.min = DFA72_min;
            this.max = DFA72_max;
            this.accept = DFA72_accept;
            this.special = DFA72_special;
            this.transition = DFA72_transition;
        }
        @Override
		public String getDescription() {
            return "498:29: ( ':' (n= 'new' )? t= typeName ( parameterList )? )?";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA72_15 = input.LA(1);

                         
                        int index72_15 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred99_EolParserRules()) ) {s = 8;}

                        else if ( (true) ) {s = 2;}

                         
                        input.seek(index72_15);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA72_10 = input.LA(1);

                         
                        int index72_10 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred99_EolParserRules()) ) {s = 8;}

                        else if ( (true) ) {s = 2;}

                         
                        input.seek(index72_10);
                        if ( s>=0 ) return s;
                        break;
                    case 2 : 
                        int LA72_5 = input.LA(1);

                         
                        int index72_5 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred99_EolParserRules()) ) {s = 8;}

                        else if ( (true) ) {s = 2;}

                         
                        input.seek(index72_5);
                        if ( s>=0 ) return s;
                        break;
                    case 3 : 
                        int LA72_11 = input.LA(1);

                         
                        int index72_11 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred99_EolParserRules()) ) {s = 8;}

                        else if ( (true) ) {s = 2;}

                         
                        input.seek(index72_11);
                        if ( s>=0 ) return s;
                        break;
                    case 4 : 
                        int LA72_17 = input.LA(1);

                         
                        int index72_17 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred99_EolParserRules()) ) {s = 8;}

                        else if ( (true) ) {s = 2;}

                         
                        input.seek(index72_17);
                        if ( s>=0 ) return s;
                        break;
                    case 5 : 
                        int LA72_3 = input.LA(1);

                         
                        int index72_3 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred99_EolParserRules()) ) {s = 8;}

                        else if ( (true) ) {s = 2;}

                         
                        input.seek(index72_3);
                        if ( s>=0 ) return s;
                        break;
                    case 6 : 
                        int LA72_4 = input.LA(1);

                         
                        int index72_4 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred99_EolParserRules()) ) {s = 8;}

                        else if ( (true) ) {s = 2;}

                         
                        input.seek(index72_4);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 72, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA75_eotS =
        "\16\uffff";
    static final String DFA75_eofS =
        "\16\uffff";
    static final String DFA75_minS =
        "\1\4\13\0\2\uffff";
    static final String DFA75_maxS =
        "\1\u00a4\13\0\2\uffff";
    static final String DFA75_acceptS =
        "\14\uffff\1\1\1\2";
    static final String DFA75_specialS =
        "\1\uffff\1\0\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\12\2\uffff}>";
    static final String[] DFA75_transitionS = {
            "\1\5\3\uffff\1\5\4\uffff\1\5\1\uffff\1\5\1\uffff\1\3\1\4\1\7"+
            "\3\uffff\1\6\126\uffff\1\10\51\uffff\1\2\2\uffff\1\1\6\uffff"+
            "\1\11\1\12\1\13",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "",
            ""
    };

    static final short[] DFA75_eot = DFA.unpackEncodedString(DFA75_eotS);
    static final short[] DFA75_eof = DFA.unpackEncodedString(DFA75_eofS);
    static final char[] DFA75_min = DFA.unpackEncodedStringToUnsignedChars(DFA75_minS);
    static final char[] DFA75_max = DFA.unpackEncodedStringToUnsignedChars(DFA75_maxS);
    static final short[] DFA75_accept = DFA.unpackEncodedString(DFA75_acceptS);
    static final short[] DFA75_special = DFA.unpackEncodedString(DFA75_specialS);
    static final short[][] DFA75_transition;

    static {
        int numStates = DFA75_transitionS.length;
        DFA75_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA75_transition[i] = DFA.unpackEncodedString(DFA75_transitionS[i]);
        }
    }

    class DFA75 extends DFA {

        public DFA75(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 75;
            this.eot = DFA75_eot;
            this.eof = DFA75_eof;
            this.min = DFA75_min;
            this.max = DFA75_max;
            this.accept = DFA75_accept;
            this.special = DFA75_special;
            this.transition = DFA75_transition;
        }
        @Override
		public String getDescription() {
            return "524:1: expressionListOrRange : ( expressionRange | expressionList );";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA75_1 = input.LA(1);

                         
                        int index75_1 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_1);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA75_2 = input.LA(1);

                         
                        int index75_2 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_2);
                        if ( s>=0 ) return s;
                        break;
                    case 2 : 
                        int LA75_3 = input.LA(1);

                         
                        int index75_3 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_3);
                        if ( s>=0 ) return s;
                        break;
                    case 3 : 
                        int LA75_4 = input.LA(1);

                         
                        int index75_4 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_4);
                        if ( s>=0 ) return s;
                        break;
                    case 4 : 
                        int LA75_5 = input.LA(1);

                         
                        int index75_5 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_5);
                        if ( s>=0 ) return s;
                        break;
                    case 5 : 
                        int LA75_6 = input.LA(1);

                         
                        int index75_6 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_6);
                        if ( s>=0 ) return s;
                        break;
                    case 6 : 
                        int LA75_7 = input.LA(1);

                         
                        int index75_7 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_7);
                        if ( s>=0 ) return s;
                        break;
                    case 7 : 
                        int LA75_8 = input.LA(1);

                         
                        int index75_8 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_8);
                        if ( s>=0 ) return s;
                        break;
                    case 8 : 
                        int LA75_9 = input.LA(1);

                         
                        int index75_9 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_9);
                        if ( s>=0 ) return s;
                        break;
                    case 9 : 
                        int LA75_10 = input.LA(1);

                         
                        int index75_10 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_10);
                        if ( s>=0 ) return s;
                        break;
                    case 10 : 
                        int LA75_11 = input.LA(1);

                         
                        int index75_11 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred102_EolParserRules()) ) {s = 12;}

                        else if ( (true) ) {s = 13;}

                         
                        input.seek(index75_11);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 75, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA78_eotS =
        "\17\uffff";
    static final String DFA78_eofS =
        "\17\uffff";
    static final String DFA78_minS =
        "\1\4\2\0\1\uffff\1\0\12\uffff";
    static final String DFA78_maxS =
        "\1\u00a4\2\0\1\uffff\1\0\12\uffff";
    static final String DFA78_acceptS =
        "\3\uffff\1\3\1\uffff\1\7\1\10\1\11\1\12\1\uffff\1\1\1\5\1\2\1\4"+
        "\1\6";
    static final String DFA78_specialS =
        "\1\uffff\1\0\1\1\1\uffff\1\2\12\uffff}>";
    static final String[] DFA78_transitionS = {
            "\1\3\3\uffff\1\3\4\uffff\1\3\1\uffff\1\3\1\uffff\1\1\1\2\1\5"+
            "\3\uffff\1\4\126\uffff\1\6\63\uffff\1\7\2\10",
            "\1\uffff",
            "\1\uffff",
            "",
            "\1\uffff",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
    };

    static final short[] DFA78_eot = DFA.unpackEncodedString(DFA78_eotS);
    static final short[] DFA78_eof = DFA.unpackEncodedString(DFA78_eofS);
    static final char[] DFA78_min = DFA.unpackEncodedStringToUnsignedChars(DFA78_minS);
    static final char[] DFA78_max = DFA.unpackEncodedStringToUnsignedChars(DFA78_maxS);
    static final short[] DFA78_accept = DFA.unpackEncodedString(DFA78_acceptS);
    static final short[] DFA78_special = DFA.unpackEncodedString(DFA78_specialS);
    static final short[][] DFA78_transition;

    static {
        int numStates = DFA78_transitionS.length;
        DFA78_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA78_transition[i] = DFA.unpackEncodedString(DFA78_transitionS[i]);
        }
    }

    class DFA78 extends DFA {

        public DFA78(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 78;
            this.eot = DFA78_eot;
            this.eof = DFA78_eof;
            this.min = DFA78_min;
            this.max = DFA78_max;
            this.accept = DFA78_accept;
            this.special = DFA78_special;
            this.transition = DFA78_transition;
        }
        @Override
		public String getDescription() {
            return "551:1: primitiveExpression : ( literalSequentialCollection | literalMapCollection | literal | featureCall | collectionType | pathName | specialType | logicalExpressionInBrackets | newExpression | variableDeclarationExpression );";
        }
        @Override
		public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            TokenStream input = (TokenStream)_input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA78_1 = input.LA(1);

                         
                        int index78_1 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred105_EolParserRules()) ) {s = 10;}

                        else if ( (synpred109_EolParserRules()) ) {s = 11;}

                         
                        input.seek(index78_1);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA78_2 = input.LA(1);

                         
                        int index78_2 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred106_EolParserRules()) ) {s = 12;}

                        else if ( (synpred109_EolParserRules()) ) {s = 11;}

                         
                        input.seek(index78_2);
                        if ( s>=0 ) return s;
                        break;
                    case 2 : 
                        int LA78_4 = input.LA(1);

                         
                        int index78_4 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred108_EolParserRules()) ) {s = 13;}

                        else if ( (synpred110_EolParserRules()) ) {s = 14;}

                         
                        input.seek(index78_4);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 78, _s, input);
            error(nvae);
            throw nvae;
        }
    }
 

    public static final BitSet FOLLOW_operationDeclaration_in_operationDeclarationOrAnnotationBlock263 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_annotationBlock_in_operationDeclarationOrAnnotationBlock267 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_100_in_modelDeclaration286 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_modelDeclaration289 = new BitSet(new long[]{0x0000000000000000L,0x0000036000000000L});
    public static final BitSet FOLLOW_modelAlias_in_modelDeclaration291 = new BitSet(new long[]{0x0000000000000000L,0x0000032000000000L});
    public static final BitSet FOLLOW_modelDriver_in_modelDeclaration294 = new BitSet(new long[]{0x0000000000000000L,0x0000022000000000L});
    public static final BitSet FOLLOW_modelDeclarationParameters_in_modelDeclaration297 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_modelDeclaration302 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_102_in_modelAlias317 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_modelAlias320 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_103_in_modelAlias323 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_modelAlias326 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_104_in_modelDriver345 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_modelDriver348 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_105_in_modelDeclarationParameters370 = new BitSet(new long[]{0x0000000000800000L,0x0000048000000000L});
    public static final BitSet FOLLOW_modelDeclarationParameter_in_modelDeclarationParameters373 = new BitSet(new long[]{0x0000000000000000L,0x0000048000000000L});
    public static final BitSet FOLLOW_103_in_modelDeclarationParameters377 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_modelDeclarationParameter_in_modelDeclarationParameters380 = new BitSet(new long[]{0x0000000000000000L,0x0000048000000000L});
    public static final BitSet FOLLOW_106_in_modelDeclarationParameters386 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_NAME_in_modelDeclarationParameter399 = new BitSet(new long[]{0x0000000000000000L,0x0000080000000000L});
    public static final BitSet FOLLOW_107_in_modelDeclarationParameter403 = new BitSet(new long[]{0x0000000000008000L});
    public static final BitSet FOLLOW_STRING_in_modelDeclarationParameter406 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_set_in_operationDeclaration427 = new BitSet(new long[]{0x00000000008E0000L});
    public static final BitSet FOLLOW_typeName_in_operationDeclaration437 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_operationDeclaration447 = new BitSet(new long[]{0x0000000000000000L,0x0000400000000000L});
    public static final BitSet FOLLOW_110_in_operationDeclaration451 = new BitSet(new long[]{0x0000000000800000L,0x0000800000000000L});
    public static final BitSet FOLLOW_formalParameterList_in_operationDeclaration454 = new BitSet(new long[]{0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_111_in_operationDeclaration459 = new BitSet(new long[]{0x0000000000000000L,0x0001020000000000L});
    public static final BitSet FOLLOW_112_in_operationDeclaration465 = new BitSet(new long[]{0x00000000008E0000L});
    public static final BitSet FOLLOW_typeName_in_operationDeclaration470 = new BitSet(new long[]{0x0000000000000000L,0x0001020000000000L});
    public static final BitSet FOLLOW_statementBlock_in_operationDeclaration476 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_113_in_importStatement496 = new BitSet(new long[]{0x0000000000008000L});
    public static final BitSet FOLLOW_STRING_in_importStatement499 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_importStatement503 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statement_in_block524 = new BitSet(new long[]{0x00000000008EA112L,0xA500400000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_105_in_statementBlock554 = new BitSet(new long[]{0x00000000008EA110L,0xA500400000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_block_in_statementBlock557 = new BitSet(new long[]{0x0000000000000000L,0x0000040000000000L});
    public static final BitSet FOLLOW_106_in_statementBlock561 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_NAME_in_formalParameter579 = new BitSet(new long[]{0x0000000000000002L,0x0001000000000000L});
    public static final BitSet FOLLOW_112_in_formalParameter582 = new BitSet(new long[]{0x00000000008E0000L});
    public static final BitSet FOLLOW_typeName_in_formalParameter586 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_formalParameter_in_formalParameterList620 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_103_in_formalParameterList623 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_formalParameter_in_formalParameterList625 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_114_in_executableAnnotation650 = new BitSet(new long[]{0xFFFFFFFFFFFFFFF0L,0xFFFFFFFFFFFFFFFFL,0x000FFFFFFFFFFFFFL});
    public static final BitSet FOLLOW_logicalExpression_in_executableAnnotation657 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_Annotation_in_annotation671 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_executableAnnotation_in_annotation675 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_annotation_in_annotationBlock692 = new BitSet(new long[]{0x0000000008000002L,0x0004000000000000L});
    public static final BitSet FOLLOW_pathName_in_typeName721 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_collectionType_in_typeName725 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_specialType_in_typeName729 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_SpecialTypeName_in_specialType746 = new BitSet(new long[]{0x0000000000000000L,0x0000400000000000L});
    public static final BitSet FOLLOW_110_in_specialType751 = new BitSet(new long[]{0x0000000000008000L});
    public static final BitSet FOLLOW_STRING_in_specialType754 = new BitSet(new long[]{0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_111_in_specialType758 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_NAME_in_pathName773 = new BitSet(new long[]{0x0000000000000000L,0x0008000000000000L});
    public static final BitSet FOLLOW_115_in_pathName775 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_packagedType_in_pathName784 = new BitSet(new long[]{0x0000000000000002L,0x0010000000000000L});
    public static final BitSet FOLLOW_116_in_pathName790 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_pathName795 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_NAME_in_packagedType816 = new BitSet(new long[]{0x0000000000000002L,0x0020000000000000L});
    public static final BitSet FOLLOW_117_in_packagedType819 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_packagedType824 = new BitSet(new long[]{0x0000000000000002L,0x0020000000000000L});
    public static final BitSet FOLLOW_set_in_collectionType854 = new BitSet(new long[]{0x0000000000000002L,0x0040400000000000L});
    public static final BitSet FOLLOW_110_in_collectionType869 = new BitSet(new long[]{0x00000000008E0000L});
    public static final BitSet FOLLOW_typeName_in_collectionType874 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_103_in_collectionType879 = new BitSet(new long[]{0x00000000008E0000L});
    public static final BitSet FOLLOW_typeName_in_collectionType883 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_111_in_collectionType891 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_118_in_collectionType903 = new BitSet(new long[]{0x00000000008E0000L});
    public static final BitSet FOLLOW_typeName_in_collectionType908 = new BitSet(new long[]{0x0000000000000000L,0x0080008000000000L});
    public static final BitSet FOLLOW_103_in_collectionType913 = new BitSet(new long[]{0x00000000008E0000L});
    public static final BitSet FOLLOW_typeName_in_collectionType917 = new BitSet(new long[]{0x0000000000000000L,0x0080008000000000L});
    public static final BitSet FOLLOW_119_in_collectionType925 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statementA_in_statement944 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statementB_in_statement948 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_assignmentStatement_in_statementA959 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_expressionStatement_in_statementA963 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_forStatement_in_statementA967 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_ifStatement_in_statementA973 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_whileStatement_in_statementA977 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_switchStatement_in_statementA981 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_returnStatement_in_statementA985 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_breakStatement_in_statementA989 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_breakAllStatement_in_statementB1001 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_returnStatement_in_statementB1005 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_transactionStatement_in_statementB1009 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_abortStatement_in_statementB1015 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_continueStatement_in_statementB1019 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_throwStatement_in_statementB1023 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_deleteStatement_in_statementB1029 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statement_in_statementOrStatementBlock1040 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statementBlock_in_statementOrStatementBlock1044 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_112_in_expressionOrStatementBlock1053 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_expressionOrStatementBlock1056 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statementBlock_in_expressionOrStatementBlock1060 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_120_in_ifStatement1073 = new BitSet(new long[]{0x0000000000000000L,0x0000400000000000L});
    public static final BitSet FOLLOW_110_in_ifStatement1076 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_ifStatement1079 = new BitSet(new long[]{0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_111_in_ifStatement1081 = new BitSet(new long[]{0x00000000008EA110L,0xA501420000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_statementOrStatementBlock_in_ifStatement1084 = new BitSet(new long[]{0x0000000000000002L,0x0200000000000000L});
    public static final BitSet FOLLOW_elseStatement_in_ifStatement1086 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_121_in_elseStatement1109 = new BitSet(new long[]{0x00000000008EA110L,0xA501420000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_statementOrStatementBlock_in_elseStatement1112 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_122_in_switchStatement1126 = new BitSet(new long[]{0x0000000000000000L,0x0000400000000000L});
    public static final BitSet FOLLOW_110_in_switchStatement1129 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_switchStatement1132 = new BitSet(new long[]{0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_111_in_switchStatement1134 = new BitSet(new long[]{0x0000000000000000L,0x0000020000000000L});
    public static final BitSet FOLLOW_105_in_switchStatement1137 = new BitSet(new long[]{0x0000000000000000L,0x1800040000000000L});
    public static final BitSet FOLLOW_caseStatement_in_switchStatement1140 = new BitSet(new long[]{0x0000000000000000L,0x1800040000000000L});
    public static final BitSet FOLLOW_defaultStatement_in_switchStatement1143 = new BitSet(new long[]{0x0000000000000000L,0x0000040000000000L});
    public static final BitSet FOLLOW_106_in_switchStatement1146 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_123_in_caseStatement1165 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_caseStatement1168 = new BitSet(new long[]{0x0000000000000000L,0x0001000000000000L});
    public static final BitSet FOLLOW_112_in_caseStatement1170 = new BitSet(new long[]{0x00000000008EA110L,0xA501420000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_block_in_caseStatement1174 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statementBlock_in_caseStatement1178 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_124_in_defaultStatement1197 = new BitSet(new long[]{0x0000000000000000L,0x0001000000000000L});
    public static final BitSet FOLLOW_112_in_defaultStatement1200 = new BitSet(new long[]{0x00000000008EA110L,0xA501420000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_block_in_defaultStatement1204 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statementBlock_in_defaultStatement1208 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_125_in_forStatement1226 = new BitSet(new long[]{0x0000000000000000L,0x0000400000000000L});
    public static final BitSet FOLLOW_110_in_forStatement1229 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_formalParameter_in_forStatement1232 = new BitSet(new long[]{0x0000000000000000L,0x4000000000000000L});
    public static final BitSet FOLLOW_126_in_forStatement1234 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_forStatement1237 = new BitSet(new long[]{0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_111_in_forStatement1239 = new BitSet(new long[]{0x00000000008EA110L,0xA501420000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_statementOrStatementBlock_in_forStatement1242 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_127_in_whileStatement1258 = new BitSet(new long[]{0x0000000000000000L,0x0000400000000000L});
    public static final BitSet FOLLOW_110_in_whileStatement1261 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_whileStatement1264 = new BitSet(new long[]{0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_111_in_whileStatement1266 = new BitSet(new long[]{0x00000000008EA110L,0xA501420000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_statementOrStatementBlock_in_whileStatement1269 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_128_in_returnStatement1291 = new BitSet(new long[]{0x00000000008EA110L,0x0000402000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_returnStatement1294 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_returnStatement1299 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_129_in_throwStatement1322 = new BitSet(new long[]{0x00000000008EA110L,0x0000402000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_throwStatement1325 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_throwStatement1330 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_130_in_deleteStatement1353 = new BitSet(new long[]{0x00000000008EA110L,0x0000402000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_deleteStatement1356 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_deleteStatement1361 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_131_in_breakStatement1387 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_breakStatement1392 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_132_in_breakAllStatement1415 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_breakAllStatement1420 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_133_in_continueStatement1443 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_continueStatement1448 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_134_in_abortStatement1471 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_abortStatement1476 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_135_in_transactionStatement1493 = new BitSet(new long[]{0x00000000008EA110L,0xA501420000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_NAME_in_transactionStatement1497 = new BitSet(new long[]{0x00000000008EA110L,0xA501428000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_103_in_transactionStatement1500 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_transactionStatement1502 = new BitSet(new long[]{0x00000000008EA110L,0xA501428000000000L,0x0000001C090000FFL});
    public static final BitSet FOLLOW_statementOrStatementBlock_in_transactionStatement1508 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_logicalExpression_in_assignmentStatement1528 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0000000000003F00L});
    public static final BitSet FOLLOW_136_in_assignmentStatement1534 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_137_in_assignmentStatement1539 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_138_in_assignmentStatement1544 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_139_in_assignmentStatement1549 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_140_in_assignmentStatement1554 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_141_in_assignmentStatement1566 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_assignmentStatement1574 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_assignmentStatement1578 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_postfixExpression_in_expressionStatement1598 = new BitSet(new long[]{0x0000000000000000L,0x0000080000000000L});
    public static final BitSet FOLLOW_107_in_expressionStatement1602 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_expressionStatement1605 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_logicalExpression_in_expressionStatement1612 = new BitSet(new long[]{0x0000000000000000L,0x0000002000000000L});
    public static final BitSet FOLLOW_101_in_expressionStatement1617 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_relationalExpression_in_logicalExpression1629 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x000000000007C000L});
    public static final BitSet FOLLOW_142_in_logicalExpression1640 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_143_in_logicalExpression1645 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_144_in_logicalExpression1650 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_145_in_logicalExpression1655 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_146_in_logicalExpression1669 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_relationalExpression_in_logicalExpression1672 = new BitSet(new long[]{0x0000000000000000L,0x0201000000000000L});
    public static final BitSet FOLLOW_set_in_logicalExpression1674 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_relationalExpression_in_logicalExpression1690 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x000000000007C000L});
    public static final BitSet FOLLOW_additiveExpression_in_relationalExpression1706 = new BitSet(new long[]{0x0000000000000002L,0x00C0080000000000L,0x0000000000780000L});
    public static final BitSet FOLLOW_147_in_relationalExpression1712 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_relationalExpression_in_relationalExpression1715 = new BitSet(new long[]{0x0000000000000002L,0x00C0080000000000L,0x0000000000780000L});
    public static final BitSet FOLLOW_107_in_relationalExpression1721 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_relationalExpression_in_relationalExpression1724 = new BitSet(new long[]{0x0000000000000002L,0x00C0080000000000L,0x0000000000780000L});
    public static final BitSet FOLLOW_119_in_relationalExpression1754 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_118_in_relationalExpression1759 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_148_in_relationalExpression1764 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_149_in_relationalExpression1769 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_150_in_relationalExpression1774 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_additiveExpression_in_relationalExpression1778 = new BitSet(new long[]{0x0000000000000002L,0x00C0080000000000L,0x0000000000780000L});
    public static final BitSet FOLLOW_multiplicativeExpression_in_additiveExpression1796 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x0000000001800000L});
    public static final BitSet FOLLOW_151_in_additiveExpression1802 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_152_in_additiveExpression1807 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_multiplicativeExpression_in_additiveExpression1811 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x0000000001800000L});
    public static final BitSet FOLLOW_unaryExpression_in_multiplicativeExpression1829 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x0000000006000000L});
    public static final BitSet FOLLOW_153_in_multiplicativeExpression1835 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_154_in_multiplicativeExpression1840 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_unaryExpression_in_multiplicativeExpression1844 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x0000000006000000L});
    public static final BitSet FOLLOW_155_in_unaryExpression1865 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_152_in_unaryExpression1870 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_shortcutOperatorExpression_in_unaryExpression1878 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_postfixExpression_in_shortcutOperatorExpression1890 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x0000000030000000L});
    public static final BitSet FOLLOW_156_in_shortcutOperatorExpression1896 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_157_in_shortcutOperatorExpression1903 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_itemSelectorExpression_in_postfixExpression1921 = new BitSet(new long[]{0x0000000000001A02L});
    public static final BitSet FOLLOW_set_in_postfixExpression1924 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_featureCall_in_postfixExpression1935 = new BitSet(new long[]{0x0000000000001A02L,0x0000000000000000L,0x0000000040000000L});
    public static final BitSet FOLLOW_158_in_postfixExpression1944 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_postfixExpression1947 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0000000080000000L});
    public static final BitSet FOLLOW_159_in_postfixExpression1949 = new BitSet(new long[]{0x0000000000001A02L,0x0000000000000000L,0x0000000040000000L});
    public static final BitSet FOLLOW_primitiveExpression_in_itemSelectorExpression1971 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x0000000040000000L});
    public static final BitSet FOLLOW_158_in_itemSelectorExpression1976 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_primitiveExpression_in_itemSelectorExpression1979 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0000000080000000L});
    public static final BitSet FOLLOW_159_in_itemSelectorExpression1981 = new BitSet(new long[]{0x0000000000000002L,0x0000000000000000L,0x0000000040000000L});
    public static final BitSet FOLLOW_simpleFeatureCall_in_featureCall1999 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_complexFeatureCall_in_featureCall2003 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_NAME_in_simpleFeatureCall2017 = new BitSet(new long[]{0x0000000000000002L,0x0000400000000000L});
    public static final BitSet FOLLOW_parameterList_in_simpleFeatureCall2020 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_110_in_parameterList2043 = new BitSet(new long[]{0x00000000008EA110L,0x0000C00000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_parameterList2046 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_103_in_parameterList2049 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_parameterList2051 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_111_in_parameterList2059 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_NAME_in_complexFeatureCall2087 = new BitSet(new long[]{0x0000000000000000L,0x0000400000000000L});
    public static final BitSet FOLLOW_110_in_complexFeatureCall2092 = new BitSet(new long[]{0x0000000000800000L,0x0000400000000000L,0x0000000340000000L});
    public static final BitSet FOLLOW_lambdaExpression_in_complexFeatureCall2096 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_lambdaExpressionInBrackets_in_complexFeatureCall2100 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_103_in_complexFeatureCall2106 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001F49000000L});
    public static final BitSet FOLLOW_logicalExpression_in_complexFeatureCall2110 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_lambdaExpressionInBrackets_in_complexFeatureCall2114 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_111_in_complexFeatureCall2121 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_110_in_lambdaExpressionInBrackets2142 = new BitSet(new long[]{0x0000000000800000L,0x0000000000000000L,0x0000000300000000L});
    public static final BitSet FOLLOW_lambdaExpression_in_lambdaExpressionInBrackets2145 = new BitSet(new long[]{0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_111_in_lambdaExpressionInBrackets2149 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_158_in_lambdaExpressionInBrackets2160 = new BitSet(new long[]{0x0000000000800000L,0x0000000000000000L,0x0000000300000000L});
    public static final BitSet FOLLOW_lambdaExpression_in_lambdaExpressionInBrackets2163 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0000000080000000L});
    public static final BitSet FOLLOW_159_in_lambdaExpressionInBrackets2167 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_formalParameterList_in_lambdaExpression2186 = new BitSet(new long[]{0x0000000000000000L,0x0000000000000000L,0x0000000300000000L});
    public static final BitSet FOLLOW_160_in_lambdaExpression2192 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_161_in_lambdaExpression2199 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_lambdaExpression2203 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_162_in_newExpression2216 = new BitSet(new long[]{0x00000000008E0000L});
    public static final BitSet FOLLOW_typeName_in_newExpression2221 = new BitSet(new long[]{0x0000000000000002L,0x0000400000000000L});
    public static final BitSet FOLLOW_parameterList_in_newExpression2225 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_163_in_variableDeclarationExpression2249 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_164_in_variableDeclarationExpression2254 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_variableDeclarationExpression2258 = new BitSet(new long[]{0x0000000000000002L,0x0001000000000000L});
    public static final BitSet FOLLOW_112_in_variableDeclarationExpression2261 = new BitSet(new long[]{0x00000000008E0000L,0x0000000000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_162_in_variableDeclarationExpression2266 = new BitSet(new long[]{0x00000000008E0000L,0x0000000000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_typeName_in_variableDeclarationExpression2272 = new BitSet(new long[]{0x0000000000000002L,0x0000400000000000L});
    public static final BitSet FOLLOW_parameterList_in_variableDeclarationExpression2276 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_CollectionTypeName_in_literalSequentialCollection2299 = new BitSet(new long[]{0x0000000000000000L,0x0000020000000000L});
    public static final BitSet FOLLOW_105_in_literalSequentialCollection2306 = new BitSet(new long[]{0x00000000008EA110L,0x0000440000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_expressionListOrRange_in_literalSequentialCollection2309 = new BitSet(new long[]{0x0000000000000000L,0x0000040000000000L});
    public static final BitSet FOLLOW_106_in_literalSequentialCollection2314 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_logicalExpression_in_expressionRange2329 = new BitSet(new long[]{0x0000000000000400L});
    public static final BitSet FOLLOW_POINT_POINT_in_expressionRange2333 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_expressionRange2336 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_logicalExpression_in_expressionList2357 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_103_in_expressionList2360 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_expressionList2362 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_expressionRange_in_expressionListOrRange2386 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_expressionList_in_expressionListOrRange2390 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_MapTypeName_in_literalMapCollection2409 = new BitSet(new long[]{0x0000000000000000L,0x0000020000000000L});
    public static final BitSet FOLLOW_105_in_literalMapCollection2414 = new BitSet(new long[]{0x00000000008EA110L,0x0000440000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_keyvalExpressionList_in_literalMapCollection2417 = new BitSet(new long[]{0x0000000000000000L,0x0000040000000000L});
    public static final BitSet FOLLOW_106_in_literalMapCollection2422 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_keyvalExpression_in_keyvalExpressionList2443 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_103_in_keyvalExpressionList2446 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_keyvalExpression_in_keyvalExpressionList2448 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_additiveExpression_in_keyvalExpression2473 = new BitSet(new long[]{0x0000000000000000L,0x0000080000000000L});
    public static final BitSet FOLLOW_107_in_keyvalExpression2477 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_keyvalExpression2480 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_literalSequentialCollection_in_primitiveExpression2495 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_literalMapCollection_in_primitiveExpression2499 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_literal_in_primitiveExpression2503 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_featureCall_in_primitiveExpression2507 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_collectionType_in_primitiveExpression2511 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_pathName_in_primitiveExpression2517 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_specialType_in_primitiveExpression2521 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_logicalExpressionInBrackets_in_primitiveExpression2525 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_newExpression_in_primitiveExpression2529 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_variableDeclarationExpression_in_primitiveExpression2533 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_110_in_logicalExpressionInBrackets2552 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_logicalExpressionInBrackets2555 = new BitSet(new long[]{0x0000000000000000L,0x0000800000000000L});
    public static final BitSet FOLLOW_111_in_logicalExpressionInBrackets2559 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_set_in_literal0 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_annotation_in_synpred16_EolParserRules692 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_110_in_synpred24_EolParserRules869 = new BitSet(new long[]{0x00000000008E0000L,0x0000000000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_typeName_in_synpred24_EolParserRules874 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_103_in_synpred24_EolParserRules879 = new BitSet(new long[]{0x00000000008E0000L,0x0000000000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_typeName_in_synpred24_EolParserRules883 = new BitSet(new long[]{0x0000000000000000L,0x0000808000000000L});
    public static final BitSet FOLLOW_111_in_synpred24_EolParserRules891 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_118_in_synpred26_EolParserRules903 = new BitSet(new long[]{0x00000000008E0000L,0x0000000000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_typeName_in_synpred26_EolParserRules908 = new BitSet(new long[]{0x0000000000000000L,0x0080008000000000L});
    public static final BitSet FOLLOW_103_in_synpred26_EolParserRules913 = new BitSet(new long[]{0x00000000008E0000L,0x0000000000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_typeName_in_synpred26_EolParserRules917 = new BitSet(new long[]{0x0000000000000000L,0x0080008000000000L});
    public static final BitSet FOLLOW_119_in_synpred26_EolParserRules925 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_statementA_in_synpred27_EolParserRules944 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_assignmentStatement_in_synpred28_EolParserRules959 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_expressionStatement_in_synpred29_EolParserRules963 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_elseStatement_in_synpred43_EolParserRules1086 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_NAME_in_synpred52_EolParserRules1497 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_103_in_synpred52_EolParserRules1500 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_NAME_in_synpred52_EolParserRules1502 = new BitSet(new long[]{0x0000000000000002L,0x0000008000000000L});
    public static final BitSet FOLLOW_postfixExpression_in_synpred58_EolParserRules1598 = new BitSet(new long[]{0x0000000000000000L,0x0000080000000000L});
    public static final BitSet FOLLOW_107_in_synpred58_EolParserRules1602 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_logicalExpression_in_synpred58_EolParserRules1605 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_147_in_synpred71_EolParserRules1712 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_relationalExpression_in_synpred71_EolParserRules1715 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_107_in_synpred71_EolParserRules1721 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_relationalExpression_in_synpred71_EolParserRules1724 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_set_in_synpred71_EolParserRules1751 = new BitSet(new long[]{0x00000000008EA110L,0x0000400000000000L,0x0000001C09000000L});
    public static final BitSet FOLLOW_additiveExpression_in_synpred71_EolParserRules1778 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_112_in_synpred99_EolParserRules2261 = new BitSet(new long[]{0x00000000008E0000L,0x0000000000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_162_in_synpred99_EolParserRules2266 = new BitSet(new long[]{0x00000000008E0000L,0x0000000000000000L,0x0000000400000000L});
    public static final BitSet FOLLOW_typeName_in_synpred99_EolParserRules2272 = new BitSet(new long[]{0x0000000000000002L,0x0000400000000000L});
    public static final BitSet FOLLOW_parameterList_in_synpred99_EolParserRules2276 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_expressionRange_in_synpred102_EolParserRules2386 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_literalSequentialCollection_in_synpred105_EolParserRules2495 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_literalMapCollection_in_synpred106_EolParserRules2499 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_featureCall_in_synpred108_EolParserRules2507 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_collectionType_in_synpred109_EolParserRules2511 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_pathName_in_synpred110_EolParserRules2517 = new BitSet(new long[]{0x0000000000000002L});

}
