


/*********************************************************************
 * (C) Copyright 2001 Albert Ludwigs University Freiburg
 *     Institute of Computer Science
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 *********************************************************************/


/*
 * THIS SOURCE CODE IS SUPPLIED  ``AS IS'' WITHOUT WARRANTY OF ANY KIND, 
 * AND ITS AUTHOR AND THE JOURNAL OF ARTIFICIAL INTELLIGENCE RESEARCH 
 * (JAIR) AND JAIR'S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL 
 * WARRANTIES, INCLUDING BUT NOT LIMITED TO ANY IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, AND
 * ANY WARRANTIES OR NON INFRINGEMENT.  THE USER ASSUMES ALL LIABILITY AND
 * RESPONSIBILITY FOR USE OF THIS SOURCE CODE, AND NEITHER THE AUTHOR NOR
 * JAIR, NOR JAIR'S PUBLISHERS AND DISTRIBUTORS, WILL BE LIABLE FOR 
 * DAMAGES OF ANY KIND RESULTING FROM ITS USE.  Without limiting the 
 * generality of the foregoing, neither the author, nor JAIR, nor JAIR's
 * publishers and distributors, warrant that the Source Code will be 
 * error-free, will operate without interruption, or will meet the needs 
 * of the user.
 */










/*********************************************************************
 * File: inst_pre.h
 * Description: headers for instantiating operators, preprocessing part.
 *              - transform domain into integers
 *              - inertia preprocessing:
 *                  - collect inertia info
 *                  - split initial state in special arrays
 *              - Wff normalization:
 *                  - simplification
 *                  - quantifier expansion
 *                  - NOT s down
 *              - negative preconditions translation
 *              - split operators into easy and hard to instantiate ones
 *
 *              - full DNF functions, only feasible for fully instantiated 
 *                formulae
 *
 * Author: Joerg Hoffmann 2000
 *
 *********************************************************************/ 











#ifndef _INST_PRE_H
#define _INST_PRE_H



void encode_domain_in_integers( void );
void collect_all_strings( void );
void create_member_nrs( void );
int position_in_types_table( char *str );
int position_in_constants_table( char *str );
int position_in_predicates_table( char *str );
void create_integer_representation( void );
void make_Fact( Fact *f, PlNode *n, int num_vars );
Bool is_subtype( int t1, int t2 );
WffNode *make_Wff( PlNode *p, int num_vars );
Effect *make_effect( PlNode *p, int num_vars );



void do_inertia_preprocessing_step_1( void );
void collect_inertia_information( void );
void split_initial_state( void );



int normalize_all_wffs( void );
void remove_unused_vars_in_wff( WffNode **w );
void decrement_inferior_vars( int var, WffNode *w );
Bool var_used_in_wff( int code_var, WffNode *w );
void simplify_wff( WffNode **w );
void expand_quantifiers_in_wff( WffNode **w, int var, int constant );
WffNode *copy_Wff( WffNode *w );
Bool possibly_positive( Fact *f );
Bool possibly_negative( Fact *f );
Bool matches( Fact *f1, Fact *f2 );
void cleanup_wff( WffNode **w );
void detect_tautologies_in_wff( WffNode **w );
Bool are_identical_ATOMs( WffNode *w1, WffNode *w2 );
void merge_ANDs_and_ORs_in_wff( WffNode **w );
void NOTs_down_in_wff( WffNode **w );



void translate_negative_preconds( void );
Bool translate_one_negative_cond( WffNode *w );
void replace_not_p_with_n_in_wff( int p, int n, WffNode **w );
void add_to_initial_state( int p, int n, int index );



void split_domain( void );
int is_dnf( WffNode *w );
void  make_normal_effects( NormOperator **nop, Operator *op );



void dnf( WffNode **w );
void ANDs_below_ORs_in_wff( WffNode **w );
void collect_hitting_sets( WffNode *ORlist, int index );
void merge_next_step_ANDs_and_ORs_in_wff( WffNode **w );



#endif /* _INST_PRE_H */
