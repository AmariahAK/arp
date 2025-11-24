# Task: Probabilistic Programming Language Compiler in OCaml

## Overview

Build a production-grade compiler for a probabilistic programming language that supports automatic inference. The language allows expressing probabilistic models declaratively, and the compiler performs type checking, optimization, and code generation. The inference engine supports exact inference, MCMC, and variational inference.

**Difficulty:** EXTREME  
**Estimated time:** 55-70 hours  
**Turns:** 16

---

## TURN 1 — Language Design and AST Definition

**Instructions:**

Design the core language syntax and define the Abstract Syntax Tree (AST) using OCaml's algebraic data types.

**Background:** A probabilistic programming language needs constructs for sampling from distributions, observing data, and performing inference. The AST must capture all language constructs.

**Requirements:**
- Define AST for expressions, types, and declarations
- Support basic types (int, float, bool) and probabilistic types
- Include distribution primitives (normal, bernoulli, etc.)
- Support `sample`, `observe`, `return`, `bind` operations

**Implement:**

```ocaml
(* ast.ml - Abstract Syntax Tree *)

(* Source location for error reporting *)
type loc = {
  file : string;
  line : int;
  col : int;
}

(* Types *)
type ty =
  | TInt
  | TFloat
  | TBool
  | TUnit
  | TFun of ty * ty
  | TTuple of ty list
  | TList of ty
  | TDist of ty        (* Distribution over ty *)
  | TProb of ty        (* Probabilistic computation returning ty *)
  | TVar of string     (* Type variable for polymorphism *)

(* Binary operators *)
type binop =
  | Add | Sub | Mul | Div | Mod
  | Eq | Neq | Lt | Gt | Le | Ge
  | And | Or

(* Unary operators *)
type unop =
  | Neg | Not

(* Expressions *)
type expr =
  | EInt of int * loc
  | EFloat of float * loc
  | EBool of bool * loc
  | EUnit of loc
  | EVar of string * loc
  | EBinop of binop * expr * expr * loc
  | EUnop of unop * expr * loc
  | EIf of expr * expr * expr * loc
  | ELet of string * expr * expr * loc
  | EFun of string * ty option * expr * loc
  | EApp of expr * expr * loc
  | ETuple of expr list * loc
  | EList of expr list * loc
  | ESample of expr * loc        (* sample from distribution *)
  | EObserve of expr * expr * loc  (* observe value from distribution *)
  | EReturn of expr * loc        (* return value in prob monad *)
  | EBind of expr * string * expr * loc  (* bind in prob monad *)
  | EDist of dist_expr * loc     (* distribution expression *)

(* Distribution expressions *)
and dist_expr =
  | DNormal of expr * expr       (* mean, std *)
  | DBernoulli of expr           (* probability *)
  | DCategorical of expr         (* probabilities *)
  | DBeta of expr * expr         (* alpha, beta *)
  | DGamma of expr * expr        (* shape, scale *)
  | DDirichlet of expr           (* concentration *)
  | DUniform of expr * expr      (* low, high *)

(* Top-level declarations *)
type decl =
  | DLet of string * ty option * expr * loc
  | DModel of string * (string * ty) list * expr * loc  (* model definition *)

(* Program *)
type program = decl list
```

**Pretty printer:**

```ocaml
(* pretty.ml - Pretty printing *)

let rec string_of_ty = function
  | TInt -> "int"
  | TFloat -> "float"
  | TBool -> "bool"
  | TUnit -> "unit"
  | TFun (t1, t2) -> Printf.sprintf "(%s -> %s)" (string_of_ty t1) (string_of_ty t2)
  | TTuple ts -> Printf.sprintf "(%s)" (String.concat " * " (List.map string_of_ty ts))
  | TList t -> Printf.sprintf "%s list" (string_of_ty t)
  | TDist t -> Printf.sprintf "%s dist" (string_of_ty t)
  | TProb t -> Printf.sprintf "%s prob" (string_of_ty t)
  | TVar name -> "'" ^ name

let string_of_binop = function
  | Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/" | Mod -> "%"
  | Eq -> "=" | Neq -> "<>" | Lt -> "<" | Gt -> ">" | Le -> "<=" | Ge -> ">="
  | And -> "&&" | Or -> "||"

let rec string_of_expr = function
  | EInt (n, _) -> string_of_int n
  | EFloat (f, _) -> string_of_float f
  | EBool (b, _) -> string_of_bool b
  | EUnit _ -> "()"
  | EVar (x, _) -> x
  | EBinop (op, e1, e2, _) ->
      Printf.sprintf "(%s %s %s)"
        (string_of_expr e1) (string_of_binop op) (string_of_expr e2)
  | ESample (e, _) -> Printf.sprintf "sample(%s)" (string_of_expr e)
  | EObserve (d, v, _) ->
      Printf.sprintf "observe(%s, %s)" (string_of_expr d) (string_of_expr e)
  | _ -> "..."  (* Abbreviated for brevity *)
```

**Tests:**

```ocaml
(* test_ast.ml *)
open OUnit2
open Ast

let test_ast_construction _ =
  (* Simple expression: 1 + 2 *)
  let expr = EBinop (Add,
                     EInt (1, dummy_loc),
                     EInt (2, dummy_loc),
                     dummy_loc) in
  assert_equal (string_of_expr expr) "(1 + 2)"

let test_probabilistic_expr _ =
  (* sample(normal(0.0, 1.0)) *)
  let dist = EDist (DNormal (EFloat (0.0, dummy_loc),
                             EFloat (1.0, dummy_loc)),
                    dummy_loc) in
  let sample_expr = ESample (dist, dummy_loc) in
  assert_equal (string_of_expr sample_expr) "sample(normal(0.0, 1.0))"

let suite = "AST Tests" >::: [
  "construction" >:: test_ast_construction;
  "probabilistic" >:: test_probabilistic_expr;
]
```

---

## TURN 2 — Lexer and Parser with Menhir

**Instructions:**

Implement lexer (using `ocamllex`) and parser (using `Menhir`) for the probabilistic language.

**Background:** Need to convert source code text into AST. Menhir generates efficient LALR parsers from grammar specifications.

**Requirements:**
- Lexer handles keywords, identifiers, literals, operators
- Parser builds AST from token stream
- Good error messages with source locations
- Support for comments and whitespace

**Implement:**

```ocaml
(* lexer.mll - Lexer specification *)
{
  open Parser
  open Lexing

  exception Lexer_error of string

  let next_line lexbuf =
    let pos = lexbuf.lex_curr_p in
    lexbuf.lex_curr_p <-
      { pos with pos_bol = lexbuf.lex_curr_pos;
                 pos_lnum = pos.pos_lnum + 1 }
}

let white = [' ' '\t']+
let newline = '\r' | '\n' | "\r\n"
let digit = ['0'-'9']
let int = '-'? digit+
let frac = '.' digit*
let exp = ['e' 'E'] ['-' '+']? digit+
let float = digit* frac? exp?
let letter = ['a'-'z' 'A'-'Z']
let ident = letter (letter | digit | '_')*

rule token = parse
  | white    { token lexbuf }
  | newline  { next_line lexbuf; token lexbuf }
  | "(*"     { comment lexbuf }
  | "//"     { line_comment lexbuf }
  
  (* Keywords *)
  | "let"      { LET }
  | "in"       { IN }
  | "if"       { IF }
  | "then"     { THEN }
  | "else"     { ELSE }
  | "fun"      { FUN }
  | "model"    { MODEL }
  | "sample"   { SAMPLE }
  | "observe"  { OBSERVE }
  | "return"   { RETURN }
  | "true"     { TRUE }
  | "false"    { FALSE }
  
  (* Distribution constructors *)
  | "normal"      { NORMAL }
  | "bernoulli"   { BERNOULLI }
  | "categorical" { CATEGORICAL }
  | "beta"        { BETA }
  | "gamma"       { GAMMA }
  
  (* Operators *)
  | '+'  { PLUS }
  | '-'  { MINUS }
  | '*'  { TIMES }
  | '/'  { DIV }
  | '%'  { MOD }
  | '='  { EQ }
  | "<>" { NEQ }
  | '<'  { LT }
  | '>'  { GT }
  | "<=" { LE }
  | ">=" { GE }
  | "&&" { AND }
  | "||" { OR }
  | '!'  { NOT }
  
  (* Delimiters *)
  | '('  { LPAREN }
  | ')'  { RPAREN }
  | '['  { LBRACK }
  | ']'  { RBRACK }
  | ','  { COMMA }
  | ';'  { SEMI }
  | ':'  { COLON }
  | "->" { ARROW }
  | "=>" { DARROW }
  
  (* Literals *)
  | int as i    { INT (int_of_string i) }
  | float as f  { FLOAT (float_of_string f) }
  | ident as id { IDENT id }
  
  | eof { EOF }
  | _ as c { raise (Lexer_error (Printf.sprintf "Unexpected character: %c" c)) }

and comment = parse
  | "*)" { token lexbuf }
  | newline { next_line lexbuf; comment lexbuf }
  | _ { comment lexbuf }
  | eof { raise (Lexer_error "Unclosed comment") }

and line_comment = parse
  | newline { next_line lexbuf; token lexbuf }
  | _ { line_comment lexbuf }
  | eof { EOF }
```

**Parser:**

```ocaml
(* parser.mly - Parser specification *)
%{
  open Ast
  
  let make_loc (startpos, endpos) = {
    file = startpos.Lexing.pos_fname;
    line = startpos.Lexing.pos_lnum;
    col = startpos.Lexing.pos_cnum - startpos.Lexing.pos_bol;
  }
%}

%token <int> INT
%token <float> FLOAT
%token <string> IDENT
%token LET IN IF THEN ELSE FUN MODEL SAMPLE OBSERVE RETURN
%token TRUE FALSE
%token NORMAL BERNOULLI CATEGORICAL BETA GAMMA
%token PLUS MINUS TIMES DIV MOD
%token EQ NEQ LT GT LE GE AND OR NOT
%token LPAREN RPAREN LBRACK RBRACK COMMA SEMI COLON ARROW DARROW
%token EOF

%left OR
%left AND
%left EQ NEQ LT GT LE GE
%left PLUS MINUS
%left TIMES DIV MOD
%nonassoc NOT

%start <Ast.program> program

%%

program:
  | decls = list(decl); EOF { decls }

decl:
  | LET; name = IDENT; EQ; e = expr
    { DLet (name, None, e, make_loc $loc) }
  | MODEL; name = IDENT; LPAREN; params = separated_list(COMMA, param); RPAREN; EQ; e = expr
    { DModel (name, params, e, make_loc $loc) }

param:
  | name = IDENT; COLON; t = ty { (name, t) }

ty:
  | IDENT { match $1 with
            | "int" -> TInt
            | "float" -> TFloat
            | "bool" -> TBool
            | "unit" -> TUnit
            | s -> TVar s }
  | t1 = ty; ARROW; t2 = ty { TFun (t1, t2) }
  | t = ty; IDENT { match $2 with
                    | "dist" -> TDist t
                    | "prob" -> TProb t
                    | "list" -> TList t
                    | _ -> failwith "Unknown type constructor" }

expr:
  | i = INT { EInt (i, make_loc $loc) }
  | f = FLOAT { EFloat (f, make_loc $loc) }
  | TRUE { EBool (true, make_loc $loc) }
  | FALSE { EBool (false, make_loc $loc) }
  | x = IDENT { EVar (x, make_loc $loc) }
  
  | e1 = expr; op = binop; e2 = expr
    { EBinop (op, e1, e2, make_loc $loc) }
  
  | IF; cond = expr; THEN; e1 = expr; ELSE; e2 = expr
    { EIf (cond, e1, e2, make_loc $loc) }
  
  | LET; x = IDENT; EQ; e1 = expr; IN; e2 = expr
    { ELet (x, e1, e2, make_loc $loc) }
  
  | FUN; x = IDENT; ARROW; e = expr
    { EFun (x, None, e, make_loc $loc) }
  
  | e1 = expr; e2 = expr
    { EApp (e1, e2, make_loc $loc) }
  
  | SAMPLE; LPAREN; d = expr; RPAREN
    { ESample (d, make_loc $loc) }
  
  | OBSERVE; LPAREN; d = expr; COMMA; v = expr; RPAREN
    { EObserve (d, v, make_loc $loc) }
  
  | RETURN; LPAREN; e = expr; RPAREN
    { EReturn (e, make_loc $loc) }
  
  | d = dist_expr { EDist (d, make_loc $loc) }
  
  | LPAREN; e = expr; RPAREN { e }

dist_expr:
  | NORMAL; LPAREN; mu = expr; COMMA; sigma = expr; RPAREN
    { DNormal (mu, sigma) }
  | BERNOULLI; LPAREN; p = expr; RPAREN
    { DBernoulli p }
  | CATEGORICAL; LPAREN; ps = expr; RPAREN
    { DCategorical ps }

%inline binop:
  | PLUS { Add }
  | MINUS { Sub }
  | TIMES { Mul }
  | DIV { Div }
  | EQ { Eq }
  | LT { Lt }
  | AND { And }
  | OR { Or }
```

**Tests:**

```ocaml
(* test_parser.ml *)
open OUnit2

let parse_string s =
  let lexbuf = Lexing.from_string s in
  Parser.program Lexer.token lexbuf

let test_parse_simple _ =
  let prog = parse_string "let x = 42" in
  assert_equal (List.length prog) 1

let test_parse_model _ =
  let prog = parse_string "
    model coin_flip(p: float) =
      sample(bernoulli(p))
  " in
  match prog with
  | [DModel ("coin_flip", [("p", TFloat)], _, _)] -> ()
  | _ -> assert_failure "Expected model declaration"

let suite = "Parser Tests" >::: [
  "simple" >:: test_parse_simple;
  "model" >:: test_parse_model;
]
```

---

**(Due to length, I'll continue with the remaining 14 turns in a structured but more concise format to fit within constraints. The pattern will be similar to previous tasks.)**

## TURN 3 — Type Inference with Hindley-Milner

**Instructions:**

Implement bidirectional type checking and Hindley-Milner type inference for the probabilistic language.

**Key implementation:** Type environment, unification algorithm, constraint generation and solving, support for polymorphic types and probabilistic types (`'a dist`, `'a prob`).

---

## TURN 4 — Intermediate Representation (ANF)

**Instructions:**

Transform AST to A-Normal Form (ANF) for easier analysis and optimization.

**Key implementation:** ANF conversion, let-binding normalization, simplification of nested expressions.

---

## TURN 5 — Force Failure: Type Error in Probabilistic Code

**Ask the AI:**

> "Your type checker accepts this invalid program where a distribution is used as a value. Show the bug and implement proper type checking for probabilistic types."

**Expected failure:** Type system doesn't distinguish between `'a dist` and `'a`.

**Fix:** Proper handling of distribution types, enforce `sample` for extracting values.

---

## TURN 6 — Exact Inference via Enumeration

**Instructions:**

Implement exact inference for discrete probabilistic models using enumeration.

**Key implementation:** Enumerate all possible values, compute joint probability, condition on observations, marginalize.

---

## TURN 7 — MCMC Inference (Metropolis-Hastings)

**Instructions:**

Implement Metropolis-Hastings MCMC sampler for approximate inference.

**Key implementation:** Proposal distribution, acceptance ratio, trace collection, convergence diagnostics.

---

## TURN 8 — Hamiltonian Monte Carlo (HMC)

**Instructions:**

Implement HMC for efficient sampling of continuous distributions.

**Key implementation:** Leapfrog integrator, automatic differentiation for gradients, energy function.

---

## TURN 9 — Automatic Differentiation (Reverse Mode)

**Instructions:**

Implement reverse-mode automatic differentiation for gradient computation.

**Key implementation:** Dual numbers, gradient tape, backpropagation through probabilistic operations.

---

## TURN 10 — Variational Inference (Mean-Field)

**Instructions:**

Implement mean-field variational inference for approximate posterior.

**Key implementation:** Variational family, ELBO optimization, coordinate ascent.

---

## TURN 11 — ADVI (Automatic Differentiation Variational Inference)

**Instructions:**

Implement ADVI for black-box variational inference.

**Key implementation:** Reparameterization trick, stochastic gradient ascent on ELBO.

---

## TURN 12 — Optimization Passes

**Instructions:**

Implement compiler optimizations for probabilistic programs.

**Key implementation:** Constant folding, dead code elimination, common subexpression elimination, distribution fusion.

---

## TURN 13 — Code Generation to OCaml

**Instructions:**

Generate executable OCaml code from optimized IR.

**Key implementation:** Code emission, runtime library integration, efficient probability computations.

---

## TURN 14 — Standard Library of Distributions

**Instructions:**

Implement comprehensive library of probability distributions.

**Key implementation:** 15+ distributions with log-probability, sampling, and gradient functions.

---

## TURN 15 — Benchmark Suite and Validation

**Instructions:**

Create comprehensive benchmark suite comparing against Stan and Pyro.

**Key implementation:** Standard models (8 schools, logistic regression, hierarchical models), performance and accuracy metrics.

---

## TURN 16 — End-to-End Validation and Documentation

**Instructions:**

Final validation of all components and comprehensive documentation.

**Deliverables:** Complete language specification, API docs, tutorial, performance report.

---

**Final Deliverables:**

1. ✅ Complete compiler (lexer, parser, type checker, optimizer, codegen)
2. ✅ Inference engine (exact, MCMC, variational)
3. ✅ Automatic differentiation
4. ✅ Standard library (15+ distributions)
5. ✅ Comprehensive test suite (>150 tests)
6. ✅ Benchmark suite (vs Stan, Pyro)
7. ✅ Complete documentation

**Estimated completion time:** 55-70 hours for expert OCaml/PL/ML engineer

**Difficulty:** EXTREME - requires mastery of programming language theory, compiler construction, probabilistic inference, and functional programming.
