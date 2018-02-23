/* glpk.js is now included (cat) in lalolib.js
if ( self.hasOwnProperty("window") ) {
	// in main window 
}
else { 
	// in worker
	importScripts("glpk.js");
	//importScripts("glpk.min.js");
}*/

// Install glpk as lp function: 
if ( typeof(lp) == "undefined" ) {
	lp = glp;
	linprog = glp;
}

function glp (c, A, b, Aeq, beq, lb , ub, integer_variables, verbose) {
/*
	Call GLPK to solve 
	min c' x s.t. Ax<= b, Aeq = beq, lb<= x <= ub, x[integer_variables] in Z
*/

/* TESTS:
Aineq = [[1, 1]; [-1,1]]
Bineq = [2; 1]
costineq = [-1; -2]
lb = [0;0]
xsol = glp(costineq, Aineq, Bineq, [], [], lb)

A = [[3,2,1,1,0],[2,5,3,0,1]]
b=[10,15]
c=[-2,-3,-4,0,0]
lb = zeros(5)
xsol = glp(c, [],[],A, b,lb,[])

*/
	var prob = glp_create_prob();
	glp_set_obj_dir ( prob, GLP_MIN ) ;
	
	if ( typeof(Aeq) == "undefined" )
		var Aeq = [];
	
	glp_add_cols(prob, c.length);
	if ( A.length + Aeq.length > 0 )
		glp_add_rows(prob, A.length + Aeq.length);

	var i;
	var j;
	var indexes ;
	var values;
	var n = c.length;
	
	if ( lb ) {
		var lbdense = vectorCopy(lb);
		for ( i=0; i < lbdense.length; i++){
			if ( !isFinite( lbdense[i] ) )
				lbdense[i] = NaN;
		}
	}
	else 
		var lbdense = [];

	if ( ub ) {
		var ubdense = vectorCopy(ub);
		for ( i=0; i < ubdense.length; i++){
			if ( !isFinite( ubdense[i] ) )
				lbdense[i] = NaN;
		}
	}
	else 
		var ubdense = [];
	
	for ( i=0; i < c.length; i++) {
		// variable bounds
		var lbi = NaN;
		var ubi = NaN; 
		if ( lbdense.length > 0)	
			lbi = lbdense[i];
		if ( ubdense.length > 0 )
			ubi = ubdense[i] ;
			
		if ( !isNaN(lbi)  && !isNaN(ubi)) 
			glp_set_col_bnds( prob, i+1, GLP_DB, lbi , ubi );
		else if ( !isNaN(lbi) )
			glp_set_col_bnds( prob, i+1, GLP_LO, lbi );
		else if ( !isNaN(ubi) )
			glp_set_col_bnds( prob, i+1, GLP_UP, 0, ubi );
		else 
			glp_set_col_bnds( prob, i+1, GLP_FR );
			
		// cost
		glp_set_obj_coef ( prob, i+1, c[i]  );
		
	}	
	
	// Integer variables
	if ( integer_variables ) {
		for ( i=0; i< integer_variables.length ; i++) 
			glp_set_col_kind(prob, integer_variables[i]+1, GLP_IV );
	}
	
	// inequalities
	if ( A.length == 1 && typeof(b) == "number")
		b = [b];
	for ( i=0; i<A.length; i++) {
		
		// RHS		
		glp_set_row_bnds(prob, i+1, GLP_UP, 0, b[i] );	// pass lb=0 otherwise ub undefined!!
		
		// LHS
		indexes = new Array(); 	
		values = new Array(); 	
		indexes.push(0);	// to make it start at 1
		values.push(0); 	
		for ( j = 0; j < n; j++ ) {
			if ( !isZero(A.val[i*n+j] )) {
				indexes.push(j+1);
				values.push( A.val[i*n+j] );
			}
		}
		glp_set_mat_row( prob, i+1, indexes.length -1, indexes, values) ;
		
	}

	// equality constraints	
	if ( Aeq.length == 1 && typeof(beq) == "number")
		beq = [beq];
	for ( i=0; i<Aeq.length; i++) {
		
		// RHS		
		glp_set_row_bnds(prob, A.length+i+1, GLP_FX, beq[i] );
				
		// LHS
		indexes = new Array(); 
		values = new Array(); 		
		indexes.push(0);	// to make it start at 1
		values.push(0); 	
		for ( j = 0; j < n; j++ ) {
			if (  !isZero(Aeq.val[i*n+j] )) {
				indexes.push(j+1);
				values.push( Aeq.val[i*n+j] );
			}
		}
		glp_set_mat_row( prob,A.length+ i+1, indexes.length -1, indexes, values) ;
	}

	//glp_write_lp(prob, undefined, function (str) {console.log(str);});

	var rc;
	if ( integer_variables && integer_variables.length > 0) {
		// Solve with MILP solver
		var iocp = new IOCP({presolve: GLP_ON});
		glp_scale_prob(prob, GLP_SF_AUTO);
		rc = glp_intopt(prob, iocp);
		
		// get solution
		if ( rc == 0 ) {
			var sol = zeros(n);
			for ( i=0; i<n; i++) {
				sol[i] = glp_mip_col_val( prob, i+1);
			}
			
			if ( verbose) {
				var obj = glp_mip_obj_val(prob);
				console.log("Status : " + glp_mip_status(prob) );
				console.log("Obj : " + obj);
			}
			return sol;
		}
		else
			return "Status : " + glp_get_prim_stat(prob);
	}
	else {
		// Parameters
		var smcp = new SMCP({presolve: GLP_ON});
		// Solve with Simplex
		glp_scale_prob(prob, GLP_SF_AUTO);
		rc = glp_simplex(prob, smcp);
		
		// get solution
		if ( rc == 0 ) {
			var sol = zeros(n);
			for ( i=0; i<n; i++) {
				sol[i] = glp_get_col_prim( prob, i+1);
			}
			if ( verbose) {
				var obj = glp_get_obj_val(prob);
				console.log("Status : " + glp_get_status(prob) + "(OPT=" + GLP_OPT + ",FEAS=" + GLP_FEAS + ",INFEAS=" + GLP_INFEAS + ",NOFEAS=" + GLP_NOFEAS + ",UNBND=" + GLP_UNBND + ",UNDEF=" + GLP_UNDEF + ")" );
				console.log("Obj : " + obj);
			}
			return sol;
		}
		else {
			GLPLASTLP = "";
			glp_write_lp(prob, undefined, function (str) {GLPLASTLP += str + "<br>";});
			return "RC=" + rc + " ; Status : "  + glp_get_status(prob) + "(OPT=" + GLP_OPT + ",FEAS=" + GLP_FEAS + ",INFEAS=" + GLP_INFEAS + ",NOFEAS=" + GLP_NOFEAS + ",UNBND=" + GLP_UNBND + ",UNDEF=" + GLP_UNDEF + ")" ;
		}
	}
	
}

///////////////////////////////:
/////// L1-minimization and sparse recovery //////////
///////////
function minl1 ( A, b) {
	/*
		Solves min ||x||_1 s.t. Ax = b
		
		as 
		
			min sum a_i s.t. -a <= x <= a and Ax = b
			
		example: 
A = randn(10,20)
r = zeros(20)
r[0:3] = randn(3)
x=minl1(A,A*r)

	*/
	const n = A.n;
	
	var Aineq = zeros ( 2*n, 2*n ) ;
	var i;
	
	//set ( Aineq, range(0,n),range(0,n) , I) ;
	//set ( Aineq, range(0,n),range(n,2*n) , I_) ;
	//set ( Aineq, range(n,2*n),range(0,n) , I_) ;
	//set ( Aineq, range(n,2*n),range(n,2*n) , I_) ;
	for ( i=0; i < n; i++) {
		Aineq.val[i*Aineq.n + i] = 1;
		Aineq.val[i*Aineq.n + n+i] = -1;
		Aineq.val[(n+i)*Aineq.n + i] = -1;
		Aineq.val[(n+i)*Aineq.n + n+i] = -1;
	}
	var bineq = zeros ( 2*n);
	
	var Aeq = zeros(A.length, 2*n);
	set ( Aeq , [], range( 0,n), A );
	
	var cost = zeros(2*n);
	set ( cost, range(n,2*n),  ones(n) );
		
	var lb = zeros(2*n);	// better to constraint a>=0
	set ( lb, range(n), mulScalarVector(-Infinity , ones( n )) ) ;	
//console.log( cost, Aineq, bineq, Aeq, b, lb);	
//	var lpsol = lp( cost, Aineq, bineq, Aeq, b, lb, [], 0 , 1e-6 );
	var lpsol = glp( cost, Aineq, bineq, Aeq, b, lb);	

	return get(lpsol, range(n) );
}



function minl0 ( A, b, M) {
	/*
		Solves min ||x||_0 s.t. Ax = b  -M <= x <= M
		
		as a mixed integer linear program
		
			min sum a_i s.t. -M a <= x <= M a , Ax = b and a_i in {0,1}
			
		example: 
A = randn(10,20)
r = zeros(20)
r[0:3] = randn(3)
x=minl0(A,A*r)

	*/
	
	if ( typeof(M) == "undefined" ) 
		var M = 10;
		
	var n = A.n;
	
	var Aineq = zeros ( 2*n, 2*n ) ;
	//set ( Aineq, range(0,n),range(0,n) , I) ;
	//set ( Aineq, range(0,n),range(n,2*n) , mul(M, I_) ) ;
	//set ( Aineq, range(n,2*n),range(0,n) , I_) ;
	//set ( Aineq, range(n,2*n),range(n,2*n) ,mul(M, I_) ) ;
	var i;
	for ( i=0; i < n; i++) {
		Aineq.val[i*Aineq.n + i] = 1;
		Aineq.val[i*Aineq.n + n+i] = -M;
		Aineq.val[(n+i)*Aineq.n + i] = -1;
		Aineq.val[(n+i)*Aineq.n + n+i] = -M;

	}
	var bineq = zeros ( 2*n);
	
	var Aeq = zeros(A.length, 2*n);
	set ( Aeq , [], range( 0,n), A );
	
	var cost = zeros(2*n);
	set ( cost, range(n,2*n),  ones(n) );
		
	var lb = zeros(2*n);	// better to constraint a>=0
	set ( lb, range(n), mulScalarVector(-M , ones( n )) ) ;	

	var ub =  ones(2*n) ;
	set(ub, range(n), mulScalarVector(M, ones(n) ) );
	
	var lpsol = glp( cost, Aineq, bineq, Aeq, b, lb, ub, range(n,2*n) );	// glptweak??

	// set to 0 the x corresponding to 0 binary variables:
	var x = entrywisemulVector( getSubVector(lpsol, range(n) ), getSubVector(lpsol, range(n,2*n) ) );

	return x;
}


///////////////////////////////////////////
/// Quadratic Programming 
////////////////
quadprog = qp;

function qp(Q,c,A,b,Aeq,beq,lb,ub,x0, epsilon) {
	// Solve quad prog by Frank-Wolfe algorithm
	/*
		min 0.5 x' * Q * x  c' * x
		s.t. Ax <= b   and   lu <= x <= ub
		
		NOTE: all variables should be bounded or constrained,
		otherwise the LP might be unbounded even if the QP is well-posed
	*/
	if (typeof(epsilon) === 'undefined')
		var epsilon = 1e-3;
		
	var normdiff;
	var normgrad;
	var grad;
	var y;
	var gamma;
	var direction;
	
	var x;
	if ( typeof(x0) === 'undefined' ) {		
		//console.log ("providing an initial x0 might be better for qp.");		
		x = glp(zeros(c.length),A, b, Aeq, beq, lb, ub, [], false) ;  
		if ( typeof(x) == "string")
			return "infeasible";
	}
	else {
		x = vectorCopy(x0);
	}

	var iter = 0;
	do {

		// Compute gradient : grad = Qx + c
		grad = add( mul( Q, x) , c );
		normgrad = norm(grad);

		// Find direction of desecnt : direction = argmin_y   y'*grad s.t. same constraints as QP
		y = glp(grad, A, b, Aeq, beq, lb, ub, [], false) ; 
/*		if ( typeof(y) == "string") 
			return x; // error return current solution;
	*/

		// Step size: gamma = -(y - x)' [ Qx + c] / (y-x)'Q(y-x) = numerator / denominator
		direction = sub (y, x);
		
		numerator = - mul(direction, grad);
		
		denominator = mul(direction, mul(Q, direction) ); 

		if ( Math.abs(denominator) > 1e-8 && denominator > 0)
			gamma = numerator / denominator; 
		else 
			gamma = 0;
			
		if ( gamma > 1 ) 
			gamma = 1;

		// Update x <- x + gamma * direction
		if ( gamma > 0 ) {
			x = add(x, mul(gamma, direction) );		
			normdiff = gamma * norm(direction) ;
		}
		else 
			normdiff = 0;

		iter++;
	} while ( normdiff > epsilon && normgrad > epsilon && iter < 10000) ;

	return x;
}


