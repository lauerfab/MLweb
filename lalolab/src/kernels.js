/*
	Kernel functions : 
		rbf
		poly
		polyh
		linear
		
		For custom kernels: typeof(kerneltype) = function (x1,x2,par)
		
		TODO: kernel() for spVectors
*/

/**
 * Evaluate a kernel function from its name/type
 * @param {(number|Float64Array|spVector)}
 * @param {(number|Float64Array|spVector)}
 * @param {(string|function)}
 * @param {?number}
 * @return {number} 
 */
function kernel(x1, x2, kerneltype, par) {
	if ( typeof(kerneltype) === 'undefined')
		var kerneltype = "rbf"; 
	if (kerneltype != "linear" && typeof(par) === 'undefined') {
		var par = kernel_default_parameter(kerneltype, size(x1,1));
	}
	
	if (typeof(kerneltype) == "function" ) {
		// Custom kernel function
		return kerneltype(x1,x2,par);
	}
	
	var ker;
	switch (kerneltype){
		case "gaussian":
		case "Gaussian":
		case "RBF":
		case "rbf": 
			if ( typeof(x1) == "number") 
				ker = rbfkernelScalar(x1, x2, 1/(2 * par * par) );
			else 
			 	ker = rbfkernel(x1, x2, 1/(2 * par * par) );
			break;		
		case "poly":
			if ( typeof(x1)=="number")
				ker = polykernelScalar(x1,x2,par);
			else
				ker = polykernel(x1,x2,par);
			break;
		case "polyh":
			if ( typeof(x1)=="number")
				ker = polyhkernelScalar(x1,x2,par);
			else
				ker = polyhkernel(x1,x2,par);
			break;
			
		case "linear":
			if ( typeof(x1)=="number")
				ker = x1*x2;
			else
				ker = dot(x1,x2);			
			break;
			
		default:
			ker = NaN;
			break;
	}
	
	return ker;
}
/**
 * Return a kernel function K to use as K(x1,x2)
 * @param {(string|function)}
 * @param {?(number|Float64Array)}
 * @param {string} 
 * @return {function} 
 */
function kernelFunction(kerneltype, par, inputType) {
	if ( typeof(kerneltype) === 'undefined')
		var kerneltype = "rbf"; 
	if (kerneltype != "linear" && typeof(par) === 'undefined') {
		var par = kernel_default_parameter(kerneltype);
	}
	if ( typeof(inputType) != 'string')
		var inputType = "vector"; 
	
	if (typeof(kerneltype) == "function" ) {
		// Custom kernel function
		return function ( x1, x2) {return kerneltype(x1,x2,par);};
	}
	
	var ker;
	switch (kerneltype){
		case "gaussian":
		case "Gaussian":
		case "RBF":
		case "rbf": 
			const gamma =  1/(2 * par * par);
			switch(inputType) {
			case "number":
			case "scalar":
				ker = function ( x1, x2) {return rbfkernelScalar(x1, x2, gamma );};
				break;
			case "vector":
			 	ker = function ( x1, x2) {return rbfkernel(x1, x2, gamma );};
			 	break;
			case "spvector":
				ker = function ( x1, x2) {return rbfkernelsparse(x1, x2, gamma );};		
				break;
			default:
				error("Unknown input type in kernelFunction ()");
			 	break;
			}	 	
			break;		
		case "poly":
			switch(inputType) {
			case "number":
			case "scalar":
				ker = function ( x1, x2) {return polykernelScalar(x1,x2,par);};
				break;
			case "vector":			
				ker = function ( x1, x2) {return polykernel(x1,x2,par);};
				break;
			default:
				error("Unknown input type in kernelFunction ()");
			 	break;
			}	 	
			break;
		case "polyh":
			switch(inputType) {
			case "number":
			case "scalar":
				ker = function ( x1, x2) {return polyhkernelScalar(x1,x2,par);};
				break;
			case "vector":			
				ker = function ( x1, x2) {return polyhkernel(x1,x2,par);};
				break;
			default:
				error("Unknown input type in kernelFunction ()");
			 	break;
			}	 	
			break;
		case "linear":
			switch(inputType) {
			case "number":
			case "scalar":
				ker = function ( x1, x2) {return x1*x2;};
				break;
			case "vector":			
				ker = function ( x1, x2) {return dot(x1,x2);};	
				break;
			case "spvector":			
				ker = function ( x1, x2) {return spdot(x1,x2);};	
				break;
			default:
				error("Unknown input type in kernelFunction ()");
			 	break;
			}	 	
			break;
			
		default:
			ker = function ( x1, x2) {return NaN;}
			break;
	}
	
	return ker;
}
/**
 * Scalar Gaussian RBF Kernel function K(x1,x2) = exp(-gamma (x1-x2)^2)
 * @param {number}
 * @param {number}
 * @return {number} 
 */
function rbfkernelScalar(x1, x2, gamma) {
	var diff = x1-x2;
	return Math.exp(-diff*diff * gamma);
}
/*function rbfkernel(x1, x2, gamma) {
	var diff = subVectors(x1,x2);
	return Math.exp(-dot(diff,diff) * gamma);
}*/
/**
 * Gaussian RBF Kernel function K(x1,x2) = exp(-gamma ||x1-x2||^2)
 * @param {Float64Array}
 * @param {Float64Array}
 * @return {number} 
 */
function rbfkernel( x1, x2, gamma) {
	// Fast evaluation with
	// ||x1-x2||^2 > thresh => exp(-||x1-x2||^2/2sigma^2) < EPS ~= 0
	const n = x1.length;
	const thresh = 50 / gamma;
	var diff = x1[0] - x2[0];
	var sqdist = diff*diff;
	var j = 1;
	while ( j < n && sqdist < thresh ) {
		diff = x1[j] - x2[j];
		sqdist += diff*diff; 
		j++;
	}
	if ( j < n ) 
		return 0;
	else
		return Math.exp(-sqdist * gamma);
}
/**
 * Gaussian RBF Kernel function K(x1,x2) = exp(-gamma ||x1-x2||^2)
 * for sparse vectors
 * @param {spVector}
 * @param {spVector}
 * @return {number} 
 */
function rbfkernelsparse( x1, x2, gamma) {
	// Fast evaluation with
	// ||x1-x2||^2 > thresh => exp(-||x1-x2||^2/2sigma^2) < EPS ~= 0

	const thresh = 50 / gamma;	
	const nnza = x1.val.length;
	const nnzb = x2.val.length;
	
	var k1 = 0;
	var k2 = 0;
	var i1;
	var i2;
	var diff;
	var sqdist = 0;
	
	while ( k1 < nnza && k2 < nnzb && sqdist < thresh ) {

		i1 = x1.ind[k1];
		i2 = x2.ind[k2];
		if ( i1 == i2 ) {
			diff = x1.val[k1] - x2.val[k2];
			k1++;
			k2++;
		}
		else if ( i1 < i2 ) {
			diff = x1.val[k1];
			k1++;
		}
		else {
			diff = x2.val[k2];
			k2++;
		}
		sqdist += diff*diff;
	}
	
	while( k1 < nnza && sqdist < thresh ) {
		diff = x1.val[k1];
		sqdist += diff*diff;		
		k1++;
	}
	while ( k2 < nnzb && sqdist < thresh ) {
		diff = x2.val[k2];	
		sqdist += diff*diff;
		k2++;
	}
	
	if ( sqdist >= thresh ) 
		return 0;
	else
		return Math.exp(-sqdist * gamma);
}

/**
 * Scalar polynomial Kernel function K(x1,x2) = (x1*x2 + 1)^deg
 * @param {number}
 * @param {number}
 * @return {number} 
 */
function polykernelScalar(x1, x2, deg) {
	var x1x2 = x1*x2 + 1;
	var ker = x1x2;
	for ( var i=1; i < deg; i++)
		ker *= x1x2;
	return ker;
}
/**
 * polynomial Kernel function K(x1,x2) = (x1'x2 + 1)^deg
 * @param {Float64Array}
 * @param {Float64Array}
 * @return {number} 
 */
function polykernel(x1, x2, deg) {
	var x1x2 = dot(x1,x2) + 1;
	var ker = x1x2;
	for ( var i=1; i < deg; i++)
		ker *= x1x2;
	return ker;
}
/**
 * polynomial Kernel function K(x1,x2) = (x1'x2 + 1)^deg
 * @param {spVector}
 * @param {spVector}
 * @return {number} 
 */
function polykernelsparse(x1, x2, deg) {
	var x1x2 = spdot(x1,x2) + 1;
	var ker = x1x2;
	for ( var i=1; i < deg; i++)
		ker *= x1x2;
	return ker;
}
/**
 * Scalar homogeneous polynomial Kernel function K(x1,x2) = (x1*x2)^deg
 * @param {number}
 * @param {number}
 * @return {number} 
 */
function polyhkernelScalar(x1, x2, deg) {
	var x1x2 = x1*x2;
	var ker = x1x2;
	for ( var i=1; i < deg; i++)
		ker *= x1x2;
	return ker;
}
/**
 * Homogeneous polynomial Kernel function K(x1,x2) = (x1'x2)^deg
 * @param {Float64Array}
 * @param {Float64Array}
 * @return {number} 
 */
function polyhkernel(x1, x2, deg) {
	var x1x2 = dot(x1,x2);
	var ker = x1x2;
	for ( var i=1; i < deg; i++)
		ker *= x1x2;
	return ker;
}
/**
 * Homogeneous polynomial Kernel function K(x1,x2) = (x1'x2)^deg
 * @param {spVector}
 * @param {spVector}
 * @return {number} 
 */
function polyhkernel(x1, x2, deg) {
	var x1x2 = spdot(x1,x2);
	var ker = x1x2;
	for ( var i=1; i < deg; i++)
		ker *= x1x2;
	return ker;
}

function custom_kernel_example(x1,x2, par) {
	// A custom kernel should define a default parameter (if any)
	if ( typeof(par) == "undefined") 
		var par = 1;
		
	// Return K(x1,x2):
	return mul(x1,x2) + par;
}
function kernel_default_parameter(kerneltype, dimension ) {
	var par;
	switch (kerneltype){
		case "gaussian":
		case "Gaussian":
		case "RBF":
		case "rbf": 
			par = 1;
			break;
			
		case "poly":
			par = 3;
			break;
		case "polyh":
			par = 3;
			break;
			
		case "linear":
			break;
			
		default:			
			break;
	}
	
	return par;
}

function kernelMatrix( X , kerneltype, kernelpar, X2) {
	var i;
	var j;

	if ( typeof(kerneltype) === 'undefined')
		var kerneltype = "rbf"; 
	if ( typeof(kernelpar) === 'undefined')			
		var kernelpar = kernel_default_parameter(kerneltype, size(X,2));

	const tX = type(X);
	var inputtype;
	if (tX == "vector" )
		inputtype = "number";
	else if ( tX == "matrix")
		inputtype = "vector";
	else if ( tX == "spmatrix")
		inputtype = "spvector";
	else
		error("Unknown input type in kernelMatrix().");
		
	var kerFunc = kernelFunction(kerneltype, kernelpar, inputtype);
	
	if ( arguments.length < 4 ) {
		// compute K(X,X) (the Gram matrix of X)
		if ( kerneltype == "linear") {
			return mul( X,transpose(X)); // this should be faster
		}
	
		switch(inputtype) {
			case "number":
				K = kernelMatrixScalars(X, kerFunc) ;
				break;
			case "vector":
				K = kernelMatrixVectors(X, kerFunc) ;
				break;
			case "spvector":
				K = kernelMatrixspVectors(X, kerFunc) ;
				break;
			default:
				K = undefined;
				break;
		}
	}
	else {		
		// compute K(X, X2)		
		var m = X.length;		
		var t2 = type(X2);
		if (t2 == "number" ) {
			// X2 is a single data number:
			if ( kerneltype == "linear") {
				return mul( X,X2); // this should be faster
			}

			var K = zeros(m);			
			for (i = 0; i<m; i++) {		
				K[i] = kerFunc(X[i], X2);
			}
		}
		else if ( t2 == "vector" && tX == "matrix" && X.n >1 )  {
			// X2 is a single data vector :
			if ( kerneltype == "linear") {
				return mul( X,X2); // this should be faster
			}

			var K = zeros(m);			
			for (i = 0; i<m; i++) {		
				K[i] = kerFunc(X.row(i), X2);
			}
		}
		else if ( t2 == "vector" && tX == "vector" )  {
			// X2 is a vector of multiple data instances in dim 1
			if ( kerneltype == "linear") {
				return outerprodVectors( X,X2); // this should be faster
			}
			var n = X2.length;
			var K = zeros(m,n);			
			var ki = 0;
			for (i = 0; i<m; i++) {		
				for (j = 0; j < n; j++) {
					K.val[ki + j] = kerFunc(X[i], X2[j]);
				}		
				ki += n	;
			}
		}
		else {
			// X2 is a matrix
			if ( kerneltype == "linear") {
				return mul( X,transpose(X2)); // this should be faster
			}

			var n = X2.length;
			var K = zeros(m,n);
			var X2j = new Array(n);
			for (j = 0; j < n; j++) {
				X2j[j] = X2.row(j);
			}
			var ki = 0;
			for (i = 0; i<m; i++) {		
				var Xi = X.row(i);
				for (j = 0; j < n; j++) {
					K.val[ki + j] = kerFunc(Xi, X2j[j]);
				}		
				ki += n	;
			}
		}		
		
	}
	return K;
}	
/*function kernelMatrixVectors2(X, kerFunc, kerPar) {
	const n = X.length;
	var K = zeros(n,n);	
	var ri = 0;
	var rj;
	var Xi;
	var ki = 0;
	var Kij;
	for (var i = 0; i<n; i++) {		
		rj = 0;
		Xi = X.val.subarray(ri, ri + X.n);		
		for (var j = 0; j < i; j++) {					
			Kij = kerFunc( Xi, X.val.subarray(rj, rj + X.n), kerPar);
			K.val[ki + j] = Kij;
			K.val[j * n + i] = Kij;		
			rj += X.n;		
		}				
		K.val[i*n + i] = kerFunc(Xi, Xi, kerPar) ;
		ri += X.n;
		ki += n;
	}
	return K;
}*/
function kernelMatrixVectors(X, kerFunc) {
	const n = X.length;
	var K = zeros(n,n);	
	var ri = 0;
	var Xi  = new Array(n);
	var ki = 0;
	var Kij;
	for (var i = 0; i<n; i++) {		
		rj = 0;
		Xi[i] = X.val.subarray(ri, ri + X.n);
		for (var j = 0; j < i; j++) {					
			Kij = kerFunc( Xi[i], Xi[j] );
			K.val[ki + j] = Kij;
			K.val[j * n + i] = Kij;		
		}				
		K.val[i*n + i] = kerFunc(Xi[i], Xi[i]) ;
		ri += X.n;
		ki += n;
	}
	return K;
}
function kernelMatrixspVectors(X, kerFunc) {
	const n = X.length;
	var K = zeros(n,n);	
	var Xi  = new Array(n);
	var ki = 0;
	var Kij;
	for (var i = 0; i<n; i++) {		
		rj = 0;
		Xi[i] = X.row(i);
		for (var j = 0; j < i; j++) {					
			Kij = kerFunc( Xi[i], Xi[j] );
			K.val[ki + j] = Kij;
			K.val[j * n + i] = Kij;		
		}				
		K.val[i*n + i] = kerFunc(Xi[i], Xi[i]) ;
		ki += n;
	}
	return K;
}
function kernelMatrixScalars(X, kerFunc, kerPar) {
	const n = X.length;
	var K = zeros(n,n);	
	var ki = 0;
	var Kij;	
	for (var i = 0; i<n; i++) {			
		for (var j = 0; j < i; j++) {
			Kij = kerFunc(X[i], X[j], kerPar);
			K.val[ki + j] = Kij;
			K.val[j * n + i] = Kij;				
		}
		K.val[ki + i] = kerFunc(X[i], X[i], kerPar) ;
		ki += n;
	}
	return K;	
}
function kernelMatrixUpdate( K , kerneltype, kernelpar, previouskernelpar) {
	/*
		Kernel matrix update for RBF and poly kernels
		K = pow(K, ... ) 
		This is an in place update that destroys previous K
		
		NOTE: with the poly kernel, all entries in K must be positive or kernelpar/previouskernelpar must be an integer.
	*/
	if ( arguments.length < 4 )
		return undefined; 
		
	const n = K.length;

	switch ( kerneltype ) {
		case "gaussian":
		case "Gaussian":
		case "RBF":
		case "rbf": 
			var power = previouskernelpar/kernelpar;
			power *= power;
			break;
		case "poly":			
		case "polyh":
			var power = kernelpar/previouskernelpar;
			break;	
		case "linear":
			return K;
			break;	
		default:
			return "undefined";
	}
	var i;
	var j;
	var ki = 0;
	if ( Math.abs(power - 2) < 1e-10 ) {
		// fast updates for K = K^2
		for (i = 0; i<n; i++) {		
			for (j = 0; j < i; j++) {
				K.val[ki + j] *= K.val[ki + j];
				K.val[j*n + i] = K.val[ki + j];//symmetric part
			}	
			K.val[ki + i] *= K.val[ki + i];
			ki += n;	
		}
	}
	else {
		// otherwise do K = pow(L, power)
		for (i = 0; i<n; i++) {		
			for (j = 0; j < i; j++) {
				K.val[ki + j] = Math.pow( K.val[ki + j], power );			
				K.val[j*n + i] = K.val[ki + j];
			}	
			K.val[ki + i] = Math.pow( K.val[ki + i], power );			
		
			ki += n;	
		}
	}
	
	return K;
}
//////////::
// Kernel Cache 
//////////////////
/**
 * @constructor
 */
function kernelCache ( X , kerneltype, kernelpar, cacheSize) { 
	// Create a kernel cache of a given size (number of entries)
	
	if ( typeof(kerneltype) === 'undefined')
		var kerneltype = "rbf"; 
	if ( typeof(kernelpar) === 'undefined')			
		var kernelpar = kernel_default_parameter(kerneltype, size(X,2));
	if  ( typeof(cacheSize) == "undefined" )
		var cacheSize = Math.floor(64 * 1024 * 1024) ; // 64*1024*1024 entries = 512 MBytes 
													   // = enough for the entire K with up to 8000 data 
	if ( cacheSize < X.length * 10 ) 
		cacheSize = X.length * 10; // cache should be large enough for 10 rows of K	
		
	const tX = type(X);
	var inputtype = "vector";
	if ( tX == "matrix")
		this.X = matrixCopy(X);
	else if (tX == "spmatrix" ) {
		if ( X.rowmajor ) 
			this.X = X.copy();
		else
			this.X = X.toRowmajor();
		inputtype = "spvector";
	}
	else
		this.X = new Matrix(X.length, 1, X); // X is a vector of numbers => single column matrix 


	this.Xi = new Array(this.X.length);
	for ( var i = 0; i< this.X.length; i++)
		this.Xi[i] = this.X.row(i);


	this.kerneltype = kerneltype;
	this.kernelpar = kernelpar;
	this.kernelFunc = kernelFunction(kerneltype, kernelpar, inputtype); // X transformed to matrix in any case (Xi = (sp)vector)
	this.inputtype = inputtype;
	
	this.previouskernelpar = undefined;	// for kernel updates
	this.rowsToUpdate = 0;
	this.needUpdate = undefined;
	
	this.cachesize = cacheSize; // in number of Kij entries
	this.size = Math.min( Math.floor(cacheSize / X.length), X.length ); // in number of rows;
	this.rowlength = X.length;
	
	this.K = zeros(this.size, X.length );
	this.rowindex = new Array(); // list of rows indexes: rowindex[i] = index of X_j whose row is store in K[i]
	this.LRU = new Array(); 	// list of last recently used rows (index in K) 
								// the last recently used is at the beginning of the list
}

/**
 * @param {number}
 * @return {Float64Array}
 */
kernelCache.prototype.get_row = function ( row ) {
	var j;
	var Krow;
	var i = this.rowindex.indexOf( row );
	
	if ( i >= 0 ) {
		// requested row already in cache
		this.updateRow(i); // update row if needed due to kernelpar change
		Krow = this.K.row(i); // for thread safe :  vectorCopy(K[i]); 
	}
	else {
		// Need to compute the new row
		
		// Find an index to store the row
		if (this.rowindex.length < this.size) {
			// There is space for this row, so append at the end
			i = this.rowindex.length;
			this.rowindex.push( row ) ; 			
		}
		else {
			// Need to erase the last recenty used:
			i = this.LRU[0]; 
			this.rowindex[i] = row;
		}
		
		// Compute kernel row
		Krow = this.K.row(i);
		//var Xrow = this.X.row(row);
		for (j = 0; j < this.rowlength; j++) {
			//Krow[j] = this.kernelFunc(Xrow, this.X.row(j));
			Krow[j] = this.kernelFunc(this.Xi[row], this.Xi[j]);
		}
	}
	
	// Update LRU
	var alreadyInLRU = this.LRU.indexOf(i);
	if ( alreadyInLRU >= 0)
		this.LRU.splice(alreadyInLRU, 1);
	this.LRU.push(i); 
	
	return Krow;	
}


kernelCache.prototype.update = function( kernelpar ) {
	// update the kernel parameter to kernelpar
	// careful: each row may have been computed with a different kernelpar (if not used during last training...)
	
	if( typeof(this.needUpdate) == "undefined" )
		this.needUpdate = new Array(this.rowindex.length);

	if( typeof(this.previouskernelpar) == "undefined" )
		this.previouskernelpar = new Array(this.rowindex.length);

	for (var i = 0; i < this.rowindex.length; i++) {
		if ( typeof(this.needUpdate[i]) == "undefined" || this.needUpdate[i] == false ) { // otherwise keep previous values			
			this.needUpdate[i] = true;
			this.previouskernelpar[i] = this.kernelpar; 
			this.rowsToUpdate++;
		}	
	}
	
	this.kernelpar = kernelpar;
	this.kernelFunc = kernelFunction(this.kerneltype, kernelpar, this.inputtype); // X transformed to matrix in any case (Xi = (sp)vector)
	
}
kernelCache.prototype.updateRow = function( i ) {
	// update the kernel row in the ith row of the cache
	
	if ( this.rowsToUpdate > 0 && typeof(this.needUpdate[i]) != "undefined" && this.needUpdate[i]) {	
		switch ( this.kerneltype ) {
			case "gaussian":
			case "Gaussian":
			case "RBF":
			case "rbf": 
				var power = this.previouskernelpar[i] / this.kernelpar;
				power *= power;
				break;
			case "poly":			
			case "polyh":
				var power = this.kernelpar / this.previouskernelpar[i];
				break;	
			default:
				return ;
		}
		var pr = Math.round(power);
		if ( Math.abs(pr - power) < 1e-12 ) 
			power = pr;
			
		var j;
		var Krow = this.K.row(i);
		if ( power == 2 ) {
			for ( j = 0; j < this.rowlength ; j++) 
				Krow[j] *= Krow[j] ;
		}
		else {
			for ( j = 0; j < this.rowlength ; j++) 
				Krow[j] = Math.pow( Krow[j], power );
		}
		
		// Mark row as updated: 
		this.needUpdate[i] = false;
		this.previouskernelpar[i] = this.kernelpar;
		this.rowsToUpdate--;
	}
}

//////////::
// Parallelized Kernel Cache 
/*
	If N < 20000, then K < 3 GB can be precomputed entirely with 
	(#CPUs-1) processors in the background. 

	for get_row(i):
	if K[i] is available, return K[i]
	otherwise, K[i] is computed by the main thread and returned.

//////////////////
function kernelCacheParallel( X , kerneltype, kernelpar, cacheSize) { 
	// Create a kernel cache of a given size (number of entries)
	
	if ( typeof(kerneltype) === 'undefined')
		var kerneltype = "rbf"; 
	if ( typeof(kernelpar) === 'undefined')			
		var kernelpar = kernel_default_parameter(kerneltype, size(X,2));
	if  ( typeof(cacheSize) == "undefined" )
		var cacheSize = Math.floor(3 * 1024 * 1024 * 1024 / 8) ; // 3 GB 
													   		// = enough for the entire K with 20,000 data 
	if ( cacheSize < X.length * X.length ) 
		return undefined;	// cannot work in this setting 
			
	this.kerneltype = kerneltype;
	this.kernelpar = kernelpar;
	this.X = matrixCopy(X);
	
	this.cachesize = cacheSize; // in number of Kij entries
	this.size = Math.min( Math.floor(cacheSize / X.length), X.length ); // in number of rows;
	this.rowlength = X.length;
	
	this.K = zeros(this.size, X.length );	// entire K
	this.rowindex = new Array(); // list of rows indexes: rowindex[i] = index of X_j whose row is store in K[i]
	this.LRU = new Array(); 	// list of last recently used rows (index in K) 
								// the last recently used is at the beginning of the list

	// workers[t] work on indexes from t*size/CPUs to (t+1)*size/CPUs workers[0] is main thread (this)
	this.CPUs = 4;
	this.workers = new Array( CPUs ); 
	for ( var t=1; t< CPUs; t++) {
		this.workers[t] = new Worker( "kernelworker.js" );
		// copy data and setup worker
		this.workers[t].postMessage( {kerneltype: kerneltype, kernelpar: kernelpar, X: X, cachesize = cacheSize, index: t} );
	}
}

kernelCacheParallel.prototype.get_row = function ( row ) {
	var j;
	var Krow;
	var i = this.rowindex.indexOf( row );
	
	if ( i >= 0 ) {
		// requested row already in cache
		Krow = this.K[i]; // for thread safe :  vectorCopy(K[i]); 
	}
	else {
		// Need to compute the new row
		
		// Find an index to store the row
		if (this.rowindex.length < this.size) {
			// There is space for this row, so append at the end
			i = this.rowindex.length;
			this.rowindex.push( row ) ; 			
		}
		else {
			// Need to erase the last recenty used:
			i = this.LRU[0]; 
			this.rowindex[i] = row;
		}
		
		// Compute kernel row
		for (j = 0; j < this.rowlength; j++) {
			Kij = kernel(this.X[row], this.X[j], this.kerneltype, this.kernelpar);
			if ( Math.abs(Kij) > 1e-8)
				this.K[i][j] = Kij;
		}
		Krow = this.K[i];
	}
	
	// Update LRU
	var alreadyInLRU = this.LRU.indexOf(i);
	if ( alreadyInLRU >= 0)
		this.LRU.splice(alreadyInLRU, 1);
	this.LRU.push(i); 
	
	return Krow;	
}
*/
