//////////////////////////////////////////////////
/////		Generic class for Regressions
////		(implements least squares by default)
///////////////////////////////////////////////////
/**
 * @constructor
 */
function Regression (algorithm, params ) {
	
	if ( typeof(algorithm) == "undefined" ) {
		var algorithm = AutoReg;
	}
	else if (typeof(algorithm) == "string") 
		algorithm = eval(algorithm);

	this.type = "Regression:" + algorithm.name;
	
	this.algorithm = algorithm.name;
	this.userParameters = params;

	// Functions that depend on the algorithm:
	this.construct = algorithm.prototype.construct; 

	this.train = algorithm.prototype.train; 
	if (  algorithm.prototype.predict ) 
		this.predict = algorithm.prototype.predict; // otherwise use default function for linear model
	
	if (  algorithm.prototype.tune ) 
		this.tune = algorithm.prototype.tune; // otherwise use default function that does not tune but simply do cv for now...
	
	if (  algorithm.prototype.path ) 
		this.path = algorithm.prototype.path; 
	
	
	// Initialization depending on algorithm
	this.construct(params);
}

Regression.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:

	this.affine = true;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			

}

Regression.prototype.tune = function ( X, y, Xv, yv ) {
	// Main function for tuning an algorithm on a given data set

	/*
		1) apply cross validation (or test on (Xv,yv) ) to estimate the performance of all sets of parameters
			in this.parameterGrid
			
		2) pick the best set of parameters and train the final model on all data
			store this model in this.* 
	*/

	var validationSet = ( typeof(Xv) != "undefined" && typeof(yv) != "undefined" );
	
	var n = 0;
	var parnames = new Array();

	if (typeof(this.parameterGrid) != "undefined" ) {
		for ( var p in this.parameterGrid ) {
			parnames[n] = p;
			n++;
		}
	}
	var validationErrors;
	var minValidError = Infinity;
	var bestfit;

	if ( n == 0 ) {
		// no hyperparater to tune, so just train and test
		if ( validationSet ) {
			this.train(X,y);
			var stats = this.test(Xv,yv, true);
		}
		else 
			var stats = this.cv(X,y);
		minValidError = stats.mse;
		bestfit = stats.fit;
	}
	else if( n == 1 ) {
		// Just one hyperparameter
		var validationErrors = zeros(this.parameterGrid[parnames[0]].length);
		var bestpar; 		

		for ( var p =0; p <  this.parameterGrid[parnames[0]].length; p++ ) {
			this[parnames[0]] = this.parameterGrid[parnames[0]][p];
			if ( validationSet ) {
				// use validation set
				this.train(X,y);
				var stats = this.test(Xv,yv, true);
			}
			else {
				// do cross validation
				var stats = this.cv(X,y);
			}
			validationErrors[p] = stats.mse;
			if ( stats.mse < minValidError ) {
				minValidError = stats.mse;
				bestfit = stats.fit;
				bestpar = this[parnames[0]];
			}
			notifyProgress( p / this.parameterGrid[parnames[0]].length ) ;
		}
		
		// retrain with all data
		this[parnames[0]] = bestpar; 
		if ( validationSet ) 
			this.train( mat([X,Xv], true),reshape( mat([y,yv],true), y.length+yv.length, 1));
		else
			this.train(X,y);
	}
	else if ( n == 2 ) {
		// 2 hyperparameters
		validationErrors = zeros(this.parameterGrid[parnames[0]].length, this.parameterGrid[parnames[1]].length);
		var bestpar = new Array(2); 		
		
		var iter = 0;
		for ( var p0 =0; p0 <  this.parameterGrid[parnames[0]].length; p0++ ) {
			this[parnames[0]] = this.parameterGrid[parnames[0]][p0];

			for ( var p1 =0; p1 <  this.parameterGrid[parnames[1]].length; p1++ ) {
				this[parnames[1]] = this.parameterGrid[parnames[1]][p1];
			
				if ( validationSet ) {
					// use validation set
					this.train(X,y);
					var stats = this.test(Xv,yv, true);
				}
				else {
					// do cross validation
					var stats = this.cv(X,y);
				}
				validationErrors.val[p0*this.parameterGrid[parnames[1]].length + p1] = stats.mse;
				if ( stats.mse < minValidError ) {
					minValidError = stats.mse;
					bestfit = stats.fit;
					bestpar[0] = this[parnames[0]];
					bestpar[1] = this[parnames[1]];
				}
				iter++;
				notifyProgress( iter / (this.parameterGrid[parnames[0]].length *this.parameterGrid[parnames[1]].length) ) ;
			}
		}
		
		// retrain with all data
		this[parnames[0]] = bestpar[0]; 
		this[parnames[1]] = bestpar[1]; 		
		if( validationSet )
			this.train( mat([X,Xv], true),reshape( mat([y,yv],true), y.length+yv.length, 1));
		else
			this.train(X,y);
	}
	else {
		// too many hyperparameters... 
		error("Too many hyperparameters to tune.");
	}	
	notifyProgress( 1 ) ;
	return {error: minValidError, fit: bestfit, validationErrors: validationErrors};
}

Regression.prototype.train = function (X, y) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error.		

	return this;

}

Regression.prototype.predict = function (X) {
	// Prediction function (default for linear model)
	
	var y = mul( X, this.w); 
	
	if ( this.affine  && this.b) 
		y = add(y, this.b);
	
	return y;
}

Regression.prototype.test = function (X, y, compute_fit) {
	// Test function: return the mean squared error (use this.predict to get the predictions)
	var prediction = this.predict( X ) ;

	var i;
	var mse = 0;
	var errors = 0;
	if ( type(y) == "vector") {
		for ( i=0; i < y.length; i++) {
			errors += this.squaredloss( prediction[i] , y[i] );
		}
		mse = errors/y.length; // javascript uses floats for integers, so this should be ok.
	}
	else {
		mse = this.squaredloss( prediction  , y  );
	}
	
	
	if ( typeof(compute_fit) != "undefined" && compute_fit) 
		return { mse: mse, fit: 100*(1 - norm(sub(y,prediction))/norm(sub(y, mean(y)))) };
	else
		return mse;

}

Regression.prototype.squaredloss = function ( y, yhat ) {
	var e = y - yhat;
	return e*e;
}

Regression.prototype.cv = function ( X, labels, nFolds) {
	// Cross validation
	if ( typeof(nFolds) == "undefined" )
		var nFolds = 5;
	
	const N = labels.length;
	const foldsize = Math.floor(N / nFolds);
	
	// Random permutation of the data set
	var perm = randperm(N);
	
	// Start CV
	var errors = zeros (nFolds);
	var fits = zeros (nFolds);	
	
	var Xtr, Ytr, Xte, Yte;
	var i;
	var fold;
	var tmp;
	for ( fold = 0; fold < nFolds - 1; fold++) {
		
		Xte = get(X, get(perm, range(fold * foldsize, (fold+1)*foldsize)), []);
		Yte = get(labels, get(perm, range(fold * foldsize, (fold+1)*foldsize)) );
		
		var tridx = new Array();
		for (i=0; i < fold*foldsize; i++)
			tridx.push(perm[i]);
		for (i=(fold+1)*foldsize; i < N; i++)
			tridx.push(perm[i]);
		
		Xtr =  get(X, tridx, []);
		Ytr = get(labels, tridx);

		this.train(Xtr, Ytr);
		tmp = this.test(Xte,Yte, true);	
		errors[fold] = tmp.mse; 
		fits[fold] = tmp.fit;
	}
	// last fold:
	this.train( get(X, get(perm, range(0, fold * foldsize)), []), get(labels, get(perm, range(0, fold * foldsize ) ) ) );		  
	tmp = this.test(get(X, get(perm, range(fold * foldsize, N)), []), get(labels, get(perm, range(fold * foldsize, N)) ), true);		  	
	errors[fold] = tmp.mse; 
	fits[fold] = tmp.fit;
	
	// Retrain on all data
	this.train(X, labels);

	// Return kFold error
	return {mse: mean(errors), fit: mean(fits)};	
}

Regression.prototype.info = function () {
	// Print information about the model
	
	var str = "{<br>";
	var i;
	var Functions = new Array();
	for ( i in this) {
		switch ( type( this[i] ) ) {
			case "string":
			case "boolean":
			case "number":
				str += i + ": " + this[i] + "<br>";
				break;
			case "vector":
				str += i + ": " + printVector(this[i]) + "<br>";
				break;
			case "matrix":
				str += i + ": matrix of size " + this[i].m + "-by-" + this[i].n + "<br>";
				break;
			case "function": 
				Functions.push( i );
				break;
			default:
				str += i + ": " + typeof(this[i]) + "<br>";
				break;			
		}
	}
	str += "<i>Functions: " + Functions.join(", ") + "</i><br>";
	str += "}";
	return str;
}
/* Utility function 
	return true if x contains a single data instance
			false otherwise
*/
Regression.prototype.single_x = function ( x ) {
	var tx = type(x);
	return (tx == "number" || ( this.dim_input > 1 && (tx == "vector" || tx == "spvector" ) ) ) ;
}

//////////////////////////////////////
//// AutoReg: Automatic selection of best algo and parameters
////////////////////////////////////
function AutoReg ( params) {
	var that = new Regression ( AutoReg, params);
	return that;
}
AutoReg.prototype.construct = function ( params ) {
	// Read params and create the required fields for a specific algorithm
	
	// Default parameters:

	this.linearMethods = ["LeastSquares", "LeastAbsolute", "RidgeRegression", "LASSO"];
	this.linearParams = [undefined, undefined, undefined, {lambda: 1}]; // only those that are not tuned
	
	this.nonlinearMethods = ["KNNreg", "KernelRidgeRegression", "SVR", "MLPreg"];
	this.nonlinearParams = [undefined, {kernel: "rbf"}, {kernel: "rbf", epsilon: 0.1}, undefined]; // only those that are not tuned

	this.excludes = []; 

	this.linear = "auto"; // possible values: "yes", "no", true, false, "auto"

	this.excludes1D = ["LASSO"]; // methods that do not work with size(X,2) = 1
		
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			

}
AutoReg.prototype.train = function ( X, y ) {

	var dim = size(X,2);
	
	var bestlinearmodel;
	var bestnonlinearmodel;
	var bestmse = Infinity;
	var bestnlmse = Infinity;
	var minmse = 1e-8;
	
	var m;
	if ( this.linear != "no" && this.linear != false ) {
		// Try linear methods
		m =0;	
		while ( m < this.linearMethods.length && bestmse > minmse ) {
			if ( this.excludes.indexOf( this.linearMethods[m] ) < 0 && (dim != 1 || this.excludes1D.indexOf( this.linearMethods[m] ) < 0) ) {
				console.log("Testing " + this.linearMethods[m] );
				var model = new Regression(this.linearMethods[m], this.linearParams[m]);
				var stats = model.cv(X,y);
				if ( stats.mse < bestmse ) {
					bestmse = stats.mse; 
					bestlinearmodel = m;
				}
			}
			m++;
		}
		console.log("Best linear method is " + this.linearMethods[bestlinearmodel] + " ( mse = " + bestmse + ")");
	}
	
	if ( this.linear != "yes" && this.linear != true ) {
		// Try nonlinear methods
		m = 0;
		while ( m < this.nonlinearMethods.length && bestnlmse > minmse ) {
			if ( this.excludes.indexOf( this.nonlinearMethods[m] ) < 0 && (dim != 1 || this.excludes1D.indexOf( this.nonlinearMethods[m] ) < 0) ) {
				console.log("Testing " + this.nonlinearMethods[m] );
				var model = new Regression(this.nonlinearMethods[m], this.nonlinearParams[m]);
				var stats = model.cv(X,y);
				if ( stats.mse < bestnlmse ) {
					bestnlmse = stats.mse; 
					bestnonlinearmodel = m;
				}
			}
			m++;
		}
		console.log("Best nonlinear method is " + this.nonlinearMethods[bestnonlinearmodel] + " ( mse = " + bestnlmse + ")");
	}
	
	// Retrain best model on all data and store it in this.model
	if ( bestmse < bestnlmse ) {
		console.log("Best method is " + this.linearMethods[bestlinearmodel] + " (linear)"); 
		this.model = new Regression(this.linearMethods[bestlinearmodel], this.linearParams[bestlinearmodel]);
	}
	else {
		console.log("Best method is " + this.nonlinearMethods[bestnonlinearmodel] + " (nonlinear)");
		this.model = new Regression(this.nonlinearMethods[bestnonlinearmodel], this.nonlinearParams[bestnonlinearmodel]);		
	}
			
	this.model.train(X,y);
	return this;
}

AutoReg.prototype.tune = function ( X, y, Xv, yv ) {
	
	this.train(X,y);
	if  ( typeof(Xv) != "undefined" && typeof(yv) != "undefined" ) 
		var stats = this.test(Xv,yv, true);
	else 
		var stats = this.model.cv(X,y);

	return {error: stats.mse, fit: stats.fit};	
}

AutoReg.prototype.predict = function ( X ) {
	if ( this.model ) 
		return this.model.predict(X);
	else
		return undefined;
}



//////////////////////////////////////
////// LeastSquares
////////////////////////////////////
function LeastSquares ( params) {
	var that = new Regression ( LeastSquares, params);
	return that;
}
LeastSquares.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:

	this.affine = true;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			

}
LeastSquares.prototype.train = function (X, y) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error.
		
	var Xreg;
	if ( this.affine) 
		Xreg = mat([X, ones(X.length)]);
	else
		Xreg = X;
	
	// w = (X'X)^-1 X' y (or QR solution if rank-defficient)
	var w = solve( Xreg, y);

	if ( this.affine ) {
		this.w = get(w, range(w.length-1));
		this.b = w[w.length-1];	 
	}
	else {
		this.w = w;
	}

	// Return training error:
	return this.test(X, y);

}


//////////////////////////////
/// LeastAbsolute: Least absolute error 
/////////////////////////////
function LeastAbsolute ( params) {
	var that = new Regression ( LeastAbsolute, params);
	return that;
}
LeastAbsolute.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:
	
	this.affine = true;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			
}

LeastAbsolute.prototype.train = function (X, y) {
	
	var Xreg;
	if ( this.affine ) 
		Xreg = mat([ X, ones(X.length) ]);
	else
		Xreg = X;
		
	var N = size(Xreg,1);
	var d = size(Xreg,2);

	var A = zeros(2*N,d + N);
	set ( A, range(N), range(d), Xreg);
	set ( A, range(N), range(d, N+d), minus(eye(N)) );
	set ( A, range(N, 2*N), range(d), minus(Xreg));
	set ( A, range(N, 2*N), range(d, N+d), minus(eye(N)) );

	var cost = zeros(d+N);
	set ( cost, range(d, d+N), 1);

	var b = zeros(2*N);
	set(b, range(N),y);
	set(b, range(N,2*N), minus(y));	
	
	var lb = zeros(d+N);
	set(lb, range(d), -Infinity);

	var sol = lp(cost, A, b, [], [], lb);
	
	if ( this.affine ) {
		this.w = get(sol,range(d-1));
		this.b = sol[d-1];
	}
	else
		this.w = get(sol, range(d));

	var e = getSubVector (sol,range(d,d+N));
	this.absoluteError = sumVector(e);
	
	// return mse
	return dot(e, e); 	
}

//////////////////////////////////////////////////
/////	K-nearest neighbors
///////////////////////////////////////////////////
function KNNreg ( params ) {
	var that = new Regression ( KNNreg, params);	
	return that;
}
KNNreg.prototype.construct = function (params) {

	// Default parameters:
	this.K = 5;	
	
	// Set parameters:
	if ( params) {
		if ( typeof(params) == "number") {
			this.K = params;
		}
		else {
			var i;	
			for (i in params)
				this[i] = params[i]; 
		}
	}		

	// Parameter grid for automatic tuning:
	this.parameterGrid = { "K" : [1,3,5,7,10,15] };		
}

KNNreg.prototype.train = function ( X, y ) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error rate.
	
	this.X = matrixCopy(X);
	this.y = vectorCopy(y); 

	// Return training error rate:
	// return this.test(X, y);
	return this;
}

KNNreg.prototype.predict = function ( x ) {
   	var N = this.X.length; 
   	if (this.K >  N) {
   		this.K = N;   		
   	}
   	const K = this.K;
   	
   	if ( K == 0 ) {
   		return undefined;
   	}

	const tx = type(x);
	const tX = type(this.X);
	if ( tx == "vector" && tX == "matrix") {
		// Single prediction of a feature vector 
		var Xtest = new Matrix(1, x.length, x);
	}
	else if (tx == "number" && tX == "vector" ) {
		var Xtest = [x];
	}
	else if ( tx == "matrix"||  ( tx == "vector" && tX == "vector")  ) { 
		// Multiple predictions for a test set
		var Xtest = x;
	}	
	else
		return "undefined";

	var labels = zeros(Xtest.length);
	var i;
	var dim = size(Xtest,2);
	for ( i=0; i < Xtest.length; i++) {
		
		if ( dim > 1 )
			var nn = knnsearchND(K, Xtest.row(i), this.X ); // knnsearch is defined in Classifier.js
		else
			var nn = knnsearch1D(K, Xtest[i], this.X ); // knnsearch is defined in Classifier.js

		labels[i] = mean(get(this.y, nn.indexes) );
		
	}
	
	if ( labels.length > 1 ) 
		return labels;
	else
		return labels[0];
}

//////////////////////////////////////
////// Ridge regression
////////////////////////////////////
function RidgeRegression ( params) {
	var that = new Regression ( RidgeRegression, params);
	return that;
}
RidgeRegression.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:
	
	this.lambda = 1;
	this.affine = true;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			
		
	// Parameter grid for automatic tuning:
	this.parameterGrid = { "lambda" : [0.01, 0.1, 1, 5, 10] };	
}

RidgeRegression.prototype.train = function (X, y) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error.
	
	var res = ridgeregression( X, y , this.lambda, this.affine);

	if ( this.affine ) {
		this.w = get(res, range(res.length-1));
		this.b = res[res.length-1];	 
	}
	else {
		this.w = res;
	}
	
	// Return training error:
	return this.test(X, y);

}

function ridgeregression( X, y , lambda, affine) {
	// simple function to compute parameter vector of ridge regression
	// (can be used on its own).
	
	if ( typeof(affine ) == "undefined")
		var affine = true;
	if( typeof(lambda) == "undefined")
		var lambda = 1;
		
	var Xreg;
	if ( affine) 
		Xreg = mat([X, ones(X.length)]);
	else
		Xreg = X;
	
	// A = X' X + lambda I
	if ( type(Xreg) == "vector" ) 
		var w = dot(X,y) / ( dot(Xreg,Xreg) + lambda); 
	else {
		var Xt = transposeMatrix(Xreg);	
		var A = mulMatrixMatrix(Xt, Xreg);
		var n = Xreg.n;
		for ( var i=0; i < Xreg.length; i++)
			A.val[i*(n+1)] += lambda;
	
		// solve Aw = X' y
		var w = solve( A, mulMatrixVector(Xt, y) );
	}
	
	return w;
}


//////////////////////////////////////
////// Kernel ridge regression
////////////////////////////////////
function KernelRidgeRegression ( params) {
	var that = new Regression ( KernelRidgeRegression, params);
	return that;
}
KernelRidgeRegression.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:
	
	this.lambda = 1;
	this.affine = false;
	this.kernel = "rbf";
	this.kernelpar = kernel_default_parameter("rbf");
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}	
	
	
	// Parameter grid for automatic tuning:
	switch (this.kernel) {
		case "gaussian":
		case "Gaussian":
		case "RBF":
		case "rbf": 
			// use multiples powers of 1/sqrt(2) for sigma => efficient kernel updates by squaring
			this.parameterGrid = { "kernelpar": pow(1/Math.sqrt(2), range(0,10)), "lambda" : [ 0.01, 0.1, 1, 5, 10] };
			break;
			
		case "poly":
			this.parameterGrid = { "kernelpar": [3,5,7,9] , "lambda" : [0.1, 1, 5, 10]  };
			break;
		case "polyh":
			this.parameterGrid = { "kernelpar": [3,5,7,9] , "lambda" : [0.1, 1, 5, 10] };
			break;
		default:
			this.parameterGrid = undefined; 
			break;
	}		
}

KernelRidgeRegression.prototype.train = function (X, y) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error.
	
	//this.K = kernelMatrix(X, this.kernel, this.kernelpar); // store K for further tuning use
		
	// alpha = (K+lambda I)^-1 y
	
	//var Kreg = add(this.K, mul(this.lambda, eye(this.K.length)));
	var Kreg = kernelMatrix(X, this.kernel, this.kernelpar);
	var kii = 0;
	for ( var i=0; i < Kreg.length; i++) {
		Kreg.val[kii] += this.lambda; 
		kii += Kreg.length + 1;
	}

	if ( y.length <= 500 )
		this.alpha = solve(Kreg, y);		// standard QR solver
	else {
		if ( norm0(Kreg) < 0.4 * y.length * y.length )
			this.alpha = spsolvecg(sparse(Kreg), y);	// sparse conjugate gradient solver
		else
			this.alpha = solvecg(Kreg, y);		// dense conjugate gradient solver
	}
	
	// Set kernel function
	this.kernelFunc = kernelFunction ( this.kernel, this.kernelpar);
	// and input dim
	this.dim_input = size(X, 2);
	if ( this.dim_input == 1 )
		this.X = mat([X]); // make it a 1-column matrix
	else 
		this.X = matrixCopy(X);
		
	/*	
	// compute training error:
	var yhat = mulMatrixVector(this.K, this.alpha);
	var error = subVectors ( yhat, y);
	
	return dot(error,error) / y.length;
	*/
	return this;
}

KernelRidgeRegression.prototype.tune = function (X, y, Xv, yv) {
	// Use fast kernel matrix updates to tune kernel parameter
	// For cv: loop over kernel parameter for each fold to update K efficiently
	
	
	// Set the kernelpar range with the dimension
	if ( this.kernel == "rbf" && size(X,2) > 1 ) {
		var saveKpGrid = zeros(this.parameterGrid.kernelpar.length);
		for ( var kp = 0; kp < this.parameterGrid.kernelpar.length ; kp ++) {
			saveKpGrid[kp] = this.parameterGrid.kernelpar[kp];
			this.parameterGrid.kernelpar[kp] *= Math.sqrt( X.n ); 
		}
	}
	
	this.dim_input = size(X,2);
	const tX = type(X);
		
	var K;
	var spK;
	var sparseK = true; // is K sparse?
	
	var addDiag = function ( value ) {
		// add scalar value on diagonal of K
		var kii = 0;
		for ( var i=0; i < K.length; i++) {
			K.val[kii] += value; 
			kii += K.length + 1;
		}
	}
	var spaddDiag = function ( value ) {
		// add scalar value on diagonal of sparse K
		for ( var i=0; i < spK.length; i++) {
			var j = spK.rows[i]; 
			var e = spK.rows[i+1]; 			
			while ( j < e && spK.cols[j] != i )
				j++;
			if ( j < e ) 
				spK.val[j] += value; 
			else {
				// error: this only works if no zero lies on the diagonal of K
				sparseK = false; 
				addDiag(value); 
				return;
			}
		}
	}

	if ( arguments.length == 4 ) {
		// validation set (Xv, yv)
		if ( tX == "vector" )
			this.X = mat([X]);
		else 
			this.X = X;
			
		// grid of ( kernelpar, lambda) values
		var validationErrors = zeros(this.parameterGrid.kernelpar.length, this.parameterGrid.C.length);
		var minValidError = Infinity;
		
		var bestkernelpar;
		var bestlambda;
		
		K = kernelMatrix( X , this.kernel, this.parameterGrid.kernelpar[0] ); 

		// Test all values of kernel par
		for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
			this.kernelpar = this.parameterGrid.kernelpar[kp];
			if ( kp > 0 ) {
				// Fast update of kernel matrix
				K = kernelMatrixUpdate( K,  this.kernel, this.kernelpar, this.parameterGrid.kernelpar[kp-1]  );
			}
			sparseK = (norm0(K) < 0.4 * y.length * y.length );
			if ( sparseK ) 
				spK = sparse(K);
			
			// Test all values of lambda for the same kernel par
			for ( var c = 0; c < this.parameterGrid.lambda.length; c++) {								
				this.lambda = this.parameterGrid.lambda[c];
				
				// K = K + lambda I
				if ( sparseK ) {
					if ( c == 0 ) 
						spaddDiag(this.lambda);
					else
						spaddDiag(this.lambda - this.parameterGrid.lambda[c-1] );
				}
				else {
					if ( c == 0 ) 
						addDiag(this.lambda);
					else 
						addDiag(this.lambda - this.parameterGrid.lambda[c-1] ); 
				}

				// Train model
				if ( y.length <= 500 )
					this.alpha = solve(K, y);		// standard QR solver
				else {
					if ( sparseK )
						this.alpha = spsolvecg(spK, y);	// sparse conjugate gradient solver
					else
						this.alpha = solvecg(K, y);		// dense conjugate gradient solver
				}
				
				validationErrors.set(kp,c, this.test(Xv,yv) );
				if ( validationErrors.get(kp,c) < minValidError ) {
					minValidError = validationErrors.get(kp,c);
					bestkernelpar = this.kernelpar;
					bestlambda = this.lambda;
				}
				
				// Recover original K = K - lambda I  for subsequent kernel matrix update
				if ( !sparseK && kp < this.parameterGrid.kernelpar.length - 1 ) 
					addDiag( -this.lambda );
			}
		}
		this.kernelpar = bestkernelpar;
		this.lambda = bestlambda;
		this.train(mat([X,Xv],true), mat([y,yv], true) ); // retrain with best values and all data
	}
	else {
		
		// 5-fold Cross validation
		const nFolds = 5;
	
		const N = y.length;
		const foldsize = Math.floor(N / nFolds);
	
		// Random permutation of the data set
		var perm = randperm(N);
	
		// Start CV
		var validationErrors = zeros(this.parameterGrid.kernelpar.length,this.parameterGrid.lambda.length);
		
	
		var Xtr, Ytr, Xte, Yte;
		var i;
		var fold;
		for ( fold = 0; fold < nFolds - 1; fold++) {
			console.log("fold " + fold);
			Xte = get(X, get(perm, range(fold * foldsize, (fold+1)*foldsize)), []);
			Yte = get(y, get(perm, range(fold * foldsize, (fold+1)*foldsize)) );
		
			var tridx = new Array();
			for (i=0; i < fold*foldsize; i++)
				tridx.push(perm[i]);
			for (i=(fold+1)*foldsize; i < N; i++)
				tridx.push(perm[i]);
		
			Xtr =  get(X, tridx, []);
			Ytr = get(y, tridx);

			if ( tX == "vector" )
				this.X = mat([Xtr]);
			else 
				this.X = Xtr;
		
			
			// grid of ( kernelpar, lambda) values
			
			K = kernelMatrix( Xtr , this.kernel, this.parameterGrid.kernelpar[0] ); 

			// Test all values of kernel par
			for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
				this.kernelpar = this.parameterGrid.kernelpar[kp];
				if ( kp > 0 ) {
					// Fast update of kernel matrix
					K = kernelMatrixUpdate( K,  this.kernel, this.kernelpar, this.parameterGrid.kernelpar[kp-1]  );
				}
				this.kernelFunc = kernelFunction (this.kernel, this.kernelpar);
				
				var sparseK =  (norm0(K) < 0.4 * K.length * K.length );
				if ( sparseK ) 
					spK = sparse(K);
			
				// Test all values of lambda for the same kernel par
				for ( var c = 0; c < this.parameterGrid.lambda.length; c++) {								
					this.lambda = this.parameterGrid.lambda[c];
				
					// K = K + lambda I
					if ( sparseK ) {
						if ( c == 0 ) 
							spaddDiag(this.lambda);
						else 
							spaddDiag(this.lambda - this.parameterGrid.lambda[c-1] ); 
					}
					else {
						if ( c == 0 ) 
							addDiag(this.lambda);
						else 
							addDiag(this.lambda - this.parameterGrid.lambda[c-1] ); 
					}

					// Train model
					if ( Ytr.length <= 500 )
						this.alpha = solve(K, Ytr);		// standard QR solver
					else {
						if ( sparseK )
							this.alpha = spsolvecg(spK, Ytr);	// sparse conjugate gradient solver
						else
							this.alpha = solvecg(K, Ytr);		// dense conjugate gradient solver
					}
				
					validationErrors.val[kp * this.parameterGrid.lambda.length + c] += this.test(Xte,Yte) ;
					
					// Recover original K = K - lambda I  for subsequent kernel matrix update
					if ( !sparseK && kp < this.parameterGrid.kernelpar.length - 1 ) 
						addDiag( -this.lambda );
				}
			}
		}
		// last fold:
		console.log("fold " + fold);		
		Xtr = get(X, get(perm, range(0, fold * foldsize)), []);
		Ytr = get(y, get(perm, range(0, fold * foldsize ) ) ); 
		Xte = get(X, get(perm, range(fold * foldsize, N)), []);
		Yte = get(y, get(perm, range(fold * foldsize, N)) );
		
		if ( tX == "vector" )
			this.X = mat([Xtr]);
		else 
			this.X = Xtr;
		
		
		// grid of ( kernelpar, lambda) values
		
		K = kernelMatrix( Xtr , this.kernel, this.parameterGrid.kernelpar[0] ); 

		// Test all values of kernel par
		for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
			this.kernelpar = this.parameterGrid.kernelpar[kp];
			if ( kp > 0 ) {
				// Fast update of kernel matrix
				K = kernelMatrixUpdate( K,  this.kernel, this.kernelpar, this.parameterGrid.kernelpar[kp-1]  );
			}
			this.kernelFunc = kernelFunction (this.kernel, this.kernelpar);
						
			var sparseK =  (norm0(K) < 0.4 * K.length * K.length );
			if ( sparseK ) 
				spK = sparse(K);
		
			// Test all values of lambda for the same kernel par
			for ( var c = 0; c < this.parameterGrid.lambda.length; c++) {								
				this.lambda = this.parameterGrid.lambda[c];
			
				// K = K + lambda I
				if ( sparseK ) {
					if ( c == 0 ) 
						spaddDiag(this.lambda);
					else 
						spaddDiag(this.lambda - this.parameterGrid.lambda[c-1] ); 
				}
				else {
					if ( c == 0 ) 
						addDiag(this.lambda);
					else 
						addDiag(this.lambda - this.parameterGrid.lambda[c-1] ); 
				}

				// Train model
				if ( Ytr.length <= 500 )
					this.alpha = solve(K, Ytr);		// standard QR solver
				else {
					if ( sparseK )
						this.alpha = spsolvecg(spK, Ytr);	// sparse conjugate gradient solver
					else
						this.alpha = solvecg(K, Ytr);		// dense conjugate gradient solver
				}
			
				validationErrors.val[kp * this.parameterGrid.lambda.length + c] += this.test(Xte,Yte) ;
				
				// Recover original K = K - lambda I  for subsequent kernel matrix update
				if ( !sparseK && kp < this.parameterGrid.kernelpar.length - 1 ) 
					addDiag( -this.lambda );
			}
		}		
	
		// Compute Kfold errors and find best parameters 
		var minValidError = Infinity;
		var bestlambda;
		var bestkernelpar;
		
		// grid of ( kernelpar, lambda) values
		for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
			for ( var c = 0; c < this.parameterGrid.lambda.length; c++) {
				validationErrors.val[kp * this.parameterGrid.lambda.length + c] /= nFolds;				
				if(validationErrors.val[kp * this.parameterGrid.lambda.length + c] < minValidError ) {
					minValidError = validationErrors.val[kp * this.parameterGrid.lambda.length + c]; 
					bestlambda = this.parameterGrid.lambda[c];
					bestkernelpar = this.parameterGrid.kernelpar[kp];
				}
			}
		}
		this.lambda = bestlambda;	
		this.kernelpar = bestkernelpar;	
	
		// Retrain on all data
		this.train(X, y);		
	}
	
	this.validationError = minValidError; 
	return {error: minValidError, validationErrors: validationErrors}; 
}

KernelRidgeRegression.prototype.predict = function (X) {
	// Prediction function f(x) = sum_i alpha_i K(x_i, x)

	/*	This works, but we do not need to store K
	var K = kernelMatrix(this.X, this.kernel, this.kernelpar, X);
	var y = transpose(mul(transposeVector(this.alpha), K));	
	*/ 
	var i,j;
	
	if ( this.single_x( X ) ) {
		if ( this.dim_input == 1 ) 
			var xvector = [X];
		else
			var xvector = X;
		var y = 0;
		for ( j=0; j < this.alpha.length; j++ )
			y += this.alpha[j] * this.kernelFunc(this.X.row(j), xvector);		
	}
	else {
		var y = zeros(X.length);
		for ( j=0; j < this.alpha.length; j++ ) {
			var Xj = this.X.row(j); 
			var aj = this.alpha[j];
			for ( i=0; i < X.length; i++ ) {
				if ( this.dim_input == 1 )
					var xvector = [X[i]];
				else
					var xvector = X.row(i);
				y[i] += aj * this.kernelFunc(Xj, xvector);		
			}
		}
	}
	return y;
}

//////////////////////////////////////////////////
/////		support vector regression (SVR)
///////////////////////////////////////////////////
function SVR ( params) {
	var that = new Regression ( SVR, params);
	return that;
}
SVR.prototype.construct = function (params) {
	
	// Default parameters:
	
	this.kernel = "linear";
	this.kernelpar = undefined;
	
	this.C = 1;
	this.epsilon = 0.1;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}		

	// Parameter grid for automatic tuning:
	switch (this.kernel) {
		case "linear":
			this.parameterGrid = { "C" : [0.001, 0.01, 0.1, 1, 5, 10, 100] };
			break;
		case "gaussian":
		case "Gaussian":
		case "RBF":
		case "rbf": 
			this.parameterGrid = { "kernelpar": [0.1,0.2,0.5,1,2,5] , "C" : [0.001, 0.01, 0.1, 1, 5, 10, 100] };
			break;
			
		case "poly":
			this.parameterGrid = { "kernelpar": [3,5,7,9] , "C" : [0.001, 0.01, 0.1, 1, 5, 10, 100]  };
			break;
		case "polyh":
			this.parameterGrid = { "kernelpar": [3,5,7,9] , "C" : [0.001, 0.01, 0.1, 1, 5, 10, 100] };
			break;
		default:
			this.parameterGrid = undefined; 
			break;
	}
}

SVR.prototype.train = function (X, y) {
	// Training SVR with SMO
	
	// Prepare
	const C = this.C;
	const epsilon = this.epsilon; 

	/* use already computed kernelcache if any;
	var kc;
	if (typeof(this.kernelcache) == "undefined") {
		// create a new kernel cache if none present
		kc = new kernelCache( X , this.kernel, this.kernelpar ); 
	}
	else 
		kc = this.kernelcache;
	*/
	var kc = new kernelCache( X , this.kernel, this.kernelpar ); 
	
	var i;
	var j;
	const N = X.length;	// Number of data
	const m = 2*N;	// number of optimization variables
	
	// linear cost c = [epsilon.1 + y; epsilon.1 - y]
	var c = zeros(m) ;
	set(c, range(N), add(epsilon, y) );
	set(c, range(N, m), sub(epsilon, y));
	
	// Initialization
	var alpha = zeros(m); // alpha = [alpha, alpha^*] 
	var b = 0;
	var grad = vectorCopy(c);
	var yc = ones(m); 
	set(yc, range(N, m), -1);	// yc = [1...1, -1...-1] (plays same role as y for classif in SMO)

	// SMO algorithm
	var index_i;
	var index_j;
	var alpha_i;
	var alpha_j;
	var grad_i;	
	var grad_j;
	var Q_i = zeros(m);
	var Q_j = zeros(m); 
	var ki;
	var kj;
	var Ki;
	var Kj;
	
	const tolalpha = 0.001; // tolerance to detect margin SV
	
	const TOL = 0.001; // TOL on the convergence
	var iter = 0;
	do {
		// working set selection => {index_i, index_j }
		var gradmax = -Infinity;
		var gradmin = Infinity;
					
		for (i=0; i< m; i++) {
			alpha_i = alpha[i];
			grad_i = grad[i];
			if ( yc[i] == 1 && alpha_i < C *(1-tolalpha) && -grad_i > gradmax ) {
				index_i = i;
				gradmax = -grad_i; 
			}
			else if ( yc[i] == -1 && alpha_i > C * tolalpha  && grad_i > gradmax ) {
				index_i = i;
				gradmax = grad_i; 
			}
			
			if ( yc[i] == -1 && alpha_i < C *(1-tolalpha) && grad_i < gradmin ) {
				index_j = i;
				gradmin = grad_i;
			}
			else if ( yc[i] == 1 && alpha_i > C * tolalpha && -grad_i < gradmin ) {
				index_j = i;
				gradmin = -grad_i;
			}
			//console.log(i,index_i,index_j,alpha_i,grad_i, gradmin, gradmax,yc[i] == -1);
		}

		
		// Analytical solution
		i = index_i;
		j = index_j;
		if ( i < N )	
			ki = i;	// index of corresponding row in K
		else
			ki = i - N;
		if ( j < N )	
			kj = j;	// index of corresponding row in K
		else
			kj = j - N;

		//  Q = [[ K, -K ], [-K, K] ]: hessian of optimization wrt 2*N vars
		//  Q[i][j] = yc_i yc_j K_ij
		//set(Q_i, range(N), mul( yc[i] , kc.get_row( ki ) )); // ith row of Q : left part for yc_j = 1
		//set(Q_i, range(N,m), mul( -yc[i] , kc.get_row( ki ) )); // ith row of Q : right part for yc_j = -1
		//set(Q_j, range(N), mul( yc[j] , kc.get_row( kj ) )); // jth row of Q : left part 
		//set(Q_j, range(N,m), mul( -yc[j] , kc.get_row( kj ) )); // jth row of Q : right part

		Ki = kc.get_row( ki );
		if ( yc[i] > 0 ) {				
			Q_i.set(Ki);
			Q_i.set(minus(Ki), N); 
		}
		else {
			Q_i.set(minus(Ki));
			Q_i.set(Ki, N); 		
		}
		
		Kj = kc.get_row( kj );
		if ( yc[j] > 0 ) {		
			Q_j.set(Kj);
			Q_j.set(minus(Kj), N); 
		}
		else {
			Q_j.set(minus(Kj));
			Q_j.set(Kj, N); 		
		}
		
		alpha_i = alpha[i];
		alpha_j = alpha[j];
		grad_i = grad[i];
		grad_j = grad[j];

		// Update alpha and correct to remain in feasible set
		if ( yc[i] != yc[j] ) {
			var diff = alpha_i - alpha_j;
			var delta = -(grad_i + grad_j ) / ( Q_i[i] + Q_j[j] + 2 * Q_i[j] );
			alpha[j] = alpha_j + delta;
			alpha[i] = alpha_i + delta; 
			
			if( diff > 0 ) {
				if ( alpha[j] < 0 ) {
					alpha[j] = 0;	
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > 0 )
			{
				if(alpha[i] > C)
				{
					alpha[i] = C;
					alpha[j] = C - diff;
				}
			}
			else
			{
				if(alpha[j] > C)
				{
					alpha[j] = C;
					alpha[i] = C + diff;
				}
			}
		}
		else {
			var sum = alpha_i + alpha_j;
			var delta = (grad_i - grad_j) / ( Q_i[i] + Q_j[j] - 2 * Q_i[j] );
			alpha[i] = alpha_i - delta; 
			alpha[j] = alpha_j + delta;
		
			if(sum > C)
			{
				if(alpha[i] > C)
				{
					alpha[i] = C;
					alpha[j] = sum - C;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C)
			{
				if(alpha[j] > C)
				{
					alpha[j] = C;
					alpha[i] = sum - C;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// gradient = Q alpha + c 
		// ==>  grad += Q_i* d alpha_i + Q_j d alpha_j; Q_i = Q[i] (Q symmetric)		
		grad = add( grad, add ( mul( alpha[i] - alpha_i, Q_i ) , mul( alpha[j] - alpha_j, Q_j ))); 
		
		
		iter++;
	} while ( iter < 100000 && gradmax - gradmin > TOL ) ; 
	

	// Compute b=(r2 - r1) / 2, r1 = sum_(0<alpha_i<C && yci==1) grad_i / #(0<alpha_i<C && yci==1), r2=same with yci==-1
	var r1 = 0;
	var r2 = 0;
	var Nr1 = 0;
	var Nr2 = 0;
	var gradmax1 = -Infinity;
	var gradmin1 = Infinity;
	var gradmax2 = -Infinity;
	var gradmin2 = Infinity;
	for(i=0;i < m ;i++) {
		if ( alpha[i] > tolalpha*C && alpha[i] < (1.0 - tolalpha) * C ) {
			if ( yc[i] == 1 ) {
				r1 += grad[i];
				Nr1++;
			}
			else {
				r2 += grad[i];
				Nr2++;
			}
		}
		
		else if ( alpha[i] >= (1.0 - tolalpha) * C ) {
			if ( yc[i] == 1 && grad[i] > gradmax1 )
				gradmax1 = grad[i];	
			if ( yc[i] == -1 && grad[i] > gradmax2 )
				gradmax2 = grad[i];					
		}			
		
		else if ( alpha[i] <= tolalpha * C ) {
			if ( yc[i] == 1 && grad[i] < gradmin1 )
				gradmin1 = grad[i];	
			if ( yc[i] == -1 && grad[i] < gradmin2 )
				gradmin2 = grad[i];					
		}			
	}
	if( Nr1 > 0 )
		r1 /= Nr1;
	else
		r1 = (gradmax1 + gradmin1) / 2;
	if( Nr2 > 0 )
		r2 /= Nr2;
	else
		r2 = (gradmax2 + gradmin2) / 2;

	b = -(r2 - r1) / 2; 
	

	
	/* Find support vectors	*/
	var nz = isGreater(alpha, 1e-6);

	alpha = entrywisemul(nz, alpha); // zeroing small alpha_i
	this.alpha = sub ( get(alpha,range(N,m)), get(alpha, range(N)) ); // alpha_final = -alpha + alpha^*
	this.SVindexes = find(this.alpha); // list of indexes of SVs
	this.SV = get(X,this.SVindexes, []) ;
		
	this.dim_input = 1;
	if ( type(X) == "matrix")
		this.dim_input = X.n; // set input dimension for checks during prediction
	
	// Compute w for linear models
	var w;
	if ( this.kernel == "linear" ) {
		w = transpose(mul( transpose( get (this.alpha, this.SVindexes) ), this.SV ));
	}
	
		
	if ( typeof(this.kernelcache) == "undefined")
		this.kernelcache = kc;
		

	this.b = b;
	this.w = w;				
	
	// Set kernel function
	if ( this.dim_input > 1 )
		this.kernelFunc = this.kernelcache.kernelFunc;
	else
		this.kernelFunc = kernelFunction(this.kernel, this.kernelpar, "number"); // for scalar input
		
	// and return training error rate:
	// return this.test(X, y); // XXX use kernel cache instead!! 
	return this;
}


SVR.prototype.predict = function ( x ) {

	var i;
	var j;
	var output;

	if ( this.kernel =="linear" && this.w)
		return add( mul(x, this.w) , this.b);
		
	if ( this.single_x(x) ) {	
		output = this.b;

		if ( this.dim_input > 1 ) {
			for ( j=0; j < this.SVindexes.length; j++) 
				output += this.alpha[this.SVindexes[j]] * this.kernelFunc( this.SV.row(j), x);
		}
		else {
			for ( j=0; j < this.SVindexes.length; j++) 
				output += this.alpha[this.SVindexes[j]] * this.kernelFunc( this.SV[j], x);
		}
		return output;
	}
	else if ( this.dim_input == 1) {
		output = zeros(x.length);
		for ( i=0; i < x.length; i++) {				
			output[i] = this.b;
			for ( j=0; j < this.SVindexes.length; j++) {
				output[i] += this.alpha[this.SVindexes[j]] * this.kernelFunc( this.SV[j], x[i]);
			}
		}
		return output;
	}	
	else {
		// Cache SVs
		var SVs = new Array(this.SVindexes.length);
		for ( j=0; j < this.SVindexes.length; j++) 
			SVs[j] = this.SV.row(j);
		
		output = zeros(x.length);
		for ( i=0; i < x.length; i++) {				
			output[i] = this.b;
			var xi = x.row(i);
			for ( j=0; j < this.SVindexes.length; j++) 
				output[i] += this.alpha[this.SVindexes[j]] * this.kernelFunc( SVs[j], xi);			
		}
		return output;
	}
}


////////////////////////////////////////////
///  LASSO 
//////////////////////////////////////////

function LASSO ( params) {
	var that = new Regression ( LASSO, params);
	return that;
}
LASSO.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:
	
	this.lambda = 1;
	this.affine = true;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			
}

LASSO.prototype.train = function (X, y, softthresholding) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error.

	var M = Infinity; // max absolute value of a parameter w and b
		
	var Xreg;
	if ( this.affine) 
		Xreg = mat([X, ones(X.length)]);
	else
		Xreg = X;
	

	
	var d = 1;
	if ( type(Xreg) == "matrix")
		d = Xreg.n;
	var dw = d;
	if ( this.affine )
		dw--;
		
	if ( typeof(softthresholding) =="undefined") {
		// Test orthonormality of columns 
		var orthonormality = norm ( sub(eye(dw) , mul(transpose(X), X) ) );
			
		if ( orthonormality < 1e-6 ) {
			console.log("LASSO: orthonormal columns detected, using soft-thresholding.");
			var softthresholding = true;
		}
		else {
			console.log("LASSO: orthonormalilty = " + orthonormality + " > 1e-6, solving QP...");
			var softthresholding = false;
		}
	}
	
	if ( softthresholding ) {
		// Apply soft-thresholding assuming orthonormal columns in Xreg

		var solLS = solve(Xreg, y); 
		var wLS = get ( solLS, range(dw) );

		var tmp = sub(abs(wLS) , this.lambda) ;

		this.w = entrywisemul( sign(wLS), entrywisemul(isGreater(tmp, 0) , tmp) );

		if ( this.affine ) {
			this.b = solLS[dw];	 
		}

		// Return training error:
		return this.test(X, y);
	}
	
	if ( dw == 1 )
		return "should do softthresholding";
	
	// Coordinate descent
	var b = randn();
	var bprev ;
	var w = zeros(dw);
	var wprev = zeros(dw);
	var residues = subVectors ( y, addScalarVector(b, mul(X, w))) ;
	var sumX2 = sum(entrywisemul(X,X),1).val;
	
	// Make Xj an Array of features 
	var Xt = transposeMatrix(X);
	var Xj = new Array(Xt.m); 
	for ( var j=0; j < dw; j++) {
		Xj[j] = Xt.row(j);
	}
	
	
	var iter = 0;
	do { 
		bprev = b;
		b += sumVector(residues) / y.length; 
		residues = add (residues , bprev - b);

		
		for ( var j=0; j < dw; j++) {
			wprev[j] = w[j];
			
			var dgdW = -dot( residues, Xj[j] );
			var updateneg = w[j] - (dgdW - this.lambda) / sumX2[j];
			var updatepos = w[j] - (dgdW + this.lambda) / sumX2[j];

			if ( updateneg < 0 ) 
				w[j] = updateneg; 
			else if (updatepos > 0 ) 
				w[j] = updatepos;
			else 
				w[j] = 0;
			if ( !isZero(w[j] - wprev[j])) 
				residues = addVectors(residues, mulScalarVector(wprev[j] - w[j], getCols(X,[j])) );	
		}
		
		iter++;
	} while ( iter < 1000 && norm(sub(w,wprev)) > y.length * 1e-6 ) ; 
	console.log("LASSO coordinate descent ended after " + iter + " iterations");
	
	this.w = w;
	this.b = b;
	
	return this;
	
}
LASSO.prototype.tune = function (X,y,Xv,yv) {
	// TODO use lars to compute the full path...
	
}


//////////////////////////////
// LARS
/////////////////////////////
function LARS ( params) {
	var that = new Regression ( LARS, params);
	return that;
}
LARS.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:
	
	this.method = "lars";
	this.n = undefined;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			

	this.affine = true;	// cannot be changed otherwise y cannot be centered
}

LARS.prototype.train = function (X, y) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error.
	if ( typeof(this.n) != "number" ) {
		console.log("LARS: using n=3 features by default");
		this.n = 3;
	}
	this.path = lars(X,y,this.method, this.n);
	this.w = get(this.path,range(X.n), this.n);
	this.b = this.path.val[X.n * this.path.n + this.n];
	this.support = find( isNotEqual(this.w, 0) );
	return this;
}
LARS.prototype.predict = function (X, y) {

	var y = add(mul( X, this.w), this.b); 
	// XXX should do a sparse multiplication depending on the support of w... 
		
	return y;
}
LARS.prototype.tune = function (X, y, Xv, yv) {
	// TODO: compute path for all folds of cross validation
	// and test with mul(Xtest, path)... 

}
LARS.prototype.path = function (X, y) {
	// TODO: compute path for all folds of cross validation
	// and test with mul(Xtest, path)... 
	return lars(X,y, this.method, this.n);
}
function lars (X,Y, method, n) {
	if ( type(X) != "matrix" )
		return "Need a matrix X with at least 2 columns to apply lars().";
	if ( arguments.length < 4 )
		var n = X.n;
	if ( arguments.length < 3 )
		var method = "lars";
	
		
	const N = X.length;
	const d = X.n;	
	
	// --- utilities ---
	//Function that updates cholesky factorization of X'X when adding column x to X
	var updateR = function (x, R, Xt, rank) {
			var xtx = dot(x,x); 
			const tR = typeof ( R ) ;
			if (tR == "undefined")
				return {R: Math.sqrt(xtx), rank: 1};
			else if ( tR == "number" ) {
				// Xt is a vector
				var Xtx = dot(Xt,x); 
				var r = Xtx / R;
				var rpp = xtx - r*r;
				var newR = zeros(2,2);
				newR.val[0] = R;
				newR.val[2] = r;
				if(rpp <= EPS) {
					rpp = EPS;
					var newrank = rank;
				}
				else {
					rpp = Math.sqrt(rpp);
					var newrank = rank+1;
				}

				newR.val[3] = rpp;		
			}
			else {
				/* X and R are matrices : we have RR' = X'X 
					 we want [R 0; r', rpp][R 0;r', rpp]' = [X x]'[X x] = [ X'X X'x; x'X x'x]
					 last column of matrix equality gives
					    Rr = X'x   
					and r'r + rpp^2 = x'x => rpp = sqrt(x'x - r'r)
				*/
				var Xtx = mulMatrixVector( Xt, x);
				var r = forwardsubstitution( R, Xtx) ;
				var rpp = xtx - dot(r,r);
				const sizeR = r.length;
				var newR = zeros(sizeR+1,sizeR+1); 
				for ( var i=0; i < sizeR; i++)
					for ( var j=0; j <= i; j++)
						newR.val[i*(sizeR+1) + j] = R.val[i*sizeR+j];
				for ( var j=0; j < sizeR; j++)
					newR.val[sizeR*(sizeR+1) + j] = r[j]; 

				if(rpp <= EPS) {
					rpp = EPS;
					var newrank = rank;
				}
				else {
					rpp = Math.sqrt(rpp);
					var newrank = rank+1;
				}
				newR.val[sizeR*(sizeR+1) + sizeR] = rpp; 

			}
			
			return {R: newR, rank: newrank};
		};
	// Function that downdates cholesky factorization of X'X when removing column j from X (for lasso)
	var downdateR = function (j, R, rank) {			
			var idx = range(R.m);
			idx.splice(j,1);
			var newR = getRows (R, idx); // remove jth row
			// apply givens rotations to zero the last row
			const n = newR.n;

			for ( var k=j;k < newR.n-1 ; k++) {
				cs = givens(newR.val[k*n + k],newR.val[k*n + k + 1]);
				
				for ( var jj=k; jj < newR.m; jj++) {
					var rj = jj*n;				
					var t1;
					var t2;
						t1 = newR.val[rj + k]; 
						t2 = newR.val[rj + k+1]; 
						newR.val[rj + k] = cs[0] * t1 - cs[1] * t2; 
						newR.val[rj + k+1] = cs[1] * t1 + cs[0] * t2;
					}	
			}
			// and remove last zero'ed row before returning
			return {R: getCols(newR, range(newR.m)), rank: rank-1}; 
			// note: RR' is correct but R has opposite signs compared to chol(X'X)
		};
	// ---- 
	
	
	// Normalize features to mean=0 and norm=1
	var i,j,k;
	var Xt = transposeMatrix(X);
	var meanX = mean(Xt,2);
	Xt = sub(Xt, outerprod(meanX,ones(N)));
	var normX = norm(Xt,2); 
	for ( j=0; j < d; j++) {
		if( isZero(normX[j]) ) {
			// TODO deal with empty features... 
		} 
		else if( Math.abs(normX[j] - 1) > EPS ) { // otherwise no need to normalize
			k=j*N;
			for ( i=0; i< N; i++)
				Xt.val[k + i] /= normX[j];
		}
	}

	// Intialization
	var meanY = mean(Y);
	var r = subVectorScalar(Y,meanY);	// residuals (first step only bias = meanY)

	var activeset = new Array(); // list of active features
	var s; // signs
	var inactiveset = ones(d); 
		
	var beta = zeros(d); 
	var betas = zeros(d+1,d+1); 	
	betas.val[d] = meanY; // (first step only bias = meanY)
	var nvars = 0;
	
	var XactiveT;
	var R;
	var w;
	var A;
	var Ginv1;
	var gamma;
	var drop = -1;
	
	var maxiters = d; 
	if ( d > N-1 )
		maxiters = N-1;
	if (maxiters > n )
		maxiters = n;
	
	var lambda = new Array(maxiters);
	lambda[0] = Infinity; // only bias term in first solution

	// Initial correlations:
	var correlations = mulMatrixVector(Xt,r);
	

	var iter = 1;
	do {		

		// update active set
		var C = -1;
		for (var k=0; k < d; k++) {
			if ( inactiveset[k] == 1 && Math.abs(correlations[k] ) > C) 
				C = Math.abs(correlations[k] ) ;
		}

		lambda[iter] = C;		

		for (var k=0; k < d; k++) {
			if ( inactiveset[k] == 1 && isZero(Math.abs(correlations[k]) - C) ) {
				activeset.push( k );	
				inactiveset[k] = 0;	
			}		
		}
		s = signVector(getSubVector(correlations, activeset) );
		
		// Use cholesky updating to compute Ginv1 
		if ( activeset.length == 1 ) {
			
			R = updateR(Xt.row(activeset[0]) ); 
			A = 1;
			w = s[0];
			XactiveT = Xt.row(activeset[0]); 
			
			var u = mulScalarVector(w, XactiveT);
		
		}
		else {
			var xj = Xt.row(activeset[activeset.length - 1]);
			R = updateR(xj, R.R, XactiveT, R.rank);
			
			if ( R.rank < activeset.length ) {
				// skip this feature that is correlated with the others
				console.log("LARS: dropping variable " + activeset[activeset.length - 1] + " (too much correlation)");
				activeset.splice(activeset.length-1, 1);
				continue;
			}
			
			Ginv1 = backsubstitution( transposeMatrix(R.R), forwardsubstitution( R.R , s) ) ;
			A = 1 / Math.sqrt(sumVector(entrywisemulVector(Ginv1, s) ));
			w = mulScalarVector(A, Ginv1 );
			
			XactiveT = getRows(Xt, activeset); 
			var u = mulMatrixVector(transposeMatrix(XactiveT), w);
		
		}

		// Compute gamma
	
		if ( activeset.length < d ) {
			gamma = Infinity; 
			var a = zeros(d);
			for ( k=0; k < d; k++) {
				if ( inactiveset[k] == 1 ) {
					a[k] = dot(Xt.row(k), u);
					var g1 = (C - correlations[k]) / (A - a[k]);
					var g2 = (C + correlations[k]) / (A + a[k]);
					if ( g1 > EPS && g1 < gamma)
						gamma = g1;
					if ( g2 > EPS && g2 < gamma)
						gamma = g2;				
				}
			}
		}
		else {
			// take a full last step 
			gamma = C / A; 
		}
		
		// LASSO modification
		if ( method == "lasso") {
			drop = -1;
			for ( k=0; k < activeset.length-1; k++) {
				var gammak = -beta[activeset[k]] / w[k] ;
				if ( gammak > EPS && gammak < gamma ) {
					gamma = gammak; 
					drop = k;
				}
			}				
		}

		// Update beta
		if ( activeset.length > 1 ) {
			for ( var j=0; j < activeset.length; j++) {
				beta[activeset[j]] += gamma * w[j]; 		
			}
		}
		else 
			beta[activeset[0]] += gamma * w;

		// LASSO modification		
		if ( drop != -1 ) {
			console.log("LARS/Lasso dropped variable " + activeset[drop] );
			// drop variable
			inactiveset[activeset[drop]] = 1;
			beta[activeset[drop]] = 0; // should already be zero 
			activeset.splice(drop, 1);	
			// downdate cholesky factorization
			R = downdateR(drop, R.R, R.rank);
			
			XactiveT = getRows(Xt,activeset); 
			
			// increase total number steps to take (lasso does not terminate in d iterations)
			maxiters++;					
			betas = appendRow ( betas );
		}				

		// compute bias = meanY - dot(meanX, betascaled)
		var betascaled = zeros(d+1); 
		var bias = meanY;
		for ( var j=0; j < activeset.length; j++) {
			k = activeset[j]; 
			betascaled[k] = beta[k] / normX[k];
			bias -= betascaled[k] * meanX[k];
		} 
		betascaled[d] = bias;

		// save beta including rescaling and bias
		setRows(betas,[iter], betascaled);	
			
		if ( iter < maxiters ) {
			// update residuals for next step
			for ( k=0; k < N; k++) 
				r[k] -= gamma * u[k];	
				
			// and update correlations 
			if ( drop != -1 ) {
				// recompute correlations from scratch due drop
				correlations = mulMatrixVector(Xt,r);
			}
			else {
				for ( k=0; k < d; k++) {
					if ( inactiveset[k] == 1 ) {
						correlations[k] -= gamma * a[k];
					}
				}
			}				
		}

		iter++;
	} while ( activeset.length < d && iter <= maxiters );
	
	lambda = new Float64Array(lambda); // make it a vector;
		
	
	// return only the computed part of the path + bias term
	if ( iter < betas.m ) {
		betas = transposeMatrix(new Matrix(iter, betas.n, betas.val.subarray(0,betas.n*iter), true));
		return betas;
	}
	else {
		betas = transposeMatrix(betas);
		return betas;
	}
}

//////////////////////////////
/// OLS: Orthogonal Least Squares
/////////////////////////////
function OLS ( params) {
	var that = new Regression ( OLS, params);
	return that;
}
OLS.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:
	
	this.epsilon = "auto";
	this.dimension = "auto";
	this.affine = true;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			
}

OLS.prototype.train = function (X, y) {

	const N = X.m;
	const n = X.n;
	
	var Xreg; 
	if (this.affine)
		Xreg = mat([X,ones(N)]);
	else
		Xreg = X;

	var epsilon ;
	var nmax ;

	if ( this.epsilon == "auto" ) 
		epsilon = sumVector( entrywisemulVector(y, y) ) * 0.001; // XXX
	else 
		epsilon = this.epsilon;


	if  (this.dimension == "auto") 
		nmax = n;
	else 
		nmax = this.dimension;		// predefined dimension... 


	const epsilon2 = epsilon * epsilon;

	var S = []; 
	var notS = range(n);

	var err = Infinity; 
	
	var i,j,k;
	var residual;
	while ( (err > epsilon2) && (S.length < nmax) ) {
		
		// For each variable not in support of solution (S),
		//	minimize the error with support = S + this variable
		var e = zeros(notS.length); 

		i=0;
		var besti = -1;		
		err = Infinity; 
	
		while (i < notS.length && err > epsilon2) {
			j = notS[i]; 

			var Sj = []; 
			for (k=0; k < S.length; k++) 
				Sj.push(S[k]);
			Sj.push(j);
			
			if ( this.affine )
				Sj.push(n);

			var XS = getCols(Xreg, Sj ); 

			var x_j = solve(XS , y);	// use cholesky updating???

			if (typeof(x_j) != "number")
				residual = subVectors(mulMatrixVector( XS , x_j) , y); 
			else
				residual = subVectors(mulScalarVector( x_j, XS) , y); 
				
			for ( k=0; k < N; k++)
				e[i] += residual[k] * residual[k] ; 
				
			// Find best variable with minimum error
			if ( e[i] < err ) {
				err = e[i];
				besti = i; 
			}
			
			i++;
		}

		// add variable to support of solution
		S.push( notS[besti]);
		notS.splice(besti, 1); 
	}	
	
	var Sj = []; 
	for (k=0; k < S.length; k++) 
		Sj.push(S[k]);
	if ( this.affine )
		Sj.push(n);

	XS = getCols(Xreg, Sj ); 
	var xhat = zeros(n + (this.affine?1:0),1);
	x_j = solve(XS , y);
	set(xhat, Sj, x_j);
	if (typeof(x_j) != "number")
		residual = subVectors(mulMatrixVector( XS , x_j) , y); 
	else
		residual = subVectors(mulScalarVector( x_j, XS) , y); 
	err = 0;
	for ( k=0; k < N; k++)
		err += residual[k] * residual[k] ; 
		
	if ( this.affine ) {
		this.w = get(xhat,range(n));
		this.b = xhat[n];
	}
	else 
		this.w = xhat; 
		
	this.support = S;
	return err; 
}

//////////////////////////////////////////////////
/////	MLPreg: Multi-Layer Perceptron for regression
///////////////////////////////////////////////////
function MLPreg ( params) {
	var that = new Regression ( MLPreg, params);
	return that;
}
MLPreg.prototype.construct = function (params) {
	
	// Default parameters:

	this.loss = "squared";
	this.hidden = 5;	
	this.epochs = 1000;
	this.learningRate = 0.01;
	this.initialweightsbound = 0.1;

	
	this.normalization = "auto";
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 			
	}		

	// Parameter grid for automatic tuning:
	this.parameterGrid = {hidden: [5,10,15,30]}; 
}
MLPreg.prototype.train = function (Xorig, y) {
	// Training function
	if ( this.loss !="squared" )
		return "loss not implemented yet.";
 
 	//  normalize data
	if ( this.normalization != "none"  ) {	
		var norminfo = normalize(Xorig);
		this.normalization = {mean: norminfo.mean, std: norminfo.std}; 
		var X = norminfo.X;		
	}	
	else {
		var X = Xorig;
	}
	
	const N = size(X,1);
	const d = size(X,2);

	const minstepsize = Math.min(epsilon, 0.1 / ( Math.pow( 10.0, Math.floor(Math.log(N) / Math.log(10.0))) ) );

	var epsilon = this.learningRate;
	const maxEpochs = this.epochs;

	const hls = this.hidden; 

	var h;
	var output;
	var delta;
	var delta_w;
	var xi;
	var index;
	var k;
	
	/* Initialize the weights */

	if ( d > 1 )
		var Win = mulScalarMatrix( this.initialweightsbound, subMatrixScalar( rand(hls, d), 0.5 ) );
	else
		var Win = mulScalarVector( this.initialweightsbound, subVectorScalar( rand(hls), 0.5 ) );
		
	var Wout = mulScalarVector( this.initialweightsbound, subVectorScalar( rand(hls), 0.5 ) );

	var bin = mulScalarVector( this.initialweightsbound/10, subVectorScalar( rand(hls), 0.5 ) );
	var bout = (this.initialweightsbound/10) * (Math.random() - 0.5) ;

	var cost = 0;	
	for(var epoch = 1; epoch<=maxEpochs; epoch++) {
		
		if( epoch % 100 == 0)
			console.log("Epoch " + epoch, "Mean Squared Error: " + (cost/N));
		
		if(epsilon >= minstepsize)
			epsilon *= 0.998;

		var seq = randperm(N); // random sequence for stochastic descent
		
		cost = 0;
		for(var i=0; i < N; i++) {
			index = seq[i];
			
			/* Hidden layer outputs h(x_i) */
			if ( d > 1 ) {
				xi = X.row( index ); 			
				h =  tanh( addVectors( mulMatrixVector(Win, xi), bin ) );
			}
			else
				h =  tanh( addVectors( mulScalarVector(X[index] , Win), bin ) );
				
			/* Output of output layer g(x_i) */
			output =  dot(Wout, h) + bout ;

			var e = output - y[index];
			cost += e * e;
			
			/* delta_i for the output layer derivative dJ_i/dv = delta_i h(xi) */
			delta = e ;
			
			/* Vector of dj's in the hidden layer derivatives dJ_i/dw_j = dj * x_i */
			delta_w = mulScalarVector(delta, Wout);

			for(k=0; k < hls; k++) 
				delta_w[k] *=  (1.0 + h[k]) * (1.0 - h[k]); // for tanh units
				
			/* Update weights of output layer: Wout = Wout - epsilon * delta * h */
			saxpy( -epsilon*delta, h, Wout);
			/*
			for(var j=0; j<hls; j++)
				Wout[j] -= epsilon * delta * h[j];*/
				
			/* Update weights of hidden layer */
			if ( d > 1 ) {
				var rk = 0;
				for(k=0; k<hls; k++) {
					var epsdelta = epsilon * delta_w[k];
					for(j=0; j<d; j++)
						Win.val[rk + j] -= epsdelta * xi[j];
					rk += d;
				}
			}
			else {
				saxpy( -epsilon * X[index], delta_w, Win);
				/*
				for(k=0; k<hls; k++)
					Win[k] -= epsilon * delta_w[k] * X[index];
					*/
			}
			
			/* Update bias of both layers */
			saxpy( -epsilon, delta_w, bin);
			/*
			for(k=0; k<hls; k++)
			  bin[k] -= epsilon * delta_w[k]; */

			bout -= epsilon * delta;
		}
	}
	
	this.W = Win;
	this.V = Wout;
	this.w0 = bin;
	this.v0 = bout;
	this.dim_input = d;
	
	return cost;	
}

MLPreg.prototype.predict = function( x_unnormalized ) {
	// normalization
	var x;
	if (typeof(this.normalization) != "string" ) 
		x = normalize(x_unnormalized, this.normalization.mean, this.normalization.std);
	else
		x = x_unnormalized;
		
	// prediction

	var i;
	var k;
	var output;

	var tx = type(x);

	if ( (tx == "vector" && this.dim_input > 1) || (tx == "number" && this.dim_input == 1) ) {

		/* Output of hidden layer */
		if ( this.dim_input > 1 )
			var hidden = tanh( addVectors( mulMatrixVector(this.W, x), this.w0 ) );
		else 
			var hidden = tanh( addVectors( mulScalarVector(x, this.W), this.w0 ) );

		/* Output of output layer */
		var output = dot(this.V, hidden) + this.v0 ;
		return output;
	}
	else if ( tx == "matrix" || (tx == "vector" && this.dim_input == 1)) {
		output = zeros(x.length);
		for ( i=0; i < x.length; i++) {
			/* output of hidden layer */
			if ( this.dim_input > 1 )
				var hidden = tanh( addVectors( mulMatrixVector(this.W, x.row(i)), this.w0 ) );
			else
				var hidden = tanh( addVectors( mulScalarVector(x[i], this.W), this.w0 ) );
				
			/* output of output layer */
			output[i] = dot(this.V, hidden) + this.v0 ;
		}
		return output;
	}	
	else
		return "undefined";
}

/////////////////////////////////////////////////////////
////////// Switching Regression
/////////////////////////////////////////////////////////


function SwitchingRegression (algorithm, params ) {
	
	if ( typeof(algorithm) == "undefined" ) {
		var algorithm = kLinReg;
	}
	
	this.algorithm = algorithm.name;
	this.userParameters = params;

	// Functions that depend on the algorithm:
	this.construct = algorithm.prototype.construct; 

	this.train = algorithm.prototype.train; 
	if (  algorithm.prototype.predict ) 
		this.predict = algorithm.prototype.predict; // otherwise use default function for linear model
	
	// Initialization depending on algorithm
	this.construct(params);
}

SwitchingRegression.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:

	this.affine = true;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			

}

SwitchingRegression.prototype.tune = function ( X, y, Xv, yv ) {
	// Main function for tuning an algorithm on a given data set
	
	/*
		1) apply cross validation to estimate the performance of all sets of parameters
			in this.parameterGrid
			
			- for each cross validation fold and each parameter set, 
					create a new model, train it, test it and delete it.
			
		2) pick the best set of parameters and train the final model on all data
			store this model in this.* 
	*/
}

SwitchingRegression.prototype.train = function (X, y) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error.
	
	
	// Return training error:
	return this.test(X, y);

}

SwitchingRegression.prototype.predict = function (x, mode) {
	// Prediction function (default for linear model)

	var Y;
	const tx = type(x);
	const tw = type(this.W);
	
	if ( (tx == "vector" && tw == "matrix") || (tx == "number" && tw == "vector") ) {
		// Single prediction
		Y = mul(this.W, x);
		if ( this.affine  && this.b) 
			Y = add(Y, this.b);

		if ( typeof(mode) == "undefined" ) 
			return Y;
		else
			return Y[mode];
	}
	
	else {
		// multiple predictions
		if ( size(x, 2) == 1 ) {
			// one-dimensional case
			Y = outerprod(x, this.W); 			
		}
		else {
			Y = mul( x, transpose(this.W)); 
		}

		if ( this.affine  && this.b) 
			Y = add(Y, outerprod(ones(Y.length), this.b));
	
		var tmode = typeof(mode); 
		if ( tmode == "undefined" ) {
			// No mode provided, return the entire prediction matrix Y = [y1, y2, ... yn]
			return Y;	
		}
		else if (tmode == "number")  {
			// output of a single linear model
			return getCols(Y, [mode]);
		}
		else {
			// mode should be a vector 
			// return y = [.. y_i,mode(i) ... ]
			var y = zeros( Y.length ) ;
	
			var j;
			var idx;
			for ( j=0; j< this.n; j++) {
				idx = find(isEqual(mode, j));
		
				set(y, idx, get(Y,idx, j) );
			}
	
			return y;	
		}

	}		
		
}

SwitchingRegression.prototype.test = function (X, y) {
	// Test function: return the mean squared error (use this.predict to get the predictions)
	var prediction = this.predict( X ) ;	// matrix 

	var i;
	var errors = 0;
	var ei;
	if ( type(y) == "vector") {
		for ( i=0; i < y.length; i++) {
			ei = sub(prediction[i],  y[i]);
			errors += min( entrywisemul(ei, ei) ) ;
		}
		return errors/y.length; // javascript uses floats for integers, so this should be ok.
	}
	else {
		ei = sub(prediction,  y);		
		return  min( entrywisemul(ei, ei)); 
	}
}

SwitchingRegression.prototype.mode = function (X, y) {
	// Test function: return the estimated mode for all rows of X
	var prediction = this.predict( X ) ;	// matrix 

	if ( type(prediction) == "vector") {
		return argmin(prediction); 
	}
	else {
		var mode =zeros(X.length);
		for (var i=0; i < y.length; i++) {
			ei = sub(prediction.row(i),  y[i]);
			mode[i] = argmin( abs(ei) ) ;
		}
		return mode; // javascript uses floats for integers, so this should be ok.
	}	
}




SwitchingRegression.prototype.info = function () {
	// Print information about the model
	
	var str = "{<br>";
	var i;
	var Functions = new Array();
	for ( i in this) {
		switch ( type( this[i] ) ) {
			case "string":
			case "boolean":
			case "number":
				str += i + ": " + this[i] + "<br>";
				break;
			case "vector":
				if (  this[i].length <= 5 ) {
					str += i + ": [ " ;
					for ( var k=0; k < this[i].length-1; k++)
						str += this[i][k] + ", ";
 					str += this[i][k] + " ]<br>";
				}
				else
					str += i + ": vector of size " + this[i].length + "<br>";
				break;
			case "matrix":
				str += i + ": matrix of size " + this[i].m + "-by-" + this[i].n + "<br>";
				break;
			case "function": 
				Functions.push( i );
				break;
			default:
				str += i + ": " + typeof(this[i]) + "<br>";
				break;			
		}
	}
	str += "<i>Functions: " + Functions.join(", ") + "</i><br>";
	str += "}";
	return str;
}
//////////////////////////////////////
////// K-LinReg
////////////////////////////////////
function kLinReg ( params) {
	var that = new SwitchingRegression ( kLinReg, params);
	return that;
}
kLinReg.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
	
	// Default parameters:

	this.affine = true;
	this.n = 2;	// number of modes;
	this.restarts = 100;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}			

}
kLinReg.prototype.train = function (X, y) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error.
		
	var Xreg;
	if ( this.affine) 
		Xreg = mat([X, ones(X.length)]);
	else
		Xreg = X;
	
	const n = this.n;
	const N = y.length;
	const d = size(Xreg, 2); 
	const restarts = this.restarts;
	
	var W; // n by d matrix of parameters
	var Y = outerprod(y, ones(n)); // Y = [y, y,..., y]
	var E;
	var E2;
	var inModej;
	var idxj;
	var lbls = zeros(N);
	var i;
	var j;
	
	var bestW;
	var bestlbls;
	var min_mse = Infinity;
	var mse;
	
	var restart;
	
	for (restart = 0; restart < restarts; restart++) {
		// Init random param uniformly
		W = sub(mul(20,rand(n,d)) , 10); 

		err = -1;
		do {
			// Classify
			E = sub(Y , mul(Xreg,transpose(W) ) );
			E2 = entrywisemul(E, E);

			for ( i=0; i< N; i++) {
				lbls[i] = argmin( E2.row(i) ) ;
			}

			// compute parameters
			err_prec = err; 
			err = 0;
			for (j=0; j < n ; j++) {
				inModej = isEqual(lbls,j);
				if (sumVector(inModej) > d ) {
					idxj = find(inModej);
					if ( d > 1 ) 
						W.row(j).set(solve( getRows(Xreg, idxj) , getSubVector(y, idxj) ) );
					else
						W[j] = solve( getSubVector(Xreg, idxj) , getSubVector(y, idxj) ) ;
					
					err += sum ( get(E2, idxj, j) );
				}	
				else {
					err = Infinity;
					break;
				}
			}	
			
		} while ( ( err_prec < 0 || err_prec > err + 0.1 ) && err > EPS );

		mse = err / N;
		if(  mse < min_mse) {
			bestW = matrixCopy(W); 
			bestlbls = vectorCopy(lbls);
			min_mse = mse;
		}
	}
	
	if ( this.affine ) {
		this.W = get(bestW, [], range(d-1));
		this.b = get(bestW, [], d-1); 
	}
	else {
		this.W = bestW;
		if ( this.b )
			delete( this.b);			
	}
	
	return min_mse;
}

///////////////////////////
//// tools 
///////////////////////////


/**
 * Normalize the columns of X to zero mean and unit variance for dense X
 			return X for sparse X
 */
function normalize( X, means, stds ) {
	var tX = type(X);
	if ( arguments.length < 3 ) {
		if ( tX == "spvector" || tX == "spmatrix" )
			return {X: X, mean: NaN, std: NaN};
			
		var m = mean(X,1);
		var s = std(X,1);
		if ( typeof(s) != "number" ) {
			for ( var j=0; j < s.val.length; j++) {
				if( isZero(s.val[j]) )	// do not normalize constant features
					s.val[j] = 1;
			}
		}
		var Xn = entrywisediv( sub(X, mul(ones(X.length),m)), mul(ones(X.length),s) );
		if ( m ) {
			if ( typeof(m) == "number" )
				return {X: Xn, mean: m, std: s};
			else
				return {X: Xn, mean: m.val, std: s.val};
		}
		else
			return Xn;
	}
	else {
		if ( tX == "spvector" || tX == "spmatrix" )
			return X;
		
		if (tX != "matrix"){
			// X: single vector interpreted as a data row 
			return entrywisediv( sub(X,means), stds);
		}
		else {
			// X: matrix to normalize
			return entrywisediv( sub(X, outerprod(ones(X.length),means)), outerprod(ones(X.length),stds) );
		}
	}
}
/**
 * return an object {X: Matrix, y: vector} of AR regressors and outputs for time series prediction
 * @param {Float64Array}
 * @param {number} 
 * @return {Object: {Matrix, Float64Array} } 
 */
function ar( x, order ) {
	var i,j,k;
	if ( typeof(order) == "undefined")
		var order = 1;
		
	var X = new Matrix ( x.length - order , order );

	var y = zeros(x.length - order);
	
	k = 0;
	for ( i=order; i < x.length; i++) {
		for ( j=1; j <= order; j++) {
			X.val[k] = x[i-j];
			k++;
		}
		y[i-order] = x[i];		
	}

	return {X: X, y: y};
}
