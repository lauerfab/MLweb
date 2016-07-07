///////////////////////////////////////////
/// Generic functions for machine learning
//////////////////////////////////////////
function train( model, X, Y ) {
	switch( type(model).split(":")[0] ) {
	case "Classifier":
	case "Regression":
	case "SwitchingRegression":
		return model.train(X,Y);
		break;
	case "DimReduction":
		return model.train(X);
		break;
	default:
		return undefined;
	}
}
function predict( model, X, mode ) {
	switch( type(model).split(":")[0] ) {
	case "Classifier":
	case "Regression":
		return model.predict(X);
		break;
	case "SwitchingRegression":
		return model.predict(X,mode);
		break;
	default:
		return undefined;
	}
}
function test( model, X, Y ) {
	switch( type(model).split(":")[0] ) {
	case "Classifier":
	case "Regression":
	case "SwitchingRegression":
		return model.test(X,Y);
		break;
	default:
		return undefined;
	}
}
///////////////////////////////////////////
/// Generic class for Classifiers
//////////////////////////////////////////
/**
 * @constructor
 */
function Classifier (algorithm, params ) {

	if (typeof(algorithm) == "string") 
		algorithm = eval(algorithm);

	this.type = "Classifier:" + algorithm.name;

	this.algorithm = algorithm.name;
	this.userParameters = params;
	
	// this.type = "classifier";

	// Functions that depend on the algorithm:
	this.construct = algorithm.prototype.construct; 
	
	// Training functions
	this.train = algorithm.prototype.train; 
	if ( algorithm.prototype.trainBinary )
		this.trainBinary = algorithm.prototype.trainBinary; 
	if ( algorithm.prototype.trainMulticlass )
		this.trainMulticlass = algorithm.prototype.trainMulticlass; 
	if ( algorithm.prototype.update )
		this.update = algorithm.prototype.update; //online training
		
	// Prediction functions
	this.predict = algorithm.prototype.predict;
	if ( algorithm.prototype.predictBinary )
		this.predictBinary = algorithm.prototype.predictBinary;
	if ( algorithm.prototype.predictMulticlass ) 
		this.predictMulticlass = algorithm.prototype.predictMulticlass;
	if ( algorithm.prototype.predictscore )
		this.predictscore = algorithm.prototype.predictscore;
	if ( algorithm.prototype.predictscoreBinary )
		this.predictscoreBinary = algorithm.prototype.predictscoreBinary;
	
	
	// Tuning function	
	if ( algorithm.prototype.tune )
		this.tune = algorithm.prototype.tune; 
		
	// Initialization depending on algorithm
	this.construct(params);

}

Classifier.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
}

Classifier.prototype.tune = function ( X, y, Xv, yv ) {
	// Main function for tuning an algorithm on a given training set (X,labels) by cross-validation
	//	or by error minimization on the validation set (Xv, labelsv);

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

	if ( n == 0 ) {
		// no hyperparameter to tune, so just train and test
		if ( validationSet ) {
			this.train(X,y);
			var stats = 1.0 - this.test(Xv,yv);
		}
		else 
			var stats = 1.0 - this.cv(X,y);
		minValidError = stats ;
	}
	else if( n == 1 ) {
		// Just one hyperparameter
		var validationErrors = zeros(this.parameterGrid[parnames[0]].length);
		var bestpar; 		

		for ( var p =0; p <  this.parameterGrid[parnames[0]].length; p++ ) {
			this[parnames[0]] = this.parameterGrid[parnames[0]][p];
			
			console.log("Trying " + parnames[0] + " = " + this[parnames[0]] );
			if ( validationSet ) {
				// use validation set
				this.train(X,y);
				var stats = 1.0 - this.test(Xv,yv);
			}
			else {
				// do cross validation
				var stats = 1.0 - this.cv(X,y);
			}
			validationErrors[p] = stats;
			if ( stats < minValidError ) {
				minValidError = stats;
				bestpar = this[parnames[0]];
				if ( minValidError < 1e-4 )
					break;					
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
			
				console.log("Trying " + parnames[0] + " = " + this[parnames[0]] + ", " + parnames[1] + " = " + this[parnames[1]]);
			
				if ( validationSet ) {
					// use validation set
					this.train(X,y);
					var stats = 1.0 - this.test(Xv,yv);
				}
				else {
					// do cross validation
					var stats = 1.0 - this.cv(X,y);
				}
				validationErrors.val[p0*this.parameterGrid[parnames[1]].length + p1] = stats;
				if ( stats < minValidError ) {
					minValidError = stats ;
					bestpar[0] = this[parnames[0]];
					bestpar[1] = this[parnames[1]];
					if ( minValidError < 1e-4 )
						break;					

				}
				iter++;
				notifyProgress( iter / (this.parameterGrid[parnames[0]].length *this.parameterGrid[parnames[1]].length) ) ;
			}
			if ( minValidError < 1e-4 )
				break;
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
	return {error: minValidError,  validationErrors: validationErrors};
}

Classifier.prototype.train = function (X, labels) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error rate.
	
	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels ) ;
	/*
	// Call training function depending on binary/multi-class case
	if ( this.labels.length > 2 ) {
		this.trainMulticlass(X, y);	
	}
	else {
		this.trainBinary(X, y);
	}
	*/
	// Return training error rate:
	// return (1 - this.test(X, labels)); // not a good idea... takes time... 
	return this.info();
}
/*
Classifier.prototype.trainBinary = function (X, y) {
	// Training function for binary classifier 
	// assume y in {-1, +1} 
}
Classifier.prototype.trainMulticlass = function (X, y) {
	// Training function for multi-class case
	// assume y in {0, ..., Nclasses-1} 	
}
*/
Classifier.prototype.update = function (X, labels) {
	// Online training function: should update the classifier
	//	with additional training data in (X,labels)
	error("Error in " + this.algorithm + ".update(): Online training is not implemented for this classifier");
	return undefined;
}

Classifier.prototype.predict = function (X) {
	// Prediction function
	var y = 0; 
	
	// should return original labels converted from the numeric ones)
	var labels = this.recoverLabels( y ) ;
	return labels;
}

/*
Classifier.prototype.predictscore = function( x ) {
	// Return a real-valued score for the categories
}
*/
Classifier.prototype.test = function (X, labels) {
	// Test function: return the recognition rate (use this.predict to get the predictions)
	var prediction = this.predict( X ) ;

	var i;
	var errors = 0;
	if ( !isScalar(labels) ) {
		for ( i=0; i < labels.length; i++) {
			if ( prediction[i] != labels[i] ){
				errors++;
				//console.log("Prediction error on sample " + i + " :  " + prediction[i] + "/" + labels[i]);
			}
		}
		return (labels.length - errors)/labels.length; // javascript uses floats for integers, so this should be ok.
	}
	else {
		return ( prediction == labels);
	}
}

Classifier.prototype.cv = function ( X, labels, nFolds) {
	// Cross validation
	if ( typeof(nFolds) == "undefined" )
		var nFolds = 5;
	
	const N = labels.length;
	const foldsize = Math.floor(N / nFolds);
	
	// Random permutation of the data set
	var perm = randperm(N);
	
	// Start CV
	var errors = zeros (nFolds);
	
	var Xtr, Ytr, Xte, Yte;
	var i;
	var fold;
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
		errors[fold] = this.test(Xte,Yte);		
	}
	// last fold:
	this.train( get(X, get(perm, range(0, fold * foldsize)), []), get(labels, get(perm, range(0, fold * foldsize ) ) ) );		  
	errors[fold] = this.test(get(X, get(perm, range(fold * foldsize, N)), []), get(labels, get(perm, range(fold * foldsize, N)) ) );		  
	
	// Retrain on all data
	this.train(X, labels);

	// Return kFold error (or recognition rate??)
	return mean(errors);	
}

Classifier.prototype.checkLabels = function ( labels, binary01 ) {
	// Make list of labels and return corresponding numerical labels for training
	
	// Default check : make labels in { 0,..., Nclasses-1 } for multi-clas case
	//							or 	{-1, +1} for binary case (unless binary01 is true)
	
	var y = zeros(labels.length); // vector of numerical labels
	this.labels = new Array(); // array of original labels : this.labels[y[i]] = labels[i]
	this.numericlabels = new Array();
	
	var i;
	for ( i = 0; i<labels.length; i++) {
		if ( typeof(labels[i]) != "undefined" ) {
			y[i] = this.labels.indexOf( labels[i] );
			if( y[i] < 0 ) {
				y[i] = this.labels.length;
				this.labels.push( labels[i] );
				this.numericlabels.push( y[i] );				
			}
		}
		else {
			y[i] = 0;	// undefined labels = 0 (for sparse labels vectors)
			if ( this.labels.indexOf(0) < 0 ) {
				this.labels.push(0);
				this.numericlabels.push(0);
			}
		}
	}
	
	// Correct labels in the binary case => y in {-1, +1}
	if ( (arguments.length == 1 || !binary01) && this.labels.length == 2 ) {
		var idx0 = find(isEqual(y, 0) ) ;
		set(y, idx0, minus(ones(idx0.length)));
		this.numericlabels[this.numericlabels.indexOf(0)] = -1;
	}

	return y;
}

Classifier.prototype.recoverLabels = function ( y ) {
	// Return a vector of labels according to the original labels in this.labels
	// from a vector of numerical labels y
	
	// Default check : make labels in { 0,..., Nclasses-1 } for multi-clas case
	//							or 	{-1, +1} for binary case
	

	if ( typeof(y) == "number" )
		return  this.labels[this.numericlabels.indexOf( y ) ];
	else {
		var labels = new Array(y.length);// vector of true labels
		
		var i;
		for ( i = 0; i < y.length; i++) {
			labels[i] = this.labels[this.numericlabels.indexOf( y[i] )];		
		}
		return labels; 
	}	
}
/**
 * @return {string}
 */
Classifier.prototype.info = function () {
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
			case "spvector":
				str += i + ": " + printVector(fullVector(this[i])) + "<br>";
				break;			
			case "matrix":
				str += i + ": matrix of size " + this[i].m + "-by-" + this[i].n + "<br>";
				break;
			case "Array":
				str += i + ": Array of size " + this[i].length + "<br>";
				break;
			case "function": 
				if ( typeof(this[i].name)=="undefined" || this[i].name.length == 0 )
					Functions.push( i );
				else
					str += i + ": " + this[i].name + "<br>";
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
	return true if x contain a single data instance
			false otherwise
*/
Classifier.prototype.single_x = function ( x ) {
	var tx = type(x);
	return (tx == "number" || ( this.dim_input > 1 && (tx == "vector" || tx == "spvector" ) ) ) ;
}

//////////////////////////////////////////////////
/////		Linear Discriminat Analysis (LDA)
///////////////////////////////////////////////////

function LDA ( params ) {
	var that = new Classifier ( LDA, params);	
	return that;
}
LDA.prototype.construct = function (params) {
	
	// Default parameters:
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}		

	// Parameter grid for automatic tuning:
	this.parameterGrid = {   };
}

LDA.prototype.tune = function ( X, labels ) {
	var recRate = this.cv(X, labels);
	return {error: (1-recRate), validationErrors: [(1-recRate)]};
}

LDA.prototype.train = function ( X, labels ) {
	// Training function

	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels ) ;
	
	// Call training function depending on binary/multi-class case
	if ( this.labels.length > 2 ) {		
		this.trainMulticlass(X, y);		
	}
	else {
		var trainedparams = this.trainBinary(X, y);
		this.w = trainedparams.w; 
		this.b = trainedparams.b;
		this.dim_input = size(X,2);
	}
	/* and return training error rate:
	return (1 - this.test(X, labels));	*/
	return this.info();
}
LDA.prototype.trainBinary = function ( X, y ) {

	var i1 = find(isEqual(y,1));
	var i2 = find(isEqual(y,-1));
	var X1 = getRows(X, i1);
	var X2 = getRows(X, i2);
	var mu1 = mean(X1,1);
	var mu2 = mean(X2,1);
	var mudiff = sub(mu1, mu2);
	var musum = add(mu1, mu2);
	
	var X1centered = sub(X1, mul(ones(i1.length), mu1));
	var X2centered = sub(X2, mul(ones(i2.length), mu2));	
	var Sigma = add( mul(transpose(X1centered), X1centered), mul(transpose(X2centered), X2centered) ) ;

	Sigma = entrywisediv( Sigma , y.length - 2 );

	var w = solve(Sigma, mudiff.val );
	var b = Math.log( i1.length / i2.length ) - 0.5 * mul(musum, w);
	
	return {w: w, b: b, Sigma: Sigma, mu: [mu1, mu2] };
}
LDA.prototype.trainMulticlass = function ( X, y) {
	// Use the 1-against-all decomposition
	
	const dim = size(X, 2);
	this.dim_input = dim;
	
	const Nclasses = this.labels.length;

	var k;
	var idx;
	var i2;
	var Xk;
	var mu = new Array(Nclasses); 
	
	this.priors = zeros(Nclasses);
	
	var Xkcentered;

	var Sigma = zeros(dim,dim); 
	
	for ( k= 0; k < Nclasses; k++) {
	
		idx = find(isEqual(y,k));
		this.priors[k] = idx.length / y.length; 
		
		Xk = getRows(X, idx);

		mu[k] = mean(Xk,1).val;
	
		Xkcentered = sub(Xk, outerprod(ones(idx.length), mu[k]));

		Sigma = add( Sigma , mul(transpose(Xkcentered), Xkcentered)); 
	}
		
	Sigma = entrywisediv( Sigma , y.length - Nclasses );
	
	this.Sigma = Sigma;
	this.mu = mu;
	
	this.SigmaInv = inv(Sigma);
}

LDA.prototype.predict = function ( x ) {
	if ( this.labels.length > 2)
		return this.predictMulticlass( x ) ;
	else 
		return this.predictBinary( x );		
}
LDA.prototype.predictBinary = function ( x ) {
	
	var scores = this.predictscoreBinary( x , this.w, this.b);
	if (typeof(scores) != "undefined")
		return this.recoverLabels( sign( scores ) );
	else
		return "undefined";	
}
LDA.prototype.predictMulticlass = function ( x ) {
	// One-against-all approach
	
	var scores = this.predictscore( x );
	if (typeof(scores) != "undefined") {
		
		if ( type ( x ) == "matrix" ) {
			// multiple preidctions for multiple test data
			var i;
			var y = new Array(x.length );
			for ( i = 0; i < x.length; i++)  {
				y[i] = findmax ( scores.row(i) ) ;
			}
			return this.recoverLabels( y );
		}
		else {
			// single prediction
			return this.recoverLabels( argmax( scores ) );
		}
		
	}
	else
		return "undefined";	
}

LDA.prototype.predictscore = function( x ) {
	if ( this.labels.length > 2) {		
		if ( this.single_x( x ) ) {
			var k;
			var output = log(this.priors);
			
			for ( k = 0; k < this.mu.length; k++)  {
				var diff = sub(x, this.mu[k]);
				output[k] -= 0.5 * mul(diff, mul(this.SigmaInv, diff) ); 				
			}
			
			return output;
		}
		else {
			var k;
			var i;
			var output = zeros(x.length, this.labels.length);
			for ( i= 0; i< x.length; i++) {
				for ( k = 0; k < this.mu.length; k++)  {
					var diff = sub(x.row(i), this.mu[k]);
					output.val[i*output.n + k] = Math.log(this.priors[k]) - 0.5 * mul(diff, mul(this.SigmaInv, diff) ); 
				}
			}
			return output;
		}		
	}
	else 
		return this.predictscoreBinary( x, this.w, this.b );
}
LDA.prototype.predictscoreBinary = function( x , w, b ) {
	var output;
	if ( this.single_x( x ) ) 
		output = b + mul(x, w);
	else 
		output = add( mul(x, w) , b);
	return output;
}

//////////////////////////////////////////////////
/////		Perceptron
///////////////////////////////////////////////////

function Perceptron ( params ) {
	var that = new Classifier ( Perceptron, params);	
	return that;
}
Perceptron.prototype.construct = function (params) {
	
	// Default parameters:
	
	this.Nepochs = 100; 
	this.learningRate = 0.9;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}		

	// Parameter grid for automatic tuning:
	this.parameterGrid = { "learningRate" : range(0.1,1,0.1) };
}

Perceptron.prototype.train = function ( X, labels ) {
	// Training function

	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels ) ;
	
	// Call training function depending on binary/multi-class case
	if ( this.labels.length > 2 ) {		
		this.trainMulticlass(X, y);
		// and return training error rate:
		return (1 - this.test(X, labels));	
	}
	else {
		var trainedparams = this.trainBinary(X, y);
		this.w = trainedparams.w; 
		this.b = trainedparams.b;
		
		return trainedparams.trainingError;
	}
}
Perceptron.prototype.trainBinary = function ( Xorig, y ) {

	if ( type ( Xorig ) == "vector" )
		var X = mat([Xorig]); // make it a matrix
	else
		var X = Xorig;
		
	const N = y.length;
	const dim = X.n;
	this.dim_input = dim;
	var w_prev = zeros(dim);
	var w = zeros(dim); 
	var b_prev;
	
	var errors=0;
	
	var i;
	var j;
	
	// Uniform Random init
	for (j=0;j<dim;j++)
		w[j] = -5 + 10*Math.random();		
	var b = -5 + 10*Math.random();

	// Training
	var epoch = 0;
	var norm_diff=Infinity;
	while ( epoch < this.Nepochs && norm_diff > 0.0000001)  {
		errors = 0;
		w_prev = vectorCopy(w);
		b_prev = b;
		
		for(i = 0;i<N ;i++) {
			var Xi = X.row(i);
			var yp = this.predictscoreBinary(Xi, w, b);
			
			if(y[i] != Math.sign(yp) ) {
				errors++;
				saxpy(this.learningRate * y[i], Xi, w);
				/*
				for(j=0;j<dim;j++) {
					w[j] +=  this.learningRate * y[i] * Xi[j];
				}
				*/
				b -= this.learningRate * y[i];									
			}			
		}

		// Stopping criterion
		norm_diff = 0;
		for(j=0;j<dim;j++)
			norm_diff += (w[j] - w_prev[j]) * (w[j] - w_prev[j]);
		norm_diff += (b - b_prev) * (b-b_prev);
		
		epoch++;
	}
	
	// Return training error rate:
	return {trainingError: (errors / N), w: w, b: b };
}
Perceptron.prototype.trainMulticlass = function ( X, y) {
	// Use the 1-against-all decomposition
	
	const Nclasses = this.labels.length;

	var k;
	var yk;
	var trainedparams;
	
	// Prepare arrays of parameters to store all binary classifiers parameters
	this.b = new Array(Nclasses);
	this.w = new Array(Nclasses);

	for ( k = 0; k < Nclasses; k++) {

		// For all classes, train a binary classifier

		yk = sub(mul(isEqual( y, this.numericlabels[k] ), 2) , 1 ); // binary y for classe k = 2*(y==k) - 1 => in {-1,+1}
		
		trainedparams = this.trainBinary(X, yk );	// provide precomputed kernel matrix
		
		// and store the result in an array of parameters
		this.b[k] = trainedparams.b;
		this.w[k] = trainedparams.w;		
	}		
}

Perceptron.prototype.predict = function ( x ) {
	if ( this.labels.length > 2)
		return this.predictMulticlass( x ) ;
	else 
		return this.predictBinary( x );		
}
Perceptron.prototype.predictBinary = function ( x ) {
	
	var scores = this.predictscoreBinary( x , this.w, this.b);
	if (typeof(scores) != "undefined")
		return this.recoverLabels( sign( scores ) );
	else
		return "undefined";	
}
Perceptron.prototype.predictMulticlass = function ( x ) {
	// One-against-all approach
	
	var scores = this.predictscore( x );
	if (typeof(scores) != "undefined") {
		
		if ( type ( x ) == "matrix" ) {
			// multiple preidctions for multiple test data
			var i;
			var y = new Array(x.length );
			for ( i = 0; i < x.length; i++)  {
				y[i] = findmax ( scores.row(i) ) ;
			}
			return this.recoverLabels( y );
		}
		else {
			// single prediction
			return this.recoverLabels( argmax( scores ) );
		}
		
	}
	else
		return "undefined";	
}

Perceptron.prototype.predictscore = function( x ) {
	if ( this.labels.length > 2) {
		var k;
		var output = new Array(this.w.length);
		for ( k = 0; k < this.w.length; k++)  {
			output[k] = this.predictscoreBinary( x, this.w[k], this.b[k] );
		}
		if( type(x) == "matrix")
			return transpose(output) ; // matrix of scores is of size Nsamples-by-Nclasses 
		else
			return output; 		// vector of scores for all classes
	}
	else 
		return this.predictscoreBinary( x, this.w,  this.b );
}
Perceptron.prototype.predictscoreBinary = function( x , w, b ) {
	var output;
	if ( this.single_x( x ) ) 
		output = b + mul(x, w);
	else 
		output = add( mul(x, w) , b);
	return output;
}

//////////////////////////////////////////////////
/////		MLP: Multi-Layer Perceptron
///////////////////////////////////////////////////
function MLP ( params) {
	var that = new Classifier ( MLP, params);
	return that;
}
MLP.prototype.construct = function (params) {
	
	// Default parameters:

	this.loss = "squared";
	this.hidden = 5;	
	this.epochs = 1000;
	this.learningRate = 0.001;
	this.initialweightsbound = 0.01;

	
	this.normalization = "auto";
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 			
	}		

	// Parameter grid for automatic tuning:
	this.parameterGrid = {"hidden": [3, 5, 8, 12] } ; 
}
MLP.prototype.train = function (X, labels) {
	// Training function
		
	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels , true) ; // make y in {0,1} for the binary case
	
	// and normalize data
	if ( this.normalization != "none"  ) {	
		var norminfo = normalize(X);
		this.normalization = {mean: norminfo.mean, std: norminfo.std}; 
		var Xn = norminfo.X;
	}
	else {
		var Xn = X;
	}
	
	// Call training function depending on binary/multi-class case
	if ( this.labels.length > 2 ) {		
		this.trainMulticlass(Xn, y);	
	}
	else 
		this.trainBinary(Xn, y);
}
MLP.prototype.trainBinary = function ( X, y) {

	const N = X.m;
	const d = X.n;

	const minstepsize = Math.min(epsilon, 0.1 / ( Math.pow( 10.0, Math.floor(Math.log(N) / Math.log(10.0))) ) );

	var epsilon = this.learningRate;
	const maxEpochs = this.epochs;

	const hls = this.hidden; 
	
	var deltaFunc;
	switch ( this.loss ) {
	case "crossentropy":
		deltaFunc = function ( yi, g ) {
						return (g-yi);
					} ;	
		break;
	case "squared":
	default:
		deltaFunc = function ( yi, g ) { 
						return (g-yi) * (1-g) * g;
					} ;			
		break;
	}
	var h;
	var output;
	var delta_v;
	var delta_w;
	var xi;
	var index;
	var k;
	
	/* Initialize the weights */

	var Win = mulScalarMatrix( this.initialweightsbound, subMatrixScalar( rand(hls, d), 0.5 ) );
	var Wout = mulScalarVector( this.initialweightsbound, subVectorScalar( rand(hls), 0.5 ) );

	var bin = mulScalarVector( this.initialweightsbound/10, subVectorScalar( rand(hls), 0.5 ) );
	var bout = (this.initialweightsbound/10) * (Math.random() - 0.5) ;

//	var cost = 0;	
	for(var epoch = 1; epoch<=maxEpochs; epoch++) {
		
		if( epoch % 100 == 0)
			console.log("Epoch " + epoch); // "cost : " + cost);
		
		if(epsilon >= minstepsize)
			epsilon *= 0.998;

		var seq = randperm(N); // random sequence for stochastic descent
		
		// cost = 0;
		for(var i=0; i < N; i++) {
			index = seq[i];
			xi = X.row( index ); 
			
			/* Hidden layer outputs h(x_i) */
			h =  tanh( addVectors( mulMatrixVector(Win, xi), bin ) );
				
			/* Output of output layer g(x_i) */
			output =  sigmoid( dot(Wout, h) + bout ) ;

/*
			var e = output - y[index] ; // XXX only squared loss here...
			cost += e * e;
	*/
			
			/* delta_i for the output layer derivative dJ_i/dv = delta_i h(xi) */
			delta_v = deltaFunc(y[index], output); 
			
			/* Vector of dj's in the hidden layer derivatives dJ_i/dw_j = dj * x_i */
			delta_w = mulScalarVector(delta_v, Wout);

			for(k=0; k < hls; k++) 
				delta_w[k] *=  (1.0 + h[k]) * (1.0 - h[k]); // for tanh units
				
			/* Update weights of output layer */
			saxpy( -epsilon*delta_v, h, Wout);
			/*for(var j=0; j<hls; j++)
				Wout[j] -= epsilon * delta_v * h[j]; */
				
			/* Update weights of hidden layer */
			var rk = 0;
			for(k=0; k<hls; k++) {
				var epsdelta = epsilon * delta_w[k];
				for(j=0; j<d; j++)
					Win.val[rk + j] -= epsdelta * xi[j];
				rk += d;
			}
				
			/* Update bias of both layers */
			saxpy( -epsilon, delta_w, bin);
			/*for(k=0; k<hls; k++)
			    bin[k] -= epsilon * delta_w[k];*/

			bout -= epsilon * delta_v;
			  			
		}
	}
	
	
	this.W = Win;
	this.V = Wout;
	this.w0 = bin;
	this.v0 = bout;
	this.dim_input = d;	
}
MLP.prototype.trainMulticlass = function ( X, y) {
	
	const N = X.m;
	const d = X.n;
	const Q = this.labels.length;

	const minstepsize = Math.min(epsilon, 0.1 / ( Math.pow( 10.0, Math.floor(Math.log(N) / Math.log(10.0))) ) );

	var epsilon = this.learningRate;
	const maxEpochs = this.epochs;

	const hls = this.hidden; 
	
	var outputFunc; 
	var deltaFunc;
	switch ( this.loss ) {
	case "crossentropy":
		outputFunc = softmax;	
		deltaFunc = function ( yi, g ) {
						var delta = minus(g);
						delta[yi] += 1;
						return delta;
					} ;	
		break;
	case "squared":
	default:
		outputFunc = function (o) {
						var res=zeros(Q);
						for(var k=0; k<Q;k++)
							res[k] = sigmoid(o[k]);
						return res;
					};
		deltaFunc = function ( yi, g ) { 
						var delta = vectorCopy(g);
						delta[yi] -= 1;
						for(var k=0; k < Q; k++) 
							delta[k] *= (1-g[k])*g[k];
						return delta;
					} ;			
		break;
	}

	var h;
	var output;
	var delta_v;
	var delta_w;
	var xi;
	var index;
	var k;
	
	/* Initialize the weights */

	var Win = mulScalarMatrix( this.initialweightsbound, subMatrixScalar( rand(hls, d), 0.5 ) );
	var Wout = mulScalarMatrix( this.initialweightsbound, subMatrixScalar( rand(Q, hls), 0.5 ) );

	var bin = mulScalarVector( this.initialweightsbound/10, subVectorScalar( rand(hls), 0.5 ) );
	var bout = mulScalarVector( this.initialweightsbound/10, subVectorScalar( rand(Q), 0.5 ) );
	
	for(var epoch = 1; epoch<=maxEpochs; epoch++) {
		if( epoch % 100 == 0) {
			console.log("Epoch " + epoch);
		}
		
		if(epsilon >= minstepsize)
			epsilon *= 0.998;

		var seq = randperm(N); // random sequence for stochastic descent
		
		for(var i=0; i < N; i++) {
			index = seq[i];
			xi = X.row( index ); 
				
			/* Output of hidden layer h(x_i) */
			h = tanh( addVectors( mulMatrixVector(Win, xi), bin ) );

			/* Output of output layer g(x_i) */
			output = outputFunc( addVectors(mulMatrixVector(Wout, h), bout ) );

			/* delta_i vector for the output layer derivatives dJ_i/dv_k = delta_ik h(xi) */
		  	delta_v = deltaFunc(y[index], output);
		  		
			/* Vector of dj's in the hidden layer derivatives dJ_i/dw_j = dj * x_i */
			delta_w = mulMatrixVector(transpose(Wout), delta_v);

			for(k=0; k < hls; k++) 
				delta_w[k] *= (1.0 + h[k]) * (1.0 - h[k]); // for tanh units
				
			/* Update weights of output layer */
			var rk = 0;
			for(k=0; k<Q; k++) {
				var epsdelta = epsilon * delta_v[k];
				for(j=0; j<hls; j++)
					Wout.val[rk + j] -= epsdelta * h[j];
				rk += hls;
			}	
			/* Update weights of hidden layer */
			var rk = 0;
			for(k=0; k<hls; k++) {
				var epsdelta = epsilon * delta_w[k];
				for(j=0; j<d; j++)
					Win.val[rk + j] -= epsdelta * xi[j];
				rk += d;
			}
			
			/* Update bias of both layers */
			saxpy( -epsilon, delta_w, bin);
			saxpy( -epsilon, delta_v, bout);
				  
		}
	}
	
	this.W = Win;
	this.V = Wout;
	this.w0 = bin;
	this.v0 = bout;
	this.dim_input = d;		
	
	this.outputFunc = outputFunc;
}
function sigmoid ( x ) {
	return 1 / (1 + Math.exp(-x));
}
function sigmoid_prim ( x ) {
	return (1 - x) * x; // if x = sigmoid(u)
}
function softmax( x ) {
	const d=x.length;
	var sum = 0;
	var res = zeros(d);
	var k;
	for ( k=0; k < d; k++) {
		res[k] = Math.exp(-x[k]);
		sum += res[k];
	}
	for ( k=0; k < d; k++) 
		res[k] /= sum;
	return res;
}

MLP.prototype.predict = function ( x ) {
	if ( this.labels.length > 2)
		return this.predictMulticlass( x ) ;
	else 
		return this.predictBinary( x );		
}
MLP.prototype.predictBinary = function ( x ) {
	
	var scores = this.predictscore( x );
	if (typeof(scores) != "undefined")
		return this.recoverLabels( isGreaterOrEqual( scores, 0.5 ) );	
	else
		return "undefined";
}
MLP.prototype.predictMulticlass = function ( x ) {

	var scores = this.predictscore( x );
	if (typeof(scores) != "undefined") {
		
		if ( type ( x ) == "matrix" ) {
			// multiple predictions for multiple test data
			var i;
			var y = new Array(x.length );
			
			for ( i = 0; i < x.length; i++)  
				y[i] = findmax ( scores.row(i) ) ;
			
			return this.recoverLabels( y );
		}
		else {
			// single prediction
			return this.recoverLabels( argmax( scores ) );
		}
		
	}
	else
		return "undefined";	
}

MLP.prototype.predictscore = function( x_unnormalized ) {
	// normalization
	var x;
	if (typeof(this.normalization) != "string" ) 
		x = normalize(x_unnormalized, this.normalization.mean, this.normalization.std);
	else
		x = x_unnormalized;
		
	// prediction
	if ( this.labels.length > 2) {
		var i;
		var k;
		var output;

		if ( this.single_x( x ) ) {
	
			/* Calcul des sorties obtenues sur la couche cachée */
			var hidden = tanh( addVectors( mulMatrixVector(this.W, x), this.w0 ) );

			/* Calcul des sorties obtenues sur la couche haute */
			var output = this.outputFunc( addVectors(mulMatrixVector(this.V, hidden), this.v0 ) );	
			return output;
		}
		else {
			output = zeros(x.length, this.labels.length);
			for ( i=0; i < x.length; i++) {
				/* Calcul des sorties obtenues sur la couche cachée */
				var hidden = tanh( addVectors( mulMatrixVector(this.W, x.row(i)), this.w0 ) );

				/* Calcul des sorties obtenues sur la couche haute */
				var o =  this.outputFunc( addVectors(mulMatrixVector(this.V, hidden), this.v0 ) );
				setRows(output, [i], o);
			}
			return output;
		}	
	}
	else {
		var i;
		var k;
		var output;

		if ( this.single_x(x) ) {
	
			/* Calcul des sorties obtenues sur la couche cachée */
			var hidden = tanh( addVectors( mulMatrixVector(this.W, x), this.w0 ) );

			/* Calcul des sorties obtenues sur la couche haute */
			var output = dot(this.V, hidden) + this.v0 ;
			return sigmoid(output);
		}
		else {
			output = zeros(x.length);
			for ( i=0; i < x.length; i++) {
				/* Calcul des sorties obtenues sur la couche cachée */
				var hidden = tanh( addVectors( mulMatrixVector(this.W, x.row(i)), this.w0 ) );

				/* Calcul des sorties obtenues sur la couche haute */
				var o = dot(this.V, hidden) + this.v0 ;
				output[i] = sigmoid(o);
			}
			return output;
		}	
	}
}

//////////////////////////////////////////////////
/////		SVM
///////////////////////////////////////////////////
function SVM ( params) {
	var that = new Classifier ( SVM, params);
	return that;
}
SVM.prototype.construct = function (params) {
	
	// Default parameters:
	
	this.kernel = "linear";
	this.kernelpar = undefined;
	
	this.C = 1;
	
	this.onevsone = false;
	
	this.normalization = "auto";	
	this.alphaseeding = {use: false, alpha: undefined, grad: undefined};
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 			
	}		

	// Parameter grid for automatic tuning:
	switch (this.kernel) {
		case "linear":
			this.parameterGrid = { "C" : [0.01, 0.1, 1, 5, 10, 100] };
			break;
		case "gaussian":
		case "Gaussian":
		case "RBF":
		case "rbf": 
			// use multiples powers of 1/sqrt(2) for sigma => efficient kernel updates by squaring
			this.parameterGrid = { "kernelpar": pow(1/Math.sqrt(2), range(-1,9)), "C" : [ 0.1, 1, 5, 10, 50] };
			break;
			
		case "poly":
			this.parameterGrid = { "kernelpar": [3,5,7,9] , "C" : [0.1, 1, 5, 10, 50]  };
			break;
		case "polyh":
			this.parameterGrid = { "kernelpar": [3,5,7,9] , "C" : [0.1, 1, 5, 10, 50] };
			break;
		default:
			this.parameterGrid = undefined; 
			break;
	}
}
SVM.prototype.tune = function ( X, labels, Xv, labelsv ) {
	// Tunes the SVM given a training set (X,labels) by cross-validation or using validation data

	/* Fast implementation uses the same kernel cache for all values of C 
		and kernel updates when changing the kernelpar.
		
		We aslo use alpha seeding when increasing C.
	*/
	
	// Set the kernelpar range with the dimension
	if ( this.kernel == "rbf" ) {
		var saveKpGrid = zeros(this.parameterGrid.kernelpar.length);
		for ( var kp = 0; kp < this.parameterGrid.kernelpar.length ; kp ++) {
			saveKpGrid[kp] = this.parameterGrid.kernelpar[kp];
			if ( typeof(this.kernelpar) == "undefined")
				this.parameterGrid.kernelpar[kp] *= Math.sqrt( X.n ); 			
		}
		if ( typeof(this.kernelpar) != "undefined")
			this.parameterGrid.kernelpar = mul(this.kernelpar, range(1.4,0.7,-0.1)); 
	}
	
	
	if ( arguments.length == 4 ) {
		// validation set (Xv, labelsv)
		
		if ( this.kernel == "linear" ) {
			// test all values of C 
			var validationErrors = zeros(this.parameterGrid.C.length);
			var minValidError = Infinity;
			var bestC;
			
			for ( var c = 0; c < this.parameterGrid.C.length; c++) {
				this.C = this.parameterGrid.C[c];
				this.train(X,labels);
				validationErrors[c] = 1.0 - this.test(Xv,labelsv);
				if ( validationErrors[c] < minValidError ) {
					minValidError = validationErrors[c];
					bestC = this.C;					
				}
			}
			this.C = bestC;
			
			this.train(mat([X,Xv]), mat([labels,labelsv]) ); // retrain with best values and all data
		}
		else {
			// grid of ( kernelpar, C) values
			var validationErrors = zeros(this.parameterGrid.kernelpar.length, this.parameterGrid.C.length);
			var minValidError = Infinity;
			
			var bestkernelpar;
			var bestC;
			
			var kc = new kernelCache( X , this.kernel, this.parameterGrid.kernelpar[0] ); 

			for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
				this.kernelpar = this.parameterGrid.kernelpar[kp];
				if ( kp > 0 ) {
					kc.update( this.kernelpar );
				}
				for ( var c = 0; c < this.parameterGrid.C.length; c++) {
					this.C = this.parameterGrid.C[c];
					this.train(X,labels, kc);	// use the same kernel cache for all values of C
					validationErrors.set(kp,c, 1.0 - this.test(Xv,labelsv) );
					if ( validationErrors.get(kp,c) < minValidError ) {
						minValidError = validationErrors.get(kp,c);
						bestkernelpar = this.kernelpar;
						bestC = this.C;
					}
				}
			}
			this.kernelpar = bestkernelpar;
			this.C = bestC;
			this.train(mat([X,Xv], true), mat([labels,labelsv], true) ); // retrain with best values and all data
		}				
	}
	else {
		
		// 5-fold Cross validation
		const nFolds = 5;
	
		const N = labels.length;
		const foldsize = Math.floor(N / nFolds);
	
		// Random permutation of the data set
		var perm = randperm(N);
	
		// Start CV
		if ( this.kernel == "linear" ) 
			var validationErrors = zeros(this.parameterGrid.C.length);
		else 
			var validationErrors = zeros(this.parameterGrid.kernelpar.length,this.parameterGrid.C.length);
		
	
		var Xtr, Ytr, Xte, Yte;
		var i;
		var fold;
		for ( fold = 0; fold < nFolds - 1; fold++) {
			console.log("fold " + fold);
			notifyProgress ( fold / nFolds);
			Xte = get(X, get(perm, range(fold * foldsize, (fold+1)*foldsize)), []);
			Yte = get(labels, get(perm, range(fold * foldsize, (fold+1)*foldsize)) );
		
			var tridx = new Array();
			for (i=0; i < fold*foldsize; i++)
				tridx.push(perm[i]);
			for (i=(fold+1)*foldsize; i < N; i++)
				tridx.push(perm[i]);
		
			Xtr =  get(X, tridx, []);
			Ytr = get(labels, tridx);

			
			if ( this.kernel == "linear" ) {
				// test all values of C 		
				for ( var c = 0; c < this.parameterGrid.C.length; c++) {
					this.C = this.parameterGrid.C[c];
					console.log("training with C = " + this.C); // + " on " , tridx, Xtr, Ytr);
					this.train(Xtr,Ytr);
					validationErrors[c] += 1.0 - this.test(Xte,Yte) ;					
				}
			}
			else {
				// grid of ( kernelpar, C) values
			
				var kc = new kernelCache( Xtr , this.kernel, this.parameterGrid.kernelpar[0] ); 

				for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
					this.kernelpar = this.parameterGrid.kernelpar[kp];
					if ( kp > 0 ) {
						kc.update( this.kernelpar );
					}
					for ( var c = 0; c < this.parameterGrid.C.length; c++) {
						this.C = this.parameterGrid.C[c];
						console.log("Training with kp = " + this.kernelpar + " C = " + this.C);
						
						// alpha seeding: intialize alpha with optimal values for previous (smaller) C
						/* (does not help with values of C too different...
							if ( c == 0 )
								this.alphaseeding.use = false; 
							else {
								this.alphaseeding.use = true;
								this.alphaseeding.alpha = this.alpha; 
							}
						*/
						this.train(Xtr,Ytr, kc);	// use the same kernel cache for all values of C
						validationErrors.val[kp * this.parameterGrid.C.length + c] += 1.0 - this.test(Xte,Yte) ;						
					}
				}
			}
		}
		console.log("fold " + fold);
		notifyProgress ( fold / nFolds);
		// last fold:
		Xtr = get(X, get(perm, range(0, fold * foldsize)), []);
		Ytr = get(labels, get(perm, range(0, fold * foldsize ) ) ); 
		Xte = get(X, get(perm, range(fold * foldsize, N)), []);
		Yte = get(labels, get(perm, range(fold * foldsize, N)) );
		
		if ( this.kernel == "linear" ) {
			// test all values of C 		
			for ( var c = 0; c < this.parameterGrid.C.length; c++) {
				this.C = this.parameterGrid.C[c];
				console.log("training with C = " + this.C); 
				this.train(Xtr,Ytr);
				validationErrors[c] += 1.0 - this.test(Xte,Yte) ;
			}
		}
		else {
			// grid of ( kernelpar, C) values
	
			var kc = new kernelCache( Xtr , this.kernel, this.parameterGrid.kernelpar[0] ); 

			for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
				this.kernelpar = this.parameterGrid.kernelpar[kp];
				if ( kp > 0 ) {
					kc.update( this.kernelpar );
				}
				for ( var c = 0; c < this.parameterGrid.C.length; c++) {
					this.C = this.parameterGrid.C[c];
					console.log("Training with kp = " + this.kernelpar + " C = " + this.C);	
					// alpha seeding: intialize alpha with optimal values for previous (smaller) C
					/*
					if ( c == 0 )
						this.alphaseeding.use = false; 
					else {
						this.alphaseeding.use = true;
						this.alphaseeding.alpha = this.alpha; 
					}*/
				
					this.train(Xtr,Ytr, kc);	// use the same kernel cache for all values of C
					validationErrors.val[kp * this.parameterGrid.C.length + c] += 1.0 - this.test(Xte,Yte) ;					
				}
			}
		}		
	
		// Compute Kfold errors and find best parameters 
		var minValidError = Infinity;
		var bestC;
		var bestkernelpar;
		
		if ( this.kernel == "linear" ) {
			for ( var c = 0; c < this.parameterGrid.C.length; c++) {
				validationErrors[c] /= nFolds; 
				if ( validationErrors[c] < minValidError ) {
					minValidError = validationErrors[c]; 
					bestC = this.parameterGrid.C[c];
				}
			}
			this.C = bestC;
		}
		else {
			// grid of ( kernelpar, C) values
			for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
				for ( var c = 0; c < this.parameterGrid.C.length; c++) {
					validationErrors.val[kp * this.parameterGrid.C.length + c] /= nFolds;				
					if(validationErrors.val[kp * this.parameterGrid.C.length + c] < minValidError ) {
						minValidError = validationErrors.val[kp * this.parameterGrid.C.length + c]; 
						bestC = this.parameterGrid.C[c];
						bestkernelpar = this.parameterGrid.kernelpar[kp];
					}
				}
			}
			this.C = bestC;	
			this.kernelpar = bestkernelpar;	
		}
		
		//this.alphaseeding.use = false;
		
		// Retrain on all data
		this.train(X, labels);
		notifyProgress ( 1 );		
	}
	
	// Restore the dimension-free kernelpar range 
	if ( this.kernel == "rbf" ) {
		for ( var kp = 0; kp < this.parameterGrid.kernelpar.length ; kp ++) {
			this.parameterGrid.kernelpar[kp] = saveKpGrid[kp];
		}
	}
	
	this.validationError = minValidError; 
	return {error: minValidError, validationErrors: validationErrors}; 
}
SVM.prototype.train = function (X, labels, kc) {
	// Training function
	
	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels ) ;
	
	// and normalize data
	if ( this.normalization != "none" && this.kernel != "linear" ) {	// linear kernel should yield an interpretable model
		var norminfo = normalize(X);
		this.normalization = {mean: norminfo.mean, std: norminfo.std}; 
		var Xn = norminfo.X;
	}
	else {
		var Xn = X;
	}
	
	// Call training function depending on binary/multi-class case
	if ( this.labels.length > 2 ) {		
		this.trainMulticlass(Xn, y, kc);		
	}
	else {
		var trainedparams = this.trainBinary(Xn, y, kc);
		this.SVindexes = trainedparams.SVindexes;  // list of indexes of SVs	
		this.SVlabels = trainedparams.SVlabels;
		this.SV = trainedparams.SV;
		this.alpha = trainedparams.alpha;
		this.b = trainedparams.b;
		this.K = trainedparams.K;	// save kernel matrix for further use (like tuning parameters)
		this.kernelcache = trainedparams.kernelcache;
		this.dim_input = trainedparams.dim_input; // set input dimension for checks during prediction
		if ( this.kernelcache ) {
			this.kernelpar = this.kernelcache.kernelpar;
			if ( this.dim_input > 1 )
				this.kernelFunc = this.kernelcache.kernelFunc;	
			else
				this.kernelFunc = kernelFunction(this.kernel, this.kernelpar, "number"); // for scalar input kernelcache uses 1D-vectors
			this.sparseinput = (this.kernelcache.inputtype == "spvector"); 
		}

		this.w = trainedparams.w;				
		this.alphaseeding.grad = trainedparams.grad;
	}
	
	// free kernel cache
	this.kernelcache = undefined; 
	
	/* and return training error rate:
	if ( labels.length <= 2000 ) 
		return (1 - this.test(X, labels));
	else
		return "Training done. Training error is not automatically computed for large training sets.";
		*/
	return this.info();
}
SVM.prototype.trainBinary = function ( X, y, kc ) {
	
	// Training binary SVM with SMO
	
	// Prepare
	const C = this.C;

	// Use a different approach for linear kernel
	if ( this.kernel == "linear" ) { // && y.length  > 1000 ? 
		// Liblinear-like faster algo for linear SVM
		return SVMlineartrain( X, y, true, C );
	}
	
	
	if (typeof(kc) == "undefined") {
		// create a new kernel cache if it is not provided
		var kc = new kernelCache( X , this.kernel, this.kernelpar ); 
	}

	var i;
	var j;
	const m = X.length;	
	
	// linear cost			
	var c = minus(ones(m));
	
	// Initialization
	var alpha;
	var grad;
	var b = 0;

	if( this.alphaseeding.use ) {
		alpha = vectorCopy(this.alphaseeding.alpha);
		grad = vectorCopy(this.alphaseeding.grad);
	}
	else {
		alpha = zeros(m);
		grad = vectorCopy(c);
	}
	
	// SMO algorithm
	var index_i;
	var index_j;
	var alpha_i;
	var alpha_j;
	var grad_i;	
	var grad_j;
	var Q_i;
	var Q_j; 
	
	// List of indexes of pos and neg examples
	var y_pos_idx = new Array();
	var y_neg_idx = new Array();
	for ( i=0; i < m; i++) {
		if ( y[i] > 0 ) 
			y_pos_idx.push(i);
		else
			y_neg_idx.push(i);
	}
	
	// Function computing Q[i,:] = y_i * y .* K[i,:]
	var computeQ_row = function ( i ) {
		var Qi = kc.get_row( i );
		var k;
		var ii;
		if ( y[i] > 0 ) {		
			var m_neg = y_neg_idx.length;
			for ( k = 0; k < m_neg; k++ ) {
				ii = y_neg_idx[k];
				Qi[ii] = -Qi[ii];
			}
		}
		else {
			var m_pos = y_pos_idx.length;
			for ( k = 0; k < m_pos; k++ ) {
				ii = y_pos_idx[k];
				Qi[ii] = -Qi[ii];
			}
		}
		return Qi;
	};
	
	
	const tolalpha = 0.001; // tolerance to detect margin SV
	const Cup = C * (1-tolalpha) ;
	const Clow = C * tolalpha;
	
	const epsilon = 0.001; // TOL on the convergence
	var iter = 0;
	do {
		// working set selection => {index_i, index_j }
		var gradmax = -Infinity;
		var gradmin = Infinity;
					
		for (i=0; i< m; i++) {
			alpha_i = alpha[i];
			grad_i = grad[i];
			if ( y[i] == 1 && alpha_i < Cup && -grad_i > gradmax ) {
				index_i = i;
				gradmax = -grad_i; 
			}
			else if ( y[i] == -1 && alpha_i > Clow  && grad_i > gradmax ) {
				index_i = i;
				gradmax = grad_i; 
			}
			
			if ( y[i] == -1 && alpha_i < Cup && grad_i < gradmin ) {
				index_j = i;
				gradmin = grad_i;
			}
			else if ( y[i] == 1 && alpha_i > Clow && -grad_i < gradmin ) {
				index_j = i;
				gradmin = -grad_i;
			}
				
		}			
		
		// Analytical solution
		i = index_i;
		j = index_j;

		//  Q[i][j] = y_i y_j K_ij
		Q_i = computeQ_row( i );
		Q_j = computeQ_row( j );
		//Q_i = entrywisemulVector( mulScalarVector( y[i] , y) , kc.get_row( i ) ); // ith row of Q
		//Q_j = entrywisemulVector( mulScalarVector( y[j] , y) , kc.get_row( j ) ); // jth row of Q
		
		alpha_i = alpha[i];
		alpha_j = alpha[j];
		grad_i = grad[i];
		grad_j = grad[j];
					
		// Update alpha and correct to remain in feasible set
		if ( y[i] != y[j] ) {
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

		// Update gradient 
		// gradient = Q alpha + c 
		// ==>  grad += Q_i* d alpha_i + Q_j d alpha_j; Q_i = Q[i] (Q symmetric)	
		var dai = alpha[i] - alpha_i;
		if ( Math.abs(dai) > 1e-8 ) {
			saxpy(dai, Q_i, grad);
			/*
			for (i=0; i< m; i++) 
				grad[i] += dai * Q_i[i];
				*/
		}
		var daj = alpha[j] - alpha_j;
		if ( Math.abs(daj) > 1e-8 ) {
			saxpy(daj, Q_j, grad);
			/*for (i=0; i< m; i++) 
				grad[i] += daj * Q_j[i];*/
		}
		
		iter++;
		if( iter % 1000 == 0)
			console.log("SVM iteration " + iter + ", stopping criterion = " + (gradmax - gradmin) );
	} while ( iter < 100000 && gradmax - gradmin > epsilon ) ; 
	

	// Compute margin: // see SVR
	var Qalpha = sub ( grad, c ); // because gradient = Q alpha + c
	var sumAlphaYK_ij = mul(alpha, Qalpha );			
	var marginvalue = 1.0 / (Math.sqrt(2 * sumAlphaYK_ij));
	
	var insideMargin = find( isEqual(alpha, C) );
	
	if ( !isFinite(marginvalue) || insideMargin.length == alpha.length) 
		b = 0;
	else {
		// Compute b (from examples well on the margin boundary, with alpha_i about C/2:
		var nMarginSV = 0;
		var tol = 0.9;
				
		while ( nMarginSV < 2 && tol > 1e-6  ) {
			tol *= 0.5;
			nMarginSV = 0;
			b = 0;				

			for(i=0;i < m ;i++) {
				if ( alpha[i] > tol*C && alpha[i] < (1.0 - tol) * C ) {
					
					var ayKi = dot(entrywisemulVector(alpha, y), kc.get_row( i ) ) ;
						
					b += y[i] - ayKi;	// b = 1/N sum_i (y_i - sum_j alpha_j y_j Kij) 
					nMarginSV++;
				}										
			}
		}
		if ( tol <= 1e-6 ) {
			b = 0;				
			for(i=0;i < m ;i++) {
				if ( alpha[i] > tol*C ) {
					
					var ayKi = dot(entrywisemulVector(alpha, y), kc.get_row( i ) ) ;
						
					b += y[i] - ayKi;	// b = 1/N sum_i (y_i - sum_j alpha_j y_j Kij) 
					nMarginSV++;
				}										
			}
		}
		
		
		b /= nMarginSV;
	}
	
	/* Find support vectors	*/
	var tola = 1e-6;
	var nz = isGreater(alpha, tola);
	var SVindexes = find(nz); // list of indexes of SVs	
	while ( SVindexes.length < 2  && tola > 2*EPS) {
		tola /= 10;
		nz = isGreater(alpha, tola);
		SVindexes = find(nz); // list of indexes of SVs		
	}
	
	var SVlabels = get(y, SVindexes);
	var SV = get(X,SVindexes, []) ;
	switch ( type ( SV ) ) {
		case "spvector":
			SV = sparse(transpose(SV)); // matrix with 1 row
			break;
		case "vector": 
			SV = transpose(SV);
			break;
	}

	alpha = entrywisemul(nz, alpha); // zeroing small alpha_i	
	var dim_input = 1;
	if ( type(X) != "vector")
		dim_input = X.n; // set input dimension for checks during prediction
	
	// Compute w for linear classifiers
	var w;
	if ( this.kernel == "linear" ) {
		if(SVindexes.length > 1)
			w = transpose(mul( transpose( entrywisemul(get (alpha, SVindexes) , SVlabels) ), SV ));
		else
			w = vectorCopy(SV);
	}
	
	
	return {"SVindexes": SVindexes, "SVlabels": SVlabels, "SV": SV, "alpha": alpha, "b": b, "kernelcache": kc, "dim_input": dim_input, "w": w, "grad": grad};
}

SVM.prototype.trainMulticlass = function ( X, y, kc) {
	
	const Nclasses = this.labels.length;

	if ( this.onevsone ) {
		// Use the 1-against-1 decomposition
		const Nclassifiers = Math.round(Nclasses*(Nclasses-1)/2); 
	
		var j,k,l;
		var ykl;
		var Xkl;
		var trainedparams;
	
		// Prepare arrays of parameters to store all binary classifiers parameters
		this.SVindexes = new Array(Nclassifiers);
		this.SVlabels = new Array(Nclassifiers);
		this.SV = new Array(Nclassifiers);
		this.alpha = new Array(Nclassifiers);
		this.b = new Array(Nclassifiers);
		this.w = new Array(Nclassifiers);
	
		// Prepare common kernel cache for all classes 
		//this.kernelcache = kc;

		// Index lists
		var indexes = new Array(Nclasses);
		for ( k = 0; k < Nclasses; k++) {
			indexes[k] = new Array();
			for ( var i=0; i < y.length; i++) {
				if ( y[i] == this.numericlabels[k] )
					indexes[k].push(i);
			}
		}


		j = 0; // index of binary classifier
		for ( k = 0; k < Nclasses-1; k++) {
			for ( l = k+1; l < Nclasses; l++) {
				// For all pair of classes (k vs l), train a binary SVM
			
				Xkl = get(X, indexes[k].concat(indexes[l]), [] );		
				ykl = ones(Xkl.length);
				set(ykl, range(indexes[k].length, Xkl.length), -1);
		
				trainedparams = this.trainBinary(Xkl, ykl);	

				// and store the result in an array of parameters
				this.SVindexes[j] = trainedparams.SVindexes;  // list of indexes of SVs	
				this.SVlabels[j] = trainedparams.SVlabels;
				this.SV[j] = trainedparams.SV;
				this.alpha[j] = trainedparams.alpha;
				this.b[j] = trainedparams.b;
				this.w[j] = trainedparams.w;
		
				if ( j == 0 ) {
					// for first classifier only:
					this.dim_input = trainedparams.dim_input; // set input dimension for checks during prediction
					if ( trainedparams.kernelcache ) {
						this.kernelpar = trainedparams.kernelcache.kernelpar;
						if ( this.dim_input > 1 )
							this.kernelFunc = trainedparams.kernelcache.kernelFunc;				
						else
							this.kernelFunc = kernelFunction(this.kernel, this.kernelpar, "number"); // for scalar input
						this.sparseinput = (trainedparams.kernelcache.inputtype == "spvector"); 
					}
				}
				trainedparams.kernelcache = undefined;
		
				console.log("SVM #" + (j+1) + " trained for classes " + (k+1) + " vs " + (l+1) + " (out of " + Nclasses+ ")");
				j++;
			}
		}		
	}
	else {
		// Use the 1-against-all decomposition
	
		var k;
		var yk;
		var trainedparams;
	
		// Prepare arrays of parameters to store all binary classifiers parameters
		this.SVindexes = new Array(Nclasses);
		this.SVlabels = new Array(Nclasses);
		this.SV = new Array(Nclasses);
		this.alpha = new Array(Nclasses);
		this.b = new Array(Nclasses);
		this.w = new Array(Nclasses);
	
		// Prepare common kernel cache for all classes 
		this.kernelcache = kc;

		for ( k = 0; k < Nclasses; k++) {

			// For all classes, train a binary SVM

			yk = sub(mul(isEqual( y, this.numericlabels[k] ), 2) , 1 ); // binary y for classe k = 2*(y==k) - 1 => in {-1,+1}
		
			trainedparams = this.trainBinary(X, yk, this.kernelcache);	// provide precomputed kernel matrix

			// and store the result in an array of parameters
			this.SVindexes[k] = trainedparams.SVindexes;  // list of indexes of SVs	
			this.SVlabels[k] = trainedparams.SVlabels;
			this.SV[k] = trainedparams.SV;
			this.alpha[k] = trainedparams.alpha;
			this.b[k] = trainedparams.b;
			this.w[k] = trainedparams.w;
		
			if ( k == 0 ) {
				// for first classifier only:
				this.kernelcache = trainedparams.kernelcache;	// save kernel cache for further use (like tuning parameters)
				this.dim_input = trainedparams.dim_input; // set input dimension for checks during prediction
				if ( this.kernelcache ) {
					this.kernelpar = this.kernelcache.kernelpar;
					if ( this.dim_input > 1 )
						this.kernelFunc = this.kernelcache.kernelFunc;				
					else
						this.kernelFunc = kernelFunction(this.kernel, this.kernelpar, "number"); // for scalar input
					this.sparseinput = (this.kernelcache.inputtype == "spvector"); 
				}
			}
		
			console.log("SVM trained for class " + (k+1) +" / " + Nclasses);
		}
	}		
}


SVM.prototype.predict = function ( x ) {
	const tx = type(x); 
	if ( this.sparseinput ) {
		if ( tx != "spvector" && tx != "spmatrix")
			x = sparse(x); 
	}
	else {
		if ( tx == "spvector" || tx == "spmatrix" )
			x = full(x);
	}
	
	if ( this.labels.length > 2)
		return this.predictMulticlass( x ) ;
	else 
		return this.predictBinary( x );		
}
SVM.prototype.predictBinary = function ( x ) {
	
	var scores = this.predictscore( x );
	if (typeof(scores) != "undefined")
		return this.recoverLabels( sign( scores ) );
	else
		return "undefined";	
}
SVM.prototype.predictMulticlass = function ( x ) {	
	var scores = this.predictscore( x );
	const Q = this.labels.length;
			
	if (typeof(scores) != "undefined") {
		var tx = type(x);
		if ( (tx == "vector" && this.dim_input == 1) || tx == "matrix" || tx == "spmatrix" ) {
			// multiple predictions for multiple test data
			var i;
			var y = new Array(x.length );
			
			if ( this.onevsone ) {
				// one-vs-one: Majority vote
				var kpos, kneg;
				var votes;
				var k = 0;
				for ( i = 0; i < x.length; i++) {
				 	votes = new Uint16Array(Q);
					for ( kpos = 0; kpos < Q -1; kpos++) {
						for ( kneg = kpos+1; kneg < Q; kneg++) {							
							if ( scores.val[k] >= 0 ) 
								votes[kpos]++;
							else
								votes[kneg]++;
							k++; 
						}
					}
					y[i] = 0;
					for ( var c = 1; c < Q; c++)
						if ( votes[c] > votes[y[i]] )
							y[i] = c ;
				}		
			}
			else {
				// one-vs-rest: argmax
				for ( i = 0; i < x.length; i++)  
					y[i] = findmax ( scores.row(i) ) ;
			}
			return this.recoverLabels( y );
		}
		else {
			// single prediction
			
			if ( this.onevsone ) {
				// one-vs-one: Majority vote
				var kpos, kneg;
				var votes = new Uint16Array(Q);
				var k = 0;
				for ( kpos = 0; kpos < Q -1; kpos++) {
					for ( kneg = kpos+1; kneg < Q; kneg++) {							
						if ( scores[k] >= 0 ) 
							votes[kpos]++;
						else
							votes[kneg]++;
						k++; 
					}
				}
				var y = 0;
				for ( var c = 1; c < Q; c++)
					if ( votes[c] > votes[y[i]] )
						y[i] = c ;
				return this.recoverLabels( y ); 	
			}
			else
				return this.recoverLabels( argmax( scores ) );
		}
		
	}
	else
		return "undefined";	
}

SVM.prototype.predictscore = function( x ) {
	// normalization
	var xn;
	if (typeof(this.normalization) != "string" ) 
		xn = normalize(x, this.normalization.mean, this.normalization.std);
	else
		xn = x;
		
	// prediction
	if ( this.labels.length > 2) {
		var k;
		var output = new Array(this.labels.length);
		for ( k = 0; k < this.alpha.length; k++)  {	
			output[k] = this.predictscoreBinary( xn, this.alpha[k], this.SVindexes[k], this.SV[k], this.SVlabels[k], this.b[k], this.w[k] );
		}
		var tx = type(xn);
		if( tx == "matrix" || tx == "spmatrix" )
			return mat(output) ; // matrix of scores is of size Nsamples-by-Nclasses/Nclassifiers 
		else
			return output; 		// vector of scores for all classes
	}
	else 
		return this.predictscoreBinary( xn, this.alpha, this.SVindexes, this.SV, this.SVlabels, this.b, this.w );
}
SVM.prototype.predictscoreBinary = function( x , alpha, SVindexes, SV, SVlabels, b, w ) {

	var i;
	var j;
	var output;	

	if ( this.single_x(x) ) {
	
		output = b;
		if ( this.kernel =="linear" && w)
			output += mul(x, w);
		else {
			for ( j=0; j < SVindexes.length; j++) {
				output += alpha[SVindexes[j]] * SVlabels[j] * this.kernelFunc(SV.row(j), x); // kernel ( SV.row(j), x, this.kernel, this.kernelpar);
			}
		}
		return output;
	}
	else {
		if (  this.kernel =="linear" && w)
			output = add( mul(x, w) , b);
		else {
			// Cache SVs
			var SVs = new Array(SVindexes.length);
			for ( j=0; j < SVindexes.length; j++) 
				SVs[j] = SV.row(j);
		
			output = zeros(x.length);
			for ( i=0; i < x.length; i++) {
				output[i] = b;
				var xi = x.row(i);
				for ( j=0; j < SVindexes.length; j++) {
					output[i] += alpha[SVindexes[j]] * SVlabels[j] * this.kernelFunc(SVs[j], xi ); 
				}
			}
		}
		return output;
	}	
}

function SVMlineartrain ( X, y, bias, C) {
	
	// Training binary LINEAR SVM with Coordinate descent in the dual 
	//	as in "A Dual Coordinate Descent Method for Large-scale Linear SVM" by Hsieh et al., ICML 2008.
/*
m=loadMNIST("examples/")
svm=new Classifier(SVM)
svm.train(m.Xtrain[0:100,:], m.Ytrain[0:100])
*/
	
	var i;
	const m = X.length;	
	var Xb;
	
	switch(type(X)) {	
		case "spmatrix":
			if ( bias ) 
				Xb = sparse(mat([full(X), ones(m)])); // TODO use spmat()
			else
				Xb = X;

			var _dot = dotspVectorVector; 
			var _saxpy = spsaxpy;
			
			break;
		case "matrix":
			if ( bias ) 
				Xb = mat([X, ones(m)]);
			else
				Xb = X;	

			var _dot = dot;
			var _saxpy = saxpy;

			break;
		default:
			return undefined;
	}
	
		
	const d = size(Xb,2);	
		
	var optimal = false; 
	var alpha = zeros(m); 
	var w = zeros(d); 
	
	const Qii = sumMatrixCols( entrywisemulMatrix( Xb,Xb) ); 

	const maxX = norminf(Xb);

	var order;
	var G = 0.0;
	var PG = 0.0;
	var ii = 0;
	var k = 0;
	var u = 0.0;
	var alpha_i = 0.0;
	var Xbi;	
	
	
	var iter = 0;
	// first iteration : G=PG=-1
	const Cinv = 1/C;
	for ( i=0; i < m; i++) {		
		if ( Qii[i] < Cinv ) 	
			alpha[i] = C;		
		else
			alpha[i] =  1 / Qii[i] ;
			
		
		Xbi = Xb.row(i);
		u = alpha[i] * y[i] ;
		for ( k=0; k < d; k++)
			w[k] += u * Xbi[k]; 
	}
	
	iter = 1; 
	do {	
		// Outer iteration 
		order = randperm(m); // random order of subproblems
		
		optimal = true;
		for (ii=0; ii < m; ii++) {
			i = order[ii];
			
			if ( Qii[i] > EPS ) {
			
				Xbi = Xb.row(i);
			
				alpha_i = alpha[i];

				G = y[i] * dot(Xbi, w) - 1; 
			
				if ( alpha_i <= EPS ) {
					PG = Math.min(G, 0);
				}
				else if ( alpha_i >= C - EPS ) {
					PG = Math.max(G, 0);
				}
				else {
					PG = G;
				} 

				if ( Math.abs(PG) > 1e-6 ) {
					optimal = false;
					alpha[i] = Math.min( Math.max( alpha_i - (G / Qii[i]), 0 ), C );
					
					// w = add(w, mul( (alpha[i] - alpha_i)*y[i] , Xb[i] ) );
					u = (alpha[i] - alpha_i)*y[i];
					if ( Math.abs(u) > 1e-6 / maxX ) {
						for ( k=0; k < d; k++)
							w[k] += u * Xbi[k]; 
					}
				}
			}
		}

		iter++;
		/*if ( Math.floor(iter / 1000) == iter / 1000 ) 
			console.log("SVM linear iteration = " + iter);*/
	} while ( iter < 10000 && !optimal ) ; 

	var b;
	if ( bias ) {
		b = w[d-1]; 
		w = get ( w, range(d-1) );
	}
	else {
		b = 0;		
	}
	
	// Compute SVs: 
	var nz = isGreater(alpha, EPS);
	var SVindexes = find(nz); // list of indexes of SVs	
	var SVlabels = get(y, SVindexes);
	var SV;// = get(X,SVindexes, []) ;
	alpha = entrywisemul(nz, alpha); // zeroing small alpha_i	
	
	return {"SVindexes": SVindexes, "SVlabels": SVlabels, "SV": SV, "alpha": alpha, "b": b, "dim_input":  w.length, "w": w};
}

//////////////////////////////////////////////////
/////		MSVM
///////////////////////////////////////////////////
function MSVM ( params) {
	var that = new Classifier ( MSVM, params);
	return that;
}
MSVM.prototype.construct = function (params) {
	
	// Default parameters:
	this.MSVMtype = "CS";
	this.kernel = "linear";
	this.kernelpar = undefined;
	
	this.C = 1;
	
	this.optimAccuracy = 0.97;
	this.maxIter = 1000000;
	
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}	
	
	// Set defaults parameters depending on MSVM type
	if ( typeof(this.chunk_size) == "undefined") {
		if ( this.MSVMtype == "CS")
			this.chunk_size = 2;
		else
			this.chunk_size = 10;
	}

	// Parameter grid for automatic tuning:
	switch (this.kernel) {
		case "linear":
			this.parameterGrid = { "C" : [0.1, 1, 5, 10, 50] };
			break;
		case "gaussian":
		case "Gaussian":
		case "RBF":
		case "rbf": 
			// use multiples powers of 1/sqrt(2) for sigma => efficient kernel updates by squaring
			this.parameterGrid = { "kernelpar": pow(1/Math.sqrt(2), range(-1,9)), "C" : [ 0.1, 1, 5, 10] };
			//this.parameterGrid = { "kernelpar": [0.1,0.2,0.5,1,2,5] , "C" : [ 0.1, 1, 5, 10, 50] };
			break;
			
		case "poly":
			this.parameterGrid = { "kernelpar": [3,5,7,9] , "C" : [ 0.1, 1, 5, 10] };
			break;
		case "polyh":
			this.parameterGrid = { "kernelpar": [3,5,7,9] , "C" : [ 0.1, 1, 5, 10] };
			break;
		default:
			this.parameterGrid = undefined; 
			break;
	}
}

MSVM.prototype.tune = function ( X, labels, Xv, labelsv ) {
	// Tunes the SVM given a training set (X,labels) by cross-validation or using validation data

	/* Fast implementation uses the same kernel cache for all values of C 
		and kernel updates when changing the kernelpar.
		
		We aslo use alpha seeding when increasing C.
	*/
	
	// Set the kernelpar range with the dimension
	if ( this.kernel == "rbf" ) {
		var saveKpGrid = zeros(this.parameterGrid.kernelpar.length);
		for ( var kp = 0; kp < this.parameterGrid.kernelpar.length ; kp ++) {
			saveKpGrid[kp] = this.parameterGrid.kernelpar[kp];
			if ( typeof(this.kernelpar) == "undefined")
				this.parameterGrid.kernelpar[kp] *= Math.sqrt( X.n ); 			
		}
		if ( typeof(this.kernelpar) != "undefined")
			this.parameterGrid.kernelpar = mul(this.kernelpar, range(1.4,0.7,-0.1)); 
	}
	
	
	if ( arguments.length == 4 ) {
		// validation set (Xv, labelsv)
		
		if ( this.kernel == "linear" ) {
			// test all values of C 
			var validationErrors = zeros(this.parameterGrid.C.length);
			var minValidError = Infinity;
			var bestC;
			
			for ( var c = 0; c < this.parameterGrid.C.length; c++) {
				this.C = this.parameterGrid.C[c];
				this.train(X,labels);
				validationErrors[c] = 1.0 - this.test(Xv,labelsv);
				if ( validationErrors[c] < minValidError ) {
					minValidError = validationErrors[c];
					bestC = this.C;					
				}
			}
			this.C = bestC;
			
			this.train(mat([X,Xv]), mat([labels,labelsv]) ); // retrain with best values and all data
		}
		else {
			// grid of ( kernelpar, C) values
			var validationErrors = zeros(this.parameterGrid.kernelpar.length, this.parameterGrid.C.length);
			var minValidError = Infinity;
			
			var bestkernelpar;
			var bestC;
			
			var kc = new kernelCache( X , this.kernel, this.parameterGrid.kernelpar[0] ); 

			for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
				this.kernelpar = this.parameterGrid.kernelpar[kp];
				if ( kp > 0 ) {
					kc.update( this.kernelpar );
				}
				for ( var c = 0; c < this.parameterGrid.C.length; c++) {
					this.C = this.parameterGrid.C[c];
					this.train(X,labels, kc);	// use the same kernel cache for all values of C
					validationErrors.set(kp,c, 1.0 - this.test(Xv,labelsv) );
					if ( validationErrors.get(kp,c) < minValidError ) {
						minValidError = validationErrors.get(kp,c);
						bestkernelpar = this.kernelpar;
						bestC = this.C;
					}
				}
			}
			this.kernelpar = bestkernelpar;
			this.C = bestC;
			this.train(mat([X,Xv], true), mat([labels,labelsv], true) ); // retrain with best values and all data
		}				
	}
	else {
		
		// 5-fold Cross validation
		const nFolds = 5;
	
		const N = labels.length;
		const foldsize = Math.floor(N / nFolds);
	
		// Random permutation of the data set
		var perm = randperm(N);
	
		// Start CV
		if ( this.kernel == "linear" ) 
			var validationErrors = zeros(this.parameterGrid.C.length);
		else 
			var validationErrors = zeros(this.parameterGrid.kernelpar.length,this.parameterGrid.C.length);
		
	
		var Xtr, Ytr, Xte, Yte;
		var i;
		var fold;
		for ( fold = 0; fold < nFolds - 1; fold++) {
			console.log("fold " + fold);
			Xte = get(X, get(perm, range(fold * foldsize, (fold+1)*foldsize)), []);
			Yte = get(labels, get(perm, range(fold * foldsize, (fold+1)*foldsize)) );
		
			var tridx = new Array();
			for (i=0; i < fold*foldsize; i++)
				tridx.push(perm[i]);
			for (i=(fold+1)*foldsize; i < N; i++)
				tridx.push(perm[i]);
		
			Xtr =  get(X, tridx, []);
			Ytr = get(labels, tridx);

			
			if ( this.kernel == "linear" ) {
				// test all values of C 		
				for ( var c = 0; c < this.parameterGrid.C.length; c++) {
					this.C = this.parameterGrid.C[c];
					console.log("training with C = " + this.C); // + " on " , tridx, Xtr, Ytr);
					this.train(Xtr,Ytr);
					validationErrors[c] += 1.0 - this.test(Xte,Yte) ;					
				}
			}
			else {
				// grid of ( kernelpar, C) values
			
				var kc = new kernelCache( Xtr , this.kernel, this.parameterGrid.kernelpar[0] ); 

				for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
					this.kernelpar = this.parameterGrid.kernelpar[kp];
					if ( kp > 0 ) {
						kc.update( this.kernelpar );
					}
					for ( var c = 0; c < this.parameterGrid.C.length; c++) {
						this.C = this.parameterGrid.C[c];
						console.log("Training with kp = " + this.kernelpar + " C = " + this.C);
						
						// alpha seeding: intialize alpha with optimal values for previous (smaller) C
						/* (does not help with values of C too different...
							if ( c == 0 )
								this.alphaseeding.use = false; 
							else {
								this.alphaseeding.use = true;
								this.alphaseeding.alpha = this.alpha; 
							}
						*/
						this.train(Xtr,Ytr, kc);	// use the same kernel cache for all values of C
						validationErrors.val[kp * this.parameterGrid.C.length + c] += 1.0 - this.test(Xte,Yte) ;						
					}
				}
			}
		}
		console.log("fold " + fold);
		// last fold:
		Xtr = get(X, get(perm, range(0, fold * foldsize)), []);
		Ytr = get(labels, get(perm, range(0, fold * foldsize ) ) ); 
		Xte = get(X, get(perm, range(fold * foldsize, N)), []);
		Yte = get(labels, get(perm, range(fold * foldsize, N)) );
		
		if ( this.kernel == "linear" ) {
			// test all values of C 		
			for ( var c = 0; c < this.parameterGrid.C.length; c++) {
				this.C = this.parameterGrid.C[c];
				console.log("training with C = " + this.C); 
				this.train(Xtr,Ytr);
				validationErrors[c] += 1.0 - this.test(Xte,Yte) ;
			}
		}
		else {
			// grid of ( kernelpar, C) values
	
			var kc = new kernelCache( Xtr , this.kernel, this.parameterGrid.kernelpar[0] ); 

			for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
				this.kernelpar = this.parameterGrid.kernelpar[kp];
				if ( kp > 0 ) {
					kc.update( this.kernelpar );
				}
				for ( var c = 0; c < this.parameterGrid.C.length; c++) {
					this.C = this.parameterGrid.C[c];
					console.log("Training with kp = " + this.kernelpar + " C = " + this.C);	
					// alpha seeding: intialize alpha with optimal values for previous (smaller) C
					/*
					if ( c == 0 )
						this.alphaseeding.use = false; 
					else {
						this.alphaseeding.use = true;
						this.alphaseeding.alpha = this.alpha; 
					}*/
				
					this.train(Xtr,Ytr, kc);	// use the same kernel cache for all values of C
					validationErrors.val[kp * this.parameterGrid.C.length + c] += 1.0 - this.test(Xte,Yte) ;					
				}
			}
		}		
	
		// Compute Kfold errors and find best parameters 
		var minValidError = Infinity;
		var bestC;
		var bestkernelpar;
		
		if ( this.kernel == "linear" ) {
			for ( var c = 0; c < this.parameterGrid.C.length; c++) {
				validationErrors[c] /= nFolds; 
				if ( validationErrors[c] < minValidError ) {
					minValidError = validationErrors[c]; 
					bestC = this.parameterGrid.C[c];
				}
			}
			this.C = bestC;
		}
		else {
			// grid of ( kernelpar, C) values
			for ( var kp = 0; kp < this.parameterGrid.kernelpar.length; kp++) {
				for ( var c = 0; c < this.parameterGrid.C.length; c++) {
					validationErrors.val[kp * this.parameterGrid.C.length + c] /= nFolds;				
					if(validationErrors.val[kp * this.parameterGrid.C.length + c] < minValidError ) {
						minValidError = validationErrors.val[kp * this.parameterGrid.C.length + c]; 
						bestC = this.parameterGrid.C[c];
						bestkernelpar = this.parameterGrid.kernelpar[kp];
					}
				}
			}
			this.C = bestC;	
			this.kernelpar = bestkernelpar;	
		}
		
		//this.alphaseeding.use = false;
		
		// Retrain on all data
		this.train(X, labels);		
	}
	
	// Restore the dimension-free kernelpar range 
	if ( this.kernel == "rbf" ) {
		for ( var kp = 0; kp < this.parameterGrid.kernelpar.length ; kp ++) {
			this.parameterGrid.kernelpar[kp] = saveKpGrid[kp];
		}
	}
	
	this.validationError = minValidError; 
	return {error: minValidError, validationErrors: validationErrors}; 
}
MSVM.prototype.train = function (X, labels) {
	// Training function
	
	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels ) ;
	
	// Check multi-class case
	if ( this.labels.length <= 2 ) {		
		return "The data set should contain more than 2 classes for M-SVMs."; 
	}
	
	const C = this.C;
	
	var alpha;

	var chunk;
	var K;
	var i;
	var k;
	var j;
	var l;
	
	var gradient;
	var gradient_update_ik;
	var H_alpha;
	var delta;
	var theta;
	var y_i;
	var y_j;
	var partial;
	var alpha_update;
	var lp_rhs;
	
	// Initialization
	const Q = this.labels.length;
	const N = X.length;
	if ( this.chunk_size > N ) {
		this.chunk_size = N;
	}
	const chunk_size = this.chunk_size;
	var chunk_vars;	// variables indexes in alpha
	
	switch ( this.MSVMtype ) {

		case "CS":
			alpha = zeros(N*Q);	// vectorized
			gradient = zeros(N*Q);
			for ( i=0; i < N; i++ ) {
				alpha[i*Q + y[i]] = C;
				gradient[i*Q + y[i]] = 1;
			}
			H_alpha = zeros(N*Q);
			break;
		case "LLW":
			alpha = zeros(N*Q);	// vectorized			
			gradient = mulScalarVector(-1.0/(Q-1.0), ones(N*Q));
			for ( i=0; i < N; i++ ) 
				gradient[i*Q + y[i]] = 0;
			H_alpha = zeros(N*Q);
			
			lp_rhs = zeros(Q-1);
			break;
		case "WW":
		default:
			alpha = zeros(N*Q);	// vectorized
			gradient = minus(ones(N*Q));
			for ( i=0; i < N; i++ ) 
				gradient[i*Q + y[i]] = 0;
			H_alpha = zeros(N*Q);
			
			lp_rhs = zeros(Q-1);
			break;		
	}
	
	// create a new kernel cache
	if ( typeof(kc) == "undefined" ) {
		// create a new kernel cache if it is not provided
		var kc = new kernelCache( X , this.kernel, this.kernelpar ); 
	}
	K = new Array(); // to store kernel rows
	
	// Create function that updates the gradient
	var update_gradient; 
	switch ( this.MSVMtype ) {
	case "LLW":
		update_gradient = function ( chunk, alpha_update ) {
			var i,y_i,k,gradient_update_ik,j,y_j,partial,l,s,e;
			for(i=0; i < N; i++) {
				y_i = y[i];

				for(k=0; k< Q; k++) {
					if(k != y_i) {
						gradient_update_ik = 0.0;
						for(j=0; j< chunk_size; j++) {
							y_j = y[chunk[j]];

							//partial = - sumVector( getSubVector(alpha_update, range(j*Q, (j+1)*Q) ) ) / Q;
							partial = 0;
							s = j*Q;
							e = s + Q;
							for ( l=s; l < e; l++)
								partial -= alpha_update[l];
							partial /= Q;
					
							partial +=  alpha_update[j*Q + k];
					
							// Use the symmetry of K: Kglobal[i][chunk[j]] = Kglobal(chunk[j], i) = K[j][i]
							gradient_update_ik += partial * K[j][i];
						}
			  			l = i*Q+k;
						gradient[l] += gradient_update_ik;
						H_alpha[l] += gradient_update_ik;
			  		}
				}
			}
		};
		break;
	case "CS":
	case "WW":
	default:
		update_gradient = function ( chunk, alpha_update ) {
			var i,y_i,k,gradient_update_ik,j,y_j,partial,l,s,e;
			for(i=0; i < N; i++) {
				y_i = y[i];

				for(k=0; k< Q; k++) {
					if(k != y_i) {
						gradient_update_ik = 0.0;
						for(j=0; j< chunk_size; j++) {
							y_j = y[chunk[j]];

							partial = 0.0;
							s = j*Q;
							e = s + Q;
							
							//if(y_j == y_i )
								//partial += sumVector( getSubVector(alpha_update, range(j*Q, (j+1)*Q) ) );
							if(y_j == y_i ) {
								for ( l=s; l < e; l++)
									partial += alpha_update[l];
							}
							//if(y_j == k )
								//partial -= sumVector( getSubVector(alpha_update, range(j*Q, (j+1)*Q) ) );

							if(y_j == k ) {
								for ( l=s; l < e; l++)
									partial -= alpha_update[l];
							}

							partial += alpha_update[s+k] - alpha_update[s + y_i] ;
							// Use the symmetry of K: Kglobal[i][chunk[j]] = Kglobal(chunk[j], i) = K[j][i]
							gradient_update_ik += partial * K[j][i];
						}
			  			l = i*Q+k;
						gradient[l] += gradient_update_ik;
						H_alpha[l] += gradient_update_ik;
			  		}
				}
			}
		};
		break;
	}
	
	// Main loop
	var info = {};
	info.primal = Infinity;
	var ratio = 0;
	var iter = 0;
	var ratio_10k = -1; 
	var ratio_stable_10k = false;
	
	var infoStep ;
	if ( this.MSVMtype == "CS")
		infoStep = Math.floor(1000/chunk_size); 
	else
		infoStep = Math.floor(10000/chunk_size); 
		
	if ( infoStep > 1000 )
		infoStep = 1000;
	if ( infoStep < 100 )
		infoStep = 100;
	
	tic();
	
	do {
		// Working set selection: chunk = data indexes	
		if ( this.MSVMtype == "CS" && iter > 0.2*N/chunk_size && (iter % 100 < 80) ) {
			chunk = MSVM_CS_workingset_selection(chunk_size, N, Q, alpha, gradient); 
		}
		else {
			chunk = MSVMselectChunk(chunk_size,N) ; // random selection for all others	
		}
		
		chunk_vars = [];
		for ( i=0; i < chunk_size; i++) {
			for ( k=0; k< Q; k++)
				chunk_vars.push(chunk[i]*Q + k);
		}

		// Compute kernel submatrix
		for ( i=0; i < chunk_size; i++) {
			K[i] = kc.get_row( chunk[i] ) ;
		}
		
		// solve LP for Frank-Wolfe step
		switch( this.MSVMtype ) {
			case "CS":
				delta = MSVM_CS_lp(Q,C, y, alpha, gradient, chunk ,chunk_vars) ;
				break;
			case "LLW":
				delta = MSVM_LLW_lp(Q,C, y, alpha, gradient, chunk ,chunk_vars, lp_rhs) ;
				break;
			case "WW":
			default:
				delta = MSVM_WW_lp(Q,C, y, alpha, gradient, chunk ,chunk_vars, lp_rhs) ;
				break;
		}

		if ( typeof(delta) != "undefined" && norm0(delta) > 0 ) {
		
			// Step length
			switch( this.MSVMtype ) {
				case "LLW":
					theta = MSVM_LLW_steplength(Q,chunk, chunk_size, chunk_vars, y, delta, K, gradient);
					break;
				case "CS":
				case "WW":
				default:
					theta = MSVM_steplength(Q,chunk, chunk_size, chunk_vars, y, delta, K, gradient);
					break;
			}

			if ( theta > 1e-10 ) {
				// Take the step
				alpha_update = mulScalarVector( -theta, delta );				
				//set ( alpha, chunk_vars, add( getSubVector(alpha, chunk_vars), alpha_update) );
				for(k=0; k<chunk_vars.length; k++) {
					alpha[chunk_vars[k]] += alpha_update[k]; 
				}
				
				// Update RHS of constraints in LP (dense format)
				switch( this.MSVMtype ) {
				case "CS":
					break;
				case "LLW":
					for(i=0; i<chunk_size; i++) {	  
						for(k=0; k<Q-1; k++) {	  
							for(l=i*Q; l<(i+1)*Q; l++)
								 lp_rhs[k] += alpha_update[l]; 
							lp_rhs[k] -= Q * alpha_update[i*Q + k];
						}
					}
					break;
				case "WW":
				default:
					for(i=0; i<chunk_size; i++) {	  
						for(k=0; k<Q-1; k++) {	  
							for(l=0; l< Q; l++) {
								if((y[chunk[i]] == k) && (l != k))
									lp_rhs[k] += alpha_update[i*Q + l];
								else if((y[chunk[i]] != k) && (l == k))
									lp_rhs[k] -= alpha_update[i*Q + l];
							}
						}
					}
					break;
				}
				
				// Update gradient 
				update_gradient( chunk, alpha_update ) ; 
				
				//	console.log(H_alpha, alpha, gradient, alpha_update);
			}			
		}
			
		if ( iter % infoStep == 0 ) {
			// Evaluate optimization accuracy
			switch( this.MSVMtype) {
				case "CS":
					info = MSVM_CS_evaloptim(Q,C, y, alpha, gradient, H_alpha, info);
					break;
				case "LLW":
					info = MSVM_LLW_evaloptim(Q,C, y, alpha, gradient, H_alpha, info);
					break;
				case "WW":
				default:
					info = MSVM_WW_evaloptim(Q,C, y, alpha, gradient, H_alpha, info);
					break;
			}

			if ( isFinite(info.primal) )
				ratio = info.dual / info.primal; 
			else
				ratio = 0;
				
			if ( iter % 10000 == 0 ) {
				ratio_stable_10k = (Math.abs(ratio - ratio_10k) < 1e-3 );
				ratio_10k = ratio;
			}
			//console.log("iter " + iter + " Remp=" + info.trainingError.toFixed(4) + " ratio=" + info.dual.toFixed(4) + "/" + info.primal.toFixed(4) + "=" + (100*ratio).toFixed(4) + " %");
			console.log("iter " + iter + ": time=" + toc() + " Remp=" + info.trainingError.toFixed(4) + " ratio= " + (100*ratio).toFixed(4) + " %");
		}
		
		
		iter++;
	} while (ratio < this.optimAccuracy && iter < this.maxIter && !ratio_stable_10k ); 
	
	// Set model parameters
	this.alpha = zeros(Q,N);
	var isSV = zeros(N);
	for ( i=0; i<N;i++) {
		for ( k=0; k < Q; k++) {
			if ( !isZero( alpha[i*Q+k] ) ) {
				this.alpha.val[k*N+i] = alpha[i*Q+k];
				
				if (this.MSVMtype != "CS" || k != y[i])
					isSV[i] = 1;
			}
		}
	}
	this.SVindexes = find(isNotEqual(isSV, 0) );
	this.SV = get(X,this.SVindexes, []);
	this.SVlabels = get(y,this.SVindexes);
	
	if ( this.MSVMtype == "CS" ) 
		this.b = zeros(Q);
	else
		this.b = info.b;
	
	this.dim_input = size(X,2);
	
	this.kernelpar = kc.kernelpar;
	if ( this.dim_input > 1 )
		this.kernelFunc = kc.kernelFunc;	
	else
		this.kernelFunc = kernelFunction(this.kernel, this.kernelpar, "number"); // for scalar input kernelcache uses 1D-vectors
	this.sparseinput = (kc.inputtype == "spvector"); 
	
	
	/* and return training error rate:
	return info.trainingError;*/
	return this.info();
}
// MSVM tools
function MSVMselectChunk ( chunk_size, N ) {
	var r = randperm(N);
	return get(r, range(chunk_size) ); 
}
function MSVM_CS_workingset_selection (chunk_size, N, Q, alpha, gradient) {
	/*
	
	select points with maximal violation of the KKT condition
	as measured by

	psi_i = max_{k, alpha_ik>0} gradient_ik   -  min_k gradient_ik

	*/
	var i;
	var j;
	var l;
	var psi = zeros(N);
	var chunk = new Array();
	var alpha_i_pos;
	var grad_i;
	
	for ( i=0; i < N; i++) {
		/*
		alpha_i_pos = find( isGreater( getSubVector(alpha, range(i*Q, (i+1)*Q)) , 0.001) );
		grad_i = getSubVector(gradient, range(i*Q, (i+1)*Q));
		if ( alpha_i_pos.length > 0 )
			psi[i] = max(getSubVector(grad_i, alpha_i_pos)) - min(grad_i);
		else
			return undefined; // should not happen due to sum_k alpha_ik = C
		*/
		var maxg = -Infinity;
		var ming = +Infinity;
		for (l=i*Q; l < (i+1)*Q; l++) {
			if ( alpha[l] > 0.001 && gradient[l] > maxg)
				maxg = gradient[l];
			if ( gradient[l] < ming )
				ming = gradient[l];
		}
		psi[i] = maxg - ming;

		// chunk= list of data indexes with ordered psi values
		j = 0;
		while ( j < chunk.length && psi[i] < psi[chunk[j]] ) {
			j++;	
		}
		if ( j < chunk.length ) {
			chunk.splice(j,0,i );
			while ( chunk.length > chunk_size ) 
				chunk.pop();
		}
		else if ( j < chunk_size ) {
			chunk.push(i);
		}
	}
	/*if ( psi[chunk[0] ] < 0.01 ) 
		console.log("psimax: " + psi[chunk[0]]);*/
	return chunk;
}

function MSVM_WW_lp (Q,C, y, alpha, gradient, chunk,chunk_vars, lp_rhs) {

	const chunk_size = chunk.length;

	var rhs;
	var lp_A;
	var col;
	var lp_sol;
	var lp_sol_table;
	var lp_sol_table_inv;
	
	const lp_nCols = (Q-1)*chunk_size;
	var lp_cost = zeros(lp_nCols);
	var lp_low = zeros(lp_nCols);
	var lp_up = mulScalarVector(C,  ones(lp_nCols));

	var i;
	var k;
	var l;
	var y_i;
	
	// objective function
	col = 0;
	lp_sol_table_inv = zeros(chunk_size, Q);
	lp_sol_table = zeros(lp_nCols,2);
	for(i=0; i<chunk_size; i++) {
		y_i = y[chunk[i]];
		for(k=0; k< Q; k++) {
		    if(k != y_i) {
      			lp_cost[col] = gradient[chunk[i]*Q + k];
				lp_sol_table.val[col*2] = i;	// keep a table of correspondance between
				lp_sol_table.val[col*2+1] = k;	// LP vector of variables and lp_sol matrix
				lp_sol_table_inv.val[i*Q+k] = col; // lp_sol[i][k] = the 'lp_solve_table_inv[i][k]'-th variable for LPSOLVE		
				col++;
    		}
    	}
  	}
  		// Make RHS of constraints
		// -- updates to cache->rhs are made in compute_new_alpha()
		//    to keep track of rhs
		//    we only need to remove the contribution of the examples in the chunk
	rhs = vectorCopy( lp_rhs );
	
	for(k=0; k < Q-1; k++) {
		for(i=0; i<chunk_size; i++) {			
			y_i = y[chunk[i]];
			for(l=0; l< Q; l++){
				if((y_i == k) && (l != k))
					rhs[k] -= alpha[chunk[i]*Q + l];
				else if((y_i != k) && (l == k))
					rhs[k] += alpha[chunk[i]*Q + l];
			}
		}
	}
	  	// Make constraints
	lp_A = zeros(Q-1, lp_nCols );
	for(k=0; k < Q-1; k++) {

		for(i=0; i< chunk_size; i++) {
		    y_i = y[chunk[i]];

		    if(y_i == k) {
		    	for(l=0; l< Q; l++) {
					if(l != k) 
						lp_A.val[k*lp_nCols + lp_sol_table_inv.val[i*Q+l]] = -1.0;
	  			}
			}	     
			else
			  lp_A.val[k*lp_nCols + lp_sol_table_inv.val[i*Q+k]] = +1.0;	      
    	}
  	}
	
		// solve
	lp_sol = lp( lp_cost, [], [], lp_A, rhs, lp_low, lp_up ) ;

	if (typeof(lp_sol) != "string") {
			// get direction from lp solution		
		var direction = zeros(chunk_size * Q);
		for ( col=0; col < lp_nCols; col++) {
			if ( lp_sol[col] < -EPS || lp_sol[col] > C + EPS ) 
				return undefined; // infeasible
				
			if ( lp_sol[col] > EPS ) 
				direction[lp_sol_table.val[col*2] * Q + lp_sol_table.val[col*2+1]] = lp_sol[col];
			else if ( lp_sol[col] > C - EPS )
				direction[lp_sol_table.val[col*2] * Q + lp_sol_table.val[col*2+1]] = C;								

		}
		
		var delta = subVectors ( getSubVector(alpha, chunk_vars) , direction );

		return delta;
	}
	else {
		return undefined; 
	}
}
function MSVM_LLW_lp (Q,C, y, alpha, gradient, chunk,chunk_vars, lp_rhs) {

	const chunk_size = chunk.length;

	var rhs;
	var lp_A;
	var col;
	var lp_sol;
	var lp_sol_table;
	var lp_sol_table_inv;
	
	const lp_nCols = Q*chunk_size;
	const lp_nRows = (Q-1);
	var lp_cost = zeros(lp_nCols);
	var lp_low = zeros(lp_nCols);
	var lp_up = mulScalarVector(C,  ones(lp_nCols));

	var i;
	var k;
	var l;
	var y_i;
	var alpha_i;
	
	// objective function
	col = 0;
	lp_sol_table_inv = zeros(chunk_size, Q);
	lp_sol_table = zeros(lp_nCols,2);
	for(i=0; i<chunk_size; i++) {
		y_i = y[chunk[i]];
		for(k=0; k< Q; k++) {
  			lp_cost[col] = gradient[chunk[i]*Q + k];
			lp_sol_table.val[col*2] = i;	// keep a table of correspondance between
			lp_sol_table.val[col*2+1] = k;	// LP vector of variables and lp_sol matrix
			lp_sol_table_inv.val[i*Q+k] = col; // lp_sol[i][k] = the 'lp_solve_table_inv[i][k]'-th variable for LPSOLVE		
			col++;
    	}
  	}
  		// Make RHS of constraints
		// -- updates to cache->rhs are made in compute_new_alpha()
		//    to keep track of rhs
		//    we only need to remove the contribution of the examples in the chunk
	rhs = vectorCopy( lp_rhs );
	
	for(k=0; k < Q-1; k++) {
		for(i=0; i<chunk_size; i++) {			
			y_i = y[chunk[i]];
			alpha_i = alpha.subarray(chunk[i]*Q, (chunk[i]+1)*Q);
			
			rhs[k] -= sumVector(alpha_i);
			rhs[k] += Q * alpha_i[k];
		}
	}
	  	// Make constraints
	lp_A = zeros(lp_nRows, lp_nCols );
	for(k=0; k < lp_nRows; k++) {

		for(i=0; i< chunk_size; i++) {
		    y_i = y[chunk[i]];

	    	for(l=0; l< Q; l++) {
				if(l != y_i) {
					if ( l == k )
						lp_A.val[k*lp_nCols + lp_sol_table_inv.val[i*Q+l]] = Q - 1.0 ;
					else
						lp_A.val[k*lp_nCols + lp_sol_table_inv.val[i*Q+l]] = -1.0 ;
				}	  			
			}	     			
    	}
  	}
	
		// solve
	lp_sol = lp( lp_cost, [], [], lp_A, rhs, lp_low, lp_up ) ;

	if (typeof(lp_sol) != "string") {
			// get direction from lp solution		
		var direction = zeros(chunk_size * Q);
		for ( col=0; col < lp_nCols; col++) {
			if ( lp_sol[col] < -EPS || lp_sol[col] > C + EPS ) 
				return undefined; // infeasible
				
			if ( lp_sol[col] > EPS ) 
				direction[lp_sol_table.val[col*2] * Q + lp_sol_table.val[col*2+1]] = lp_sol[col];
			else if ( lp_sol[col] > C - EPS )
				direction[lp_sol_table.val[col*2] * Q + lp_sol_table.val[col*2+1]] = C;								

		}
		
		var delta = subVectors ( getSubVector(alpha, chunk_vars) , direction );

		return delta;
	}
	else {
		return undefined; 
	}
}

function MSVM_CS_lp (Q,C, y, alpha, gradient, chunk,chunk_vars) {

	const chunk_size = chunk.length;
	
	var col;
	var lp_sol;
	var lp_sol_table;
	var lp_sol_table_inv;
	
	const lp_nCols = Q*chunk_size;
	const lp_nRows = chunk_size;
	var lp_cost = zeros(lp_nCols);
	var lp_low = zeros(lp_nCols);
//	var lp_up = mul(C,  ones(lp_nCols));// implied by equality constraints, but set anyway...
	var rhs;
	var lp_A;

	var i;
	var k;
	var l;
	var y_i;
	
	// objective function
	col = 0;
	lp_sol_table_inv = zeros(chunk_size, Q);
	lp_sol_table = zeros(lp_nCols,2);
	for(i=0; i<chunk_size; i++) {
		for(k=0; k< Q; k++) {
  			lp_cost[col] = gradient[chunk[i]*Q + k];
			lp_sol_table.val[col*2] = i;	// keep a table of correspondance between
			lp_sol_table.val[col*2+1] = k;	// LP vector of variables and lp_sol matrix
			lp_sol_table_inv.val[i*Q+k] = col; // lp_sol[i][k] = the 'lp_solve_table_inv[i][k]'-th variable for LPSOLVE		
			col++;
    	}
  	}
  	
  		// Make constraints : forall i,  sum_k alpha_ik = C_yi
	rhs = mulScalarVector(C, ones(chunk_size) );
	lp_A = zeros(lp_nRows, lp_nCols);
	for(i=0; i<chunk_size; i++) {
		//set(lp_A, i, range(i*Q, (i+1)*Q), ones(Q) );
		for ( l = i*Q; l < (i+1)*Q; l++)
			lp_A.val[lp_nCols*i + l] = 1;
	}
	
		// solve
	lp_sol = lp( lp_cost, [], [], lp_A, rhs, lp_low ) ;

	if (typeof(lp_sol) != "string") {
			// get direction from lp solution		
		var direction = zeros(chunk_size * Q);
		for ( col=0; col < lp_nCols; col++) {
			if ( lp_sol[col] < -EPS || lp_sol[col] > C + EPS ) 
				return undefined; // infeasible
				
			if ( lp_sol[col] > EPS ) 
				direction[lp_sol_table.val[col*2] * Q + lp_sol_table.val[col*2+1]] = lp_sol[col];
			else if ( lp_sol[col] > C - EPS )
				direction[lp_sol_table.val[col*2] * Q + lp_sol_table.val[col*2+1]] = C;								
		}
		
		var delta = subVectors ( getSubVector(alpha, chunk_vars) , direction );
		
		return delta;
	}
	else {
		return undefined; 
	}
}

function MSVM_steplength (Q,chunk, chunk_size, chunk_vars, y, delta, K, gradient){
	var Hdelta = zeros(Q*chunk_size);
	var i;
	var j;
	var k;
	var y_i;
	var partial;
	var l;
	var s;
	var e;
	
	for ( i=0;i<chunk_size; i++) {
		y_i = y[chunk[i]];
		for ( k=0; k < Q; k++) {
			 if(k != y_i ) {
	  			for(j=0; j<chunk_size; j++) {
					partial = 0.0;
					s = j*Q;
					e = s + Q;
					
					if( y[chunk[j]] == y_i ) {
						//partial += sumVector( getSubVector(delta, range(j*Q, (j+1)*Q) ) );
						for ( l=s; l < e; l++)
							partial += delta[l]; 
					}
					if( y[chunk[j]] == k ) {
						//partial -=  sumVector( getSubVector(delta, range(j*Q, (j+1)*Q) ) );
						for ( l=s; l < e; l++)
							partial -= delta[l]; 
					}
				
					partial += delta[s + k] - delta[s + y_i];
					Hdelta[i*Q + k] += partial * K[i][chunk[j]];
				}
			}
		}
	}

  	var den = dot(delta, Hdelta);
  	var theta;
	if ( den < EPS ) 
		theta = 0;
	else  {
		theta = dot(getSubVector ( gradient, chunk_vars) , delta ) / den; 

		if (theta > 1 ) 
			theta = 1;
	}
	return theta;
}
function MSVM_LLW_steplength (Q,chunk, chunk_size, chunk_vars, y, delta, K, gradient){
	var Hdelta = zeros(Q*chunk_size);
	var i;
	var j;
	var k;
	var y_i;
	var partial;
	var l,s,e;
	
	for ( i=0;i<chunk_size; i++) {
		y_i = y[chunk[i]];
		for ( k=0; k < Q; k++) {
			 if(k != y_i ) {
	  			for(j=0; j<chunk_size; j++) {
					
					//partial = -sumVector( getSubVector(delta, range(j*Q, (j+1)*Q) ) ) / Q;
					partial = 0.0;
					s = j*Q;
					e = s + Q;
					for ( l=s; l < e; l++)
							partial -= delta[l]; 
					partial /= Q;
							
					partial += delta[s+k];
					
					Hdelta[i*Q + k] += partial * K[i][ chunk[j] ];
				}
			}
		}
	}

  	var den = dot(delta, Hdelta);
  	var theta;
	if ( den < EPS ) 
		theta = 0;
	else  {
		theta = dot(getSubVector ( gradient, chunk_vars) , delta ) / den; 

		if (theta > 1 ) 
			theta = 1;
	}
	return theta;
}
function MSVM_WW_evaloptim (Q,C, y, alpha, gradient, H_alpha, info) {
	// Evaluate accuracy of the optimization while training an MSVM

	var i;
	const N = y.length;
	
	// Estimate b
	var b = MSVM_WW_estimate_b(Q,C,y,alpha,gradient);
	
	// Compute R_emp
	var R_emp = 0;
	var trainingErrors = 0;
	var delta;
	for ( i=0; i < N; i++) {
		delta = addVectors(gradient.subarray(i*Q, (i+1)*Q), subScalarVector(b[y[i]], b) );
		delta[y[i]] = 0;
		
		R_emp -= sumVector( minVectorScalar(delta, 0) ) ;	

		if ( minVector(delta) <= -1 )
			trainingErrors++;
	}
	R_emp *= C;

	// Compute dual objective
	var alpha_H_alpha = dot(alpha, H_alpha) ;	// or H_alpha = add(gradient,1);

	var dual = -0.5 * alpha_H_alpha + sumVector(alpha);
	
	// Compute primal objective
	var primal = 0.5 * alpha_H_alpha + R_emp;
	
	if ( primal > info.primal ) 
		primal = info.primal; 
				
	return {primal: primal, dual: dual, Remp: R_emp, trainingError: trainingErrors / N, b: b};
}

function MSVM_WW_estimate_b(Q,C,y,alpha, gradient) {

	var i;
	var k;
	var l;

	const N = y.length;

	var delta_b = zeros(Q* Q);
	var nb_margin_vect = zeros(Q* Q);
	var b = zeros(Q);

	for(i=0; i<N; i++) {
		for(k=0; k<Q; k++) {
			if( alpha[i*Q+k] < C - 0.001*C && alpha[i*Q+k] > 0.001*C) {
				// margin boundary SV for class k
				delta_b[y[i]*Q+k] -= gradient[i*Q+k];
				nb_margin_vect[y[i] * Q + k] ++; 
			}
		}
	}
	
	for(k=0; k< Q; k++) {
		for(l=0; l < Q; l++) {
	    	if(k != l) {
			    if( nb_margin_vect[k*Q + l] > 0 ) {
					delta_b[k*Q + l] /= nb_margin_vect[k * Q + l];
				}
			}
		}
	}

	for(k=1; k< Q; k++) {
		var nb_k = nb_margin_vect[k * Q] + nb_margin_vect[k];
		if(nb_k > 0) {
			b[k] = (nb_margin_vect[k*Q] * delta_b[k*Q] - nb_margin_vect[k] * delta_b[k]) / nb_k;
	    }
	}
	
	// make sum(b) = 0; 
	b = subVectorScalar(b, sumVector(b) / Q );
	
	return b;
}

function MSVM_LLW_evaloptim (Q,C, y, alpha, gradient, H_alpha, info) {
	// Evaluate accuracy of the optimization while training an MSVM
	var i;
	const N = y.length;
	
	// Estimate b
	var b = MSVM_LLW_estimate_b(Q,C,y,alpha,gradient);
	
	// Compute R_emp
	var R_emp = 0;
	var trainingErrors = 0;
	var output;
	var xi_i;

	for ( i=0; i < N; i++) {
		output = subVectors(b, H_alpha.subarray(i*Q, (i+1)*Q) )
		output[y[i]] = 0;
		output[y[i]] = -sumVector(output);

		xi_i = subVectors(b,  gradient.subarray(i*Q, (i+1)*Q) )
		xi_i[y[i]] = 0;
		
		R_emp += sumVector( maxVectorScalar ( xi_i, 0) );

		if ( argmax(output) != y[i] )
			trainingErrors++;
	}
	R_emp *= C;

	// Compute dual objective
	var alpha_H_alpha = dot(alpha, H_alpha) ;

	var dual = -0.5 * alpha_H_alpha + (sumVector(alpha) / (Q-1));

	// Compute primal objective
	var primal = 0.5 * alpha_H_alpha + R_emp;
	
	if ( primal > info.primal ) 
		primal = info.primal; 
				
	return {primal: primal, dual: dual, Remp: R_emp, trainingError: trainingErrors / N, b: b};
}
function MSVM_LLW_estimate_b(Q,C,y,alpha, gradient) {

	var i;
	var k;
	var l;

	const N = y.length;

	var nb_margin_vect = zeros(Q);
	var b = zeros(Q);

	if ( maxVector(alpha) <= EPS )
		return b;

	for(i=0; i<N; i++) {
		for(k=0; k<Q; k++) {
			if( k != y[i] && alpha[i*Q+k] < C - 0.001*C && alpha[i*Q+k] > 0.001*C) {
				// margin boundary SV for class k
				b[k] += gradient[i*Q+k];
				nb_margin_vect[k] ++; 
			}
		}
	}

	for(k=0; k< Q; k++) {
		if( nb_margin_vect[k] > 0 ) {
			b[k] /= nb_margin_vect[k];
		}
	}

	// make sum(b) = 0; 
	b = subVectorScalar(b, sumVector(b) / Q );

	return b;
}

function MSVM_CS_evaloptim (Q,C, y, alpha, gradient, H_alpha, info) {
	// Evaluate accuracy of the optimization while training an MSVM
	var i;
	const N = y.length;
	
	// Compute R_emp
	var R_emp = 0;
	var trainingErrors = 0;

	var xi = maxVectorScalar( subScalarVector(1, H_alpha) , 0 );

	var ximax ;
	var sum_alpha_iyi = 0;
	
	for ( i=0; i < N; i++) {
		xi[i*Q + y[i]] = 0;
		
		ximax = maxVector( xi.subarray( i*Q, (i+1)*Q) );
		
		R_emp += ximax;	

		if ( ximax > 1 )
			trainingErrors++;
			
		sum_alpha_iyi += alpha[i*Q + y[i] ];
	}
	R_emp *= C;

	// Compute dual objective
	var alpha_H_alpha = dot(alpha, H_alpha) ;

	var dual = -0.5 * alpha_H_alpha + C*N - sum_alpha_iyi;
	
	// Compute primal objective
	var primal = 0.5 * alpha_H_alpha + R_emp;
	
	if ( primal > info.primal ) 
		primal = info.primal; 
				
	return {primal: primal, dual: dual, Remp: R_emp, trainingError: trainingErrors / N};
}

MSVM.prototype.predict = function ( x ) {

	var scores = this.predictscore( x );
	if (typeof(scores) != "undefined") {
		if ( this.single_x( x ) ) {		
			// single prediction
			return this.recoverLabels( argmax( scores ) );
		}
		else {
			// multiple predictions for multiple test data
			var i;
			var y = new Float64Array(x.length );
			for ( i = 0; i < x.length; i++)  {
				y[i] = findmax ( scores.row(i) ) ;
			}
			return this.recoverLabels( y );
		}		
	}
	else
		return "undefined";	
}
MSVM.prototype.predictscore = function( x ) {


	const Q = this.labels.length;
	const N = size(this.alpha,2);

	if ( this.single_x( x ) ) {		
	
		var output = vectorCopy(this.b);
		var i;
		var k;
		var range_k;
		var partial;
		
		for(i =0; i< this.SVindexes.length; i++ ) {
			
			//partial = sumVector( get(alphaSV, i, []) )
			partial = 0;
			for (k=0; k< Q; k++)
				partial += this.alpha.val[k*N + this.SVindexes[i] ];
			
			var Ki = this.kernelFunc(this.SV.row(i), x);
			
			for (k=0; k< Q; k++) {				
			    
			    switch ( this.MSVMtype ) {
					case "CS":
					    if(this.SVlabels[i] == k) 
							output[k] += (partial - this.alpha.val[k*N + this.SVindexes[i] ]) * Ki ;	
						else
							output[k] -= this.alpha.val[k*N + this.SVindexes[i] ] * Ki;
						break;
					case "LLW":
					case "MSVM2":					
						output[k] += (partial / Q - this.alpha.val[k*N + this.SVindexes[i] ]) * Ki;
						break;
					case "WW":
					default:
					    if(this.SVlabels[i] == k) 
					    	output[k] += partial * Ki;
						else
							output[k] -= this.alpha.val[k*N + this.SVindexes[i] ] * Ki;
						break;					
				}
	 	    }
		}
		return output; 		// vector of scores for all classes
	}
	else {
		var xi;
		const m = x.length;
		var output = zeros(m, Q);

		var i;
		var k;
		var Kij;
		var partial;

		// Cache SVs
		var SVs = new Array(this.SVindexes.length);
		for ( var j=0; j < this.SVindexes.length; j++) 
			SVs[j] = this.SV.row(j);
		
		for (xi=0; xi < m; xi++) {
			for (k=0; k< Q; k++)
				output.val[xi*Q+k] = this.b[k];

			var Xi = x.row(xi);
			
			for(i =0; i< this.SVindexes.length; i++ ) {
				Kij = this.kernelFunc(SVs[i], Xi);
				
				//partial = sumVector( get(alphaSV, i, []) )
				partial = 0;
				for (k=0; k< Q; k++)
					partial += this.alpha.val[k*N + this.SVindexes[i] ];
			
				for (k=0; k< Q; k++) {				
			    
					switch ( this.MSVMtype ) {
						case "CS":
							if(this.SVlabels[i] == k) 
								output.val[xi*Q+k] += (partial - this.alpha.val[k*N + this.SVindexes[i] ]) * Kij;	
							else
								output.val[xi*Q+k] -= this.alpha.val[k*N + this.SVindexes[i] ] * Kij;
							break;
						case "LLW":
						case "MSVM2":
							output.val[xi*Q+k] += (partial / Q - this.alpha.val[k*N + this.SVindexes[i] ]) * Kij;
							break;
						case "WW":
						default:
							if(this.SVlabels[i] == k) 
								output.val[xi*Q+k] += partial * Kij;
							else
								output.val[xi*Q+k] -= this.alpha.val[k*N + this.SVindexes[i] ] * Kij;
							break;					
					}				
		 	    }								
			}
		}

		return output; // matrix of scores is of size Nsamples-by-Nclasses 
	}
}

//////////////////////////////////////////////////
/////	K-nearest neighbors
///////////////////////////////////////////////////
function KNN ( params ) {
	var that = new Classifier ( KNN, params);	
	return that;
}
KNN.prototype.construct = function (params) {

	// Default parameters:
	this.K = 3;	
	
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
	this.parameterGrid = { "K" : range(1,16) };		
}

KNN.prototype.train = function ( X, labels ) {
	// Training function: should set trainable parameters of the model
	//					  and return the training error rate.
	
	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels ) ; 

	if ( size(X,2) == 1 ) {
		this.X = matrixCopy(X);
		this.featuresOrder = [0];
		this.relevantFeatures = [0]; 		
	}
	else {
		var v = variance(X,1); 		
		this.featuresOrder = sort(v.val,true, true);

		var relevantFeatures = find ( isNotEqual(v.val, 0) ) ;
		this.X = getCols(X, getSubVector(this.featuresOrder, relevantFeatures) );
	}
	this.y = matrixCopy(labels); // KNN works directly on true labels
	
	/* Return training error rate:
	if ( labels.length < 2000)
		return (1 - this.test(X, labels));
		*/
	return this.info();
}
KNN.prototype.update = function ( X, Y ) {
	// Online training: add X,labels to training set
	if ( typeof(this.X) == "undefined" )
		return this.train(X, Y);
	
	this.X = mat([this.X, X], true);

	if ( this.single_x( X ) ) 
		var labels = [Y];
	else 
		var labels = Y;

	// Make y an Array (easier to push new labels
	if ( Array.isArray(this.y) ) 
		var y = this.y; 
	else
		var y = arrayCopy(this.y);

	for (var i = 0; i<labels.length; i++) {
		y.push( Y[i] );
		
		// update labels if adding new classes...
		var numericlbl = this.labels.indexOf( labels[i] );
		if(numericlbl < 0 ) {
			numericlbl = this.labels.length;
			this.labels.push( labels[i] );
			this.numericlabels.push( numericlbl );				
		}
	}
	this.y = y;
	
	return this.info();
}

KNN.prototype.predictslow = function ( x ) {
   	const N = this.X.length; 
   	if (K >  N) {
   		this.K = N;   		
   	}
   	const K = this.K;
   	if ( K == 0 ) {
   		return undefined;
   	}

	var tx = type ( x );
	var tX = type(this.X);
	if ( (tx == "vector" && tX == "matrix") || (tx == "number" && tX == "vector" ) ) {
		// Single prediction of a feature vector 
		var Xtest = [x];
	}
	else if ( tx == "matrix" || tx == "vector" && tX == "vector") {
		// Multiple predictions for a test set
		var Xtest = x;
	}	
	else
		return "undefined";

	var labels = new Array(Xtest.length);
	var i;
	var j;

	var distance = zeros(N);
	
	for ( i=0; i < Xtest.length; i++) {
		var xi = Xtest[i];
	   	//var D = sub(outerprod(ones(N), xi ) , this.X );
	   	//var distance = sum( entrywisemul(D,D), 2); // distance[i] = ||x - Xi||^2	
		for ( j=0; j < N; j++) {
			var diff = subVectors(this.X[j], xi);
			distance[j] = dot(diff,diff);
		}
		var votes = zeros(this.labels.length); 
		var exaequodistances = zeros(this.labels.length); 
	   
		var idx;
		var lbl;
		var k;
		for(k = 0;k < K; k++) {
		
			idx = findmin( distance );
			lbl = this.labels.indexOf(this.y[idx]);
			votes[ lbl ] += 1; 
			exaequodistances[ lbl ] += distance[idx]; 
		
			distance[idx] = +Infinity;					
		}

		var label = 0;
		var labelcandidates = new Array();
		labelcandidates.push(0);
		var j;
		for(j = 1; j < votes.length; j++) {
			if(votes[j] > votes[label]) {	// find maximum votes
				label = j;
				while (labelcandidates.length > 0 ) 
					labelcandidates.pop();
			}
			else if ( votes[j] == votes[label] ) {	// while dealing with exaequo
				labelcandidates.push(j);
			}
		}
		// for ex-aequo: take the min sum of distances	
		if ( labelcandidates.length > 1 ) 
			label = findmin( exaequodistances );
			
		labels[i] = this.labels[label]; 
	}
	
	if ( labels.length > 1 ) 
		return labels;
	else
		return labels[0];
}

/**
 * @param {Float64Array}
 * @param {Matrix}
 * @param {Array}
 * @return {{indexes: Array, distances: Array}} 
 */
function nnsearch(x, X, featuresOrder) {

	var neighbor = 0
	var distance = Infinity; 
	const dim = X.n;
	const N = X.m
	
	var i;
	var k;
	var Xi; 
	Xi = 0;
	for ( i=0; i < N;  i++) {
	   	var dist = 0; 
	   	k=0;
	   	while ( k < dim && dist < distance ) {
			var diff = X.val[Xi + k ] - x[ featuresOrder[k] ];
			dist += diff*diff;
			k++;
		}
	
		if ( dist < distance ) {
			distance = dist; 
			neighbor = i;
		}
				
	   	Xi += dim;
	}
	
	return {indexes: [neighbor], distances: [distance]}; 
}

function knnsearch(K, x, X, featuresOrder) {
	if (type(X) == "vector")
		return knnsearch1D(K, x, X);
	else
		return knnsearchND(K, x, X, featuresOrder);
}

/**
 * @param {number}
 * @param {Float64Array}
 * @param {Matrix}
 * @param {Array}
 * @return {{indexes: Array, distances: Array}} 
 */
function knnsearchND(K, x, X, featuresOrder) {

	if (type(X) == "vector")
		return knnsearch1D(K, x, X);
		
	var neighbors = new Array(K);		
	var distances = zeros(K);
	var maxmindist = Infinity; 
	const dim = X.n;
	const N = X.m
	if ( typeof (featuresOrder)  == "undefined" )
		var featuresOrder = range(dim);
	
	var i;

	var Xi = 0;
	for ( i=0; i < N;  i++) {
	   	var dist = 0; 

	   	var k=0;

	   	while ( k < dim && dist < maxmindist ) {
			var diff = X.val[Xi + k ] - x[ featuresOrder[k] ];
			dist += diff*diff;
			k++;
		}
	
		if ( dist < maxmindist ) {
			// Find position of Xi in the list of neighbors
			var l = 0;
			/*if ( i < K ) {
				while ( l < i && dist > distances[l] ) 
					l++;					
			}
			else {
				while ( l < K && l < i && dist > distances[l] ) 
					l++;							
			}		*/	
			while ( l < K && l < i && dist > distances[l] ) 
				l++;	
			// insert Xi as the kth neighbor
			//if (l < K ) {				
				neighbors.splice(l,0, i);
				neighbors.pop();
				for (var j=K-1; j > l ; j--) {
					distances[j] = distances[j-1];
				}
				distances[l] = dist;

				if ( l == K-1 )
					maxmindist = dist;					
			//}			
		}	
		
	   	Xi += dim;
	}

	return {indexes: neighbors, distances: distances}; 
}
/**
 * @param {number}
 * @param {number}
 * @param {Float64Array}
 * @return {{indexes: Array, distances: Array}} 
 */
function knnsearch1D(K, x, X) {

	var neighbors = new Array(K);		
	var distances = zeros(K);
	var maxmindist = Infinity; 

	const N = X.length
		
	var i;

	for ( i=0; i < N;  i++) {
	   	var dist = X[i] - x; 
		dist *= dist;			   	
	
		if ( dist < maxmindist ) {
			// Find position of Xi in the list of neighbors
			var l = 0;
			
			while ( l < K && l < i && dist > distances[l] ) 
				l++;	
			// insert Xi as the kth neighbor
			//if (l < K ) {				
				neighbors.splice(l,0, i);
				neighbors.pop();
				for (var j=K-1; j > l ; j--) {
					distances[j] = distances[j-1];
				}
				distances[l] = dist;

				if ( l == K-1 )
					maxmindist = dist;					
			//}			
		}	
	}

	return {indexes: neighbors, distances: distances}; 
}


KNN.prototype.predict = function ( x ) {

   	const N = this.X.length; 
   	if (this.K >  N) {
   		this.K = N;   		
   	}
   	const K = this.K;
   	
   	if ( K == 0 ) {
   		return undefined;
   	}

	const tx = type ( x );
	const tX = type(this.X);
	if ( (tx == "vector" && tX == "matrix") || (tx == "number" && tX == "vector" ) ) {
		// Single prediction of a feature vector 
		var Xtest = new Matrix(1, x.length, x);
	}
	else if ( tx == "matrix" || tx == "vector" && tX == "vector") {
		// Multiple predictions for a test set
		var Xtest = x;
	}	
	else
		return "undefined";

	var labels = new Array(Xtest.length);
	var i;
	var k;
	var idx;
	var lbl;
	var label ;
	var nn ;
	var votes;
	var sumdistances ;
	for ( i=0; i < Xtest.length; i++) {
		// KNN search
		if ( K > 1 ) 
			nn = knnsearch(K, Xtest.row(i), this.X,  this.featuresOrder );
		else
			nn = nnsearch( Xtest.row(i), this.X,  this.featuresOrder);
		
		// Compute votes
		votes = zeros(this.labels.length); 
		sumdistances = zeros(this.labels.length); 	   
		
		for(k = 0;k < K; k++) {		
			idx = nn.indexes[k];
			lbl = this.labels.indexOf(this.y[idx]);
			votes[ lbl ] += 1; 
			sumdistances[ lbl ] += nn.distances[k]; 
		}
		
		// Compute label
		label = 0;

		for(k = 1; k < votes.length; k++) {
			if( (votes[k] > votes[label])  || ( votes[k] == votes[label] && sumdistances[ k ] < sumdistances[label] ) ) {
				label = k;				
			}
		}
		
		labels[i] = this.labels[label]; 
	}

	if ( labels.length > 1 ) 
		return labels;
	else
		return labels[0];
}
KNN.prototype.tune = function ( X, labels, Xv, labelsv ) {
	// Main function for tuning an algorithm on a given training set (X,labels) by cross-validation
	//	or by error minimization on the validation set (Xv, labelsv);
	
	if ( arguments.length == 4 ) {
		// validation set (Xv, labelsv)
		
		// Predict with maximum K while computign labels for all K < maxK
	   	var K = this.parameterGrid.K[this.parameterGrid.K.length - 1];
	   	var N = this.X.length; 
	   	if (K >  N) {
	   		K = N;   		
	   	}
	   	if ( K == 0 ) {
	   		return undefined;
	   	}

		var tx = type ( Xv );
		var tX = type(this.X);
		if ( (tx == "vector" && tX == "matrix") || (tx == "number" && tX == "vector" ) ) {
			// Single prediction of a feature vector 
			var Xtest = [Xv];
		}
		else if ( tx == "matrix" || tx == "vector" && tX == "vector") {
			// Multiple predictions for a test set
			var Xtest = Xv;
		}	
		else
			return "undefined";

		var validationErrors = zeros(K); 
		var i;
		var j;
		var k;
		var idx;
		var lbl;
		var label ;
		var nn ;
		var votes;
		var sumdistances ;
		for ( i=0; i < Xtest.length; i++) {
			// KNN search
			if ( K > 1 ) 
				nn = knnsearch(K, Xtest.val.subarray(i*Xtest.n, (i+1)*Xtest.n), this.X,  this.featuresOrder );
			else
				nn = nnsearch( Xtest.val.subarray(i*Xtest.n, (i+1)*Xtest.n), this.X,  this.featuresOrder );
		
			// Compute votes
			votes = zeros(this.labels.length); 
			sumdistances = zeros(this.labels.length); 	   
	
			for(k = 0;k < K; k++) {		
				idx = nn.indexes[k];
				lbl = this.labels.indexOf(this.y[idx]);
				votes[ lbl ] += 1; 
				sumdistances[ lbl ] += nn.distances[k]; 
			
				
				// Compute label with K=k+1 neighbors
				label = 0;

				for(j = 1; j < votes.length; j++) {
					if( (votes[j] > votes[label])  || ( votes[j] == votes[label] && sumdistances[ j ] < sumdistances[label] ) ) {
						label = j;				
					}
				}
				
				// Compute validation error for all K <= maxK
				if ( labelsv[i] != this.labels[label] )
					validationErrors[k] ++;  
			}
	
		}
		var bestK=0;
		var minValidError = Infinity;
		for ( k=0; k < K; k++) {
			validationErrors[k] /= Xtest.length; 
			if ( validationErrors[k] < minValidError ) {
				bestK = k+1;
				minValidError = validationErrors[k]; 
			}
		}
		
		// set best K in the classifier
		this.K = bestK;
		
		// and return stats
		return {K: bestK, error: minValidError, validationErrors: validationErrors, Kvalues: range(1,K) };

	}
	else {
		//fast LOO
		/*
			Xi is always the nearest neighbor when testing on Xi
		=>  LOO = single test on (Xtrain,Ytrain) with K+1 neighbors without taking the first one into account
		*/ 
		
		// Store whole training set:
		var y = this.checkLabels( labels ) ; 
		var v = variance(X,1); 
		this.featuresOrder = sort(v.val,true, true);

		var relevantFeatures = find ( isNotEqual(v.val, 0) ) ;
		this.X = getCols(X, getSubVector(this.featuresOrder, relevantFeatures) );
		this.y = matrixCopy(labels); // KNN works directly on true labels

		var N = this.X.length; 
		var K = this.parameterGrid.K[this.parameterGrid.K.length - 1]; // max K 
		if ( K > N-1) 
			K = N-1; // N-1 because LOO... 
		K += 1 ; // max K + 1
	   	
		var Xtest = X;

		var LOO = zeros(K-1);
		var i;
		var k;
		var idx;
		var lbl;
		var label ;
		var nn ;
		var votes;
		var sumdistances ;
		for ( i=0; i < Xtest.length; i++) {
			// KNN search
			nn = knnsearch(K, Xtest.val.subarray(i*Xtest.n, (i+1)*Xtest.n), this.X,  this.featuresOrder );
		
			// Compute votes
			votes = zeros(this.labels.length); 
			sumdistances = zeros(this.labels.length); 	   
		
			// Compute label with k neighbors with indexes from 1 to k < K, 
			// not taking the first one (of index 0) into account
			for(k = 1 ; k < K; k++) {		
				idx = nn.indexes[k];
				lbl = this.labels.indexOf(this.y[idx]);
				votes[ lbl ] += 1; 
				sumdistances[ lbl ] += nn.distances[k]; 

				
				label = 0;
				for(j = 1; j < votes.length; j++) {
					if( (votes[j] > votes[label])  || ( votes[j] == votes[label] && sumdistances[ j ] < sumdistances[label] ) ) {
						label = j;				
					}
				}
				
				// Compute LOO error
				if ( labels[i] != this.labels[label] )
					LOO[k-1]++; 
			}	
			
			// Progress
			if ( (i / Math.floor(0.1*Xtest.length)) - Math.floor(i / Math.floor(0.1*Xtest.length)) == 0 )
				notifyProgress( i / Xtest.length );			
		}
		
		var bestK=0;
		var minLOOerror = Infinity;
		for ( k=0; k < K-1; k++) {
			LOO[k] /= N; 
			if ( LOO[k] < minLOOerror ) {
				bestK = k + 1;
				minLOOerror = LOO[k]; 
			}
		}
		
		// set best K in the classifier
		this.K = bestK;
		
		notifyProgress( 1 );	
		return {K: bestK, error: minLOOerror, LOOerrors: LOO, Kvalues: range(1,K) };

	}

	
}



//////////////////////////////////////////////////
/////		Naive Bayes classifier
///////////////////////////////////////////////////

function NaiveBayes ( params ) {
	var that = new Classifier ( NaiveBayes, params);	
	return that;
}
NaiveBayes.prototype.construct = function (params) {
	
	// Default parameters:
	this.distribution = "Gaussian";
	this.epsilon = 1;
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}		

	// Make sure distribution is a string (easier to save/load from files)
	if ( typeof(this.distribution) == "function" ) 
		this.distribution = this.distribution.name;

	// Parameter grid for automatic tuning:
	this.parameterGrid = {   };
}

NaiveBayes.prototype.tune = function ( X, labels ) {
	var recRate = this.cv(X, labels);
	return {error: (1-recRate), validationErrors: [(1-recRate)]};
}
NaiveBayes.prototype.train = function ( X, labels ) {
	// Training function

	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels , true) ; // use 0,1 instead of -1, 1 for binary case
	
	const dim = X.n; 
	this.dim_input = dim;
	this.N = X.m;
	
	var k;
	this.priors = zeros(this.labels.length);
	this.pX = new Array(this.labels.length);
	for ( k=0; k < this.labels.length; k++ ) {
		var idx = find ( isEqual ( y, this.numericlabels[k] ) );

		this.priors[k] = idx.length / y.length;
				
		this.pX[k] = new Distribution ( this.distribution );		
		this.pX[k].estimate( get(X, idx, []) );

		if ( this.distribution == "Bernoulli" ) {
			// Add smoothing to avoid issues with words never (or always) occuring in training set
			this.pX[k].mean = entrywisediv(add(this.epsilon, mul(idx.length , this.pX[k].mean)), idx.length + 2*this.epsilon);
			this.pX[k].variance = entrywisemul(this.pX[k].mean, sub(1, this.pX[k].mean)) ;
			this.pX[k].std = sqrt(this.pX[k].variance);
		}
	}
	
	return this;
}
NaiveBayes.prototype.update = function ( X, labels ) {
	// Online training function
	if ( (typeof(this.distribution) == "string" && this.distribution != "Bernoulli") || (typeof(this.distribution) == "function" && this.distribution.name != "Bernoulli") ) {
		error("Online update of NaiveBayes classifier is only implemented for Bernoulli distribution yet");
		return undefined;
	}
	if ( typeof(this.priors) == "undefined" )
		return this.train(X, labels);
	
	const dim = this.dim_input; 	
	var tx;
	
	var oneupdate = function ( x, y, that ) {
		for ( var k=0; k < that.labels.length ; k++) {
			if ( k == y ) {
				var Nk = that.N * that.priors[k]; 
				
				that.priors[y] = (Nk + 1) / ( that.N + 1 );
				
				if ( tx == "vector")  {
					for ( var j=0;j<dim; j++)
						that.pX[k].mean[j] = (that.pX[k].mean[j] * (Nk + 2*that.epsilon) + x[j] ) / (Nk + 1 + 2*that.epsilon);
				}
				else if ( tx == "spvector" ) {
					var jj = 0;
					for ( var j=0;j < x.ind[jj]; j++)
						that.pX[k].mean[j] = (that.pX[k].mean[j] * (Nk + 2*that.epsilon) ) / (Nk + 1 + 2*that.epsilon);
					that.pX[k].mean[x.ind[jj]] = (that.pX[k].mean[x.ind[jj]] * (Nk + 2*that.epsilon) + x.val[jj] ) / (Nk + 1 + 2*that.epsilon);
					jj++;
					while ( jj < x.val.length ) {
						for ( var j=x.ind[jj-1];j<x.ind[jj]; j++)
							that.pX[k].mean[j] = (that.pX[k].mean[j] * (Nk + 2*that.epsilon) ) / (Nk + 1 + 2*that.epsilon);
						that.pX[k].mean[x.ind[jj]] = (that.pX[k].mean[x.ind[jj]] * (Nk + 2*that.epsilon) + x.val[jj] ) / (Nk + 1 + 2*that.epsilon);

						jj++;
					}
				}
			}
			else {
				that.priors[k] = (that.priors[k] * that.N ) / ( that.N + 1 ); 
			}
		}
		that.N++;	
	};



	if ( this.single_x(X) ) {
		tx = type(X);
		oneupdate( X, this.labels.indexOf( labels ), this );		
	}
	else {
		var Y = this.checkLabels( labels , true) ;
		tx = type(X.row(0));
		for ( var i=0; i < Y.length; i++)
			oneupdate(X.row(i), Y[i], this);
			
	}
	return this;
}

NaiveBayes.prototype.predict = function ( x ) {

	var scores = this.predictscore( x );

	if (typeof(scores) != "undefined") {
		
		if ( this.single_x( x ) ) {
			// single prediction
			return this.recoverLabels( argmax( scores ) );
		}
		else {		
			// multiple predictions for multiple test data
			var i;
			var y = zeros(x.length );
			for ( i = 0; i < x.length; i++)  {
				y[i] = findmax ( scores.row(i) ) ;
			}

			return this.recoverLabels( y );
		}		
	}
	else
		return undefined;	
}

NaiveBayes.prototype.predictscore = function( x ) {

	const tx = type(x);
	
	if ( this.single_x(x) ) {
		var z = log(this.priors);
		for ( var k=0; k < this.labels.length; k++ ) {
			z[k] += this.pX[k].logpdf( x ) ;
		}
	
		return z;
	}
	else {
		var z = new Array(this.labels.length);
		for ( var k=0; k < this.labels.length; k++ ) {
			z[k] = addScalarVector(Math.log(this.priors[k]), this.pX[k].logpdf( x ));
		}
		
		return mat(z);
	}
	
	return z;	
}


//////////////////////////////////////////////////
/////		Decision Trees
///////////////////////////////////////////////////

function DecisionTree ( params ) {
	var that = new Classifier ( DecisionTree, params);	
	return that;
}
DecisionTree.prototype.construct = function (params) {
	
	// Default parameters:
	this.tol = 3;
	this.criterion = "error";
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}		

	// Parameter grid for automatic tuning:
	this.parameterGrid = {   };
}

DecisionTree.prototype.tune = function ( X, labels ) {
	var recRate = this.cv(X, labels);
	return {error: (1-recRate), validationErrors: [(1-recRate)]};
}
DecisionTree.prototype.train = function ( X, labels ) {
	// Training function

	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels , false) ; // use 0,1 instead of -1, 1 for binary case
	
	const dim = X.n; 
	this.dim_input = dim;
	
	var createNode = function ( indexes, Q, tol ) {
		var node = {};

		// Compute the node label
		var NinClasses = zeros(Q);
		for ( var i=0; i < indexes.length; i++) {
			NinClasses[ y[indexes[i]] ] ++;
		}

		node.label = findmax(NinClasses);
		var Nerrors = indexes.length - NinClasses[node.label];
		
		if ( Nerrors > tol ) {
			var best_I = Infinity;
			var best_j = 0;
			var best_s = 0;
			for(var j=0; j < dim; j++) {
				for(var i=0;i<indexes.length;i++) {
					var s = X.get(indexes[i],j);
					
					var idx12 = splitidx( indexes, j, s );
					if ( idx12[0].length > 0 && idx12[1].length > 0 ) {
						var I1 = impurity(idx12[0], Q)	;
						var I2 = impurity(idx12[1], Q)	;
						var splitcost = I1 * idx12[0].length + I2 * idx12[1].length;
						if ( splitcost < best_I ) {
							best_j = j;
							best_s = s;
							best_I = splitcost;
						}
					}
				}
			}
			// Create the node with its children	
			node.j = best_j;
			node.s = best_s;
			var idx12 = splitidx( indexes, best_j, best_s );
			node.child1 = createNode( idx12[0], Q, tol );
			node.child2 = createNode( idx12[1], Q, tol );
			node.isLeaf = false;			
		}
		else 
			node.isLeaf = true;
		
		return node;
	}
	var NfromClass = function (indexes, k ) {
		var n = 0;
		for ( var i=0; i < indexes.length; i++) {
			if ( y[indexes[i]] == k )
				n++;
		}
		return n;
	}
	
	var splitidx = function  ( indexes, j ,s) {
		var idx1 = new Array();
		var idx2 = new Array();
		for ( var i=0; i < indexes.length; i++) {
			if ( X.get(indexes[i],j) <= s ) 
				idx1.push(indexes[i]);
			else
				idx2.push(indexes[i]);
		}
		return [idx1,idx2];
	}
	
	// create impurity function depending on chosen criterion 
	var impurity; 
	switch(this.criterion) {
		case "error":
			impurity = function  ( indexes, Q ) {
				if ( indexes.length == 0 )
					return 0;
			
				var NinClasses = zeros(Q);
				for ( var i=0; i < indexes.length; i++) {
					NinClasses[ y[indexes[i]] ] ++;
				}
				// misclassification rate:
				return (indexes.length - maxVector(NinClasses) ) / indexes.length;
			};
			break;
			/*
		case "gini":
			impurity = function  ( indexes, Q ) {
				console.log("Not yet implemented.");
				return undefined;
			}
			break;
		case "gini":
			impurity = function  ( indexes, Q ) {
				console.log("Not yet implemented.");
				return undefined;
			}
			break;*/
		default:
			return "Unknown criterion: criterion must be \"error\", \"gini\" or \"crossentropy\" (only error is implemented).\n";		
	}
	
	this.tree = createNode( range(y.length), this.labels.length, this.tol );
	
}

DecisionTree.prototype.predict = function ( x ) {
	
	var pred = function (node, x) {
		// works only with a vector x
		if ( node.isLeaf ) {
			return node.label;
		}
		else {
			if ( x[node.j] <= node.s)
				return pred(node.child1, x);
			else
				return pred(node.child2, x);
		}
	}
	
	var tx = type(x) ;
	if ( tx == "matrix" || tx == "vector" && this.dim_input == 1) {
		var lbls = new Array(x.length);
		if( tx == "vector") {
			for ( var i=0; i < x.length; i++) 
				lbls[i] = this.labels[pred(this.tree, x[i] )];
		}
		else {
			for ( var i=0; i < x.length; i++) 
				lbls[i] = this.labels[pred(this.tree, x.row(i) )];		
		}
		return lbls;
	}
	else
		return this.labels[pred(this.tree, x)];
}


//////////////////////////////////////////////////
/////		Logistic Regression (LogReg)
/////
/////  by Pedro Ernesto Garcia Rodriguez, 2014-2015
///////////////////////////////////////////////////

function LogReg ( params ) {
	var that = new Classifier ( LogReg, params);	
	return that;
}

LogReg.prototype.construct = function (params) {
	
	// Default parameters:
	
	// Set parameters:
	var i;
	if ( params) {
		for (i in params)
			this[i] = params[i]; 
	}		

	// Parameter grid for automatic tuning:
	this.parameterGrid = {   };
}

LogReg.prototype.tune = function ( X, labels ) {
	var recRate = this.cv(X, labels);
	return {error: (1-recRate), validationErrors: [(1-recRate)]};
}

LogReg.prototype.train = function ( X, labels ) {
	// Training function

	// should start by checking labels (and converting them to suitable numerical values): 
	var y = this.checkLabels( labels ) ;
	
	// Call training function depending on binary/multi-class case
	if ( this.labels.length > 2 ) {		
		this.trainMulticlass(X, y);		
	}
	else {
		var trainedparams = this.trainBinary(X, y);
		this.w = trainedparams.w; 
		this.b = trainedparams.b;
		this.dim_input = size(X,2);
	}
	/* and return training error rate:
	return (1 - this.test(X, labels));	*/
	return this.info();
}

LogReg.prototype.predict = function ( x ) {
	if ( this.labels.length > 2)
		return this.predictMulticlass( x ) ;
	else 
		return this.predictBinary( x );		
}
LogReg.prototype.predictBinary = function ( x ) {
	
	var scores = this.predictscoreBinary( x , this.w, this.b);
	if (typeof(scores) != "undefined")
		return this.recoverLabels( sign( scores ) );
	else
		return "undefined";	
}

LogReg.prototype.predictMulticlass = function ( x ) {
	
	var scores = this.predictscore( x );
	if (typeof(scores) != "undefined") {
		
		if ( type ( x ) == "matrix" ) {
			// multiple predictions for multiple test data
			var i;
			var y = new Array(x.length );
			for ( i = 0; i < x.length; i++)  {
				var si = scores.row(i);
				y[i] = findmax ( si );
				if (si[y[i]] < 0) 
					y[i] = this.labels.length-1;	// The test case belongs to the reference class Q, 
													// i.e., the last one
			}
			return this.recoverLabels( y );
		}
		else {
			// single prediction
			y = findmax ( scores );
			if (scores[y] < 0) 
				y = this.labels.length-1;  // The test case belongs to the reference class Q, 
										  // i.e., the last one
			return this.recoverLabels( y );
		}
		
	}
	else
		return "undefined";	
}

LogReg.prototype.predictscore = function( x ) {
	var output;
	if ( this.labels.length > 2) {
		
		// single prediction
		if ( this.single_x( x ) ) {
			output = add(mul(this.w,x), this.b);
			return output;
		}
		else {
		// multiple prediction for multiple test data
			output = add(mul(x, transposeMatrix(this.w)), mul(ones(x.m), transposeVector(this.b)));
			return output;
		}	
	}
	else 
		return this.predictscoreBinary( x, this.w, this.b );
}

LogReg.prototype.predictscoreBinary = function( x , w, b ) {
	var output;
	if ( this.single_x( x ) ) 
		output = b + mul(x, w);
	else 
		output = add( mul(x, w) , b);
	return output;
}

/// LogReg training for Binary Classification ////
LogReg.prototype.trainBinary = function ( x, y ) {
 
	//Adding a column of 'ones' to 'X' to accommodate the y-intercept 'b' 
	var X = mat([x, ones(x.m)]);
	
	var MaxError = 1e-3;
	// var LearningRate = 1e-6, MaxError = 1e-3, params = LogRegBinaryDetGradAscent(x, y, LearningRate, MaxError); 
	// var LearningRate = 1e-6, MaxError = 1e-3, params = LogRegBinaryStochGradAscent(x, y, LearningRate, MaxError);
	
	const AlgorithmKind = "Newton-Raphson";
	
		
	///	--- utility functions ---
	function LogRegBinaryStochGradient(j,beta) {
	
		// Computing the jth-term of the Gradient of the Cost Function
	 	// p = X.n-1 is the feature-space dimensionality.
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b

		var C = (y[j]==1 ? 1 : 0) - 1/(1 + Math.exp(-dot(beta,X.row(j)))); // Note that X.row(j) outputs a column instead a row vector. 
				                                                           // Function dot() requires two column vectors

		return mulScalarVector(C, X.row(j));
	}

	function LogRegBinaryDetGradient(beta) {

		// Computing the Deterministic Gradient of the Cost Function
	 	// p = X.n-1 is the feature-space dimensionality.
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b
	
		var beta_grad = zeros(X.n) ; 
	
		for ( var i = 0; i < X.m; i++) {
			var C = (y[i]==1 ? 1 : 0) - 1/(1 + Math.exp(-dot(beta,X.row(i))));  // Function dot() requires two column vectors. Note that 
				                                                                // X.row(i) outputs a column instead a row vector.
			beta_grad = addVectors(beta_grad, mulScalarVector(C, X.row(i)));
		}
	 
		return beta_grad;
	}

	function LogRegBinaryHessian(beta) {
	
		// Computing the Hessian matrix of the Cost Function
		// p = X.n-1 is the feature-space dimensionality.
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b
	
		var v_diag = zeros(X.m);
		for ( var i = 0; i < X.m; i++) {
			var p = 1/(1 + Math.exp(-dot(beta,X.row(i))));
			v_diag[i] = p*(p-1);
		}	
		var W = diag(v_diag);

		var Hessian = mulMatrixMatrix(transposeMatrix(X), mulMatrixMatrix(W,X));
	
		return Hessian;
	}

	function LogRegBinaryCostFunction(beta) {
	
		// p = X.n-1 is the feature-space dimensionality.
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b
	
		var L = 0;			
		for ( var i = 0; i < X.m; i++ ) {
			var betaXi = dot(beta,X.row(i));
			var K = 1 + Math.exp(betaXi);
			L -= Math.log(K);
			if ( y[i] == 1 )
				L += betaXi ;
		}
	
		return L;
	}


	function LogRegBinaryLearningRate(beta_old, GradientOld_v, Lambda, LambdaMin, LambdaMax, MaxErrorL, MaxError) {

		// Computing the first point 'beta_new' = (w_new, b_new)
		beta = addVectors(beta_old, mulScalarVector(Lambda, GradientOld_v));
		
		do {
			var Lambda_old = Lambda;
		
			// Computing the first derivative of the Cost Function respect to the learning rate "Lambda"
			var GradientNew_v = LogRegBinaryDetGradient(beta);
			var FirstDerivLambda = dot(GradientNew_v,GradientOld_v);
		
			// Computing the second derivative of the Cost Function respect to the learning rate "Lambda"
			var HessianNew = LogRegBinaryHessian(beta);
			var SecondDerivLambda = dot(GradientOld_v, mulMatrixVector(transposeMatrix(HessianNew), GradientOld_v));

			if (!isZero(SecondDerivLambda)) { 
				Lambda -= FirstDerivLambda/SecondDerivLambda;
				// console.log("FirstDer:", FirstDerivLambda, "SecondDer:", SecondDerivLambda, -FirstDerivLambda/SecondDerivLambda, Lambda);
				// console.log("Lambda:", Lambda, "Lambda_old:", Lambda_old, Lambda - Lambda_old);
			}
		
			if (Lambda > LambdaMax || Lambda < LambdaMin)
				Lambda = LambdaMin;
		
			// Updating the values of the parameters 'beta'
			beta = addVectors(beta_old, mulScalarVector(Lambda, GradientOld_v));
			
		} while( Math.abs(Lambda-Lambda_old) > MaxErrorL && Math.abs(FirstDerivLambda) > MaxError );
		// if (Lambda > LambdaMax || Lambda < LambdaMin) Lambda = Lambda_old;
	
		return Lambda;
	}
	// --- end of utility functions ---

	if (AlgorithmKind == "Stochastic Gradient Ascent" ) {
	
	 	// Stochastic Gradient Optimization Algorithm
	 	// to compute the Y-intercept 'b' and a p-dimensional vector 'w',
		// where p = X.n is the feature-space dimensionality
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b
	
		// Initial guess for values of parameters beta = w,b
		var beta = divScalarVector(1,transpose(norm(X,1))); 
	
		var i = 0;
		var CostFunctionOld;
		var CostFunctionNew = LogRegBinaryCostFunction(beta);
		do {
			var beta_old = matrixCopy(beta);
	
			// LearningRate optimization
			if (LearningRate > 1e-8) 
				LearningRate *= 0.9999;
		
			// -- Updating the values of the parameters 'beta' --
			// Doing a complete random sweep across the whole training set
			// before verifying convergence 
			var seq = randperm(X.m);
			for ( var k = 0; k < X.m; k++) {
			
				index = seq[k];
				var GradientOld_v = LogRegBinaryStochGradient(index, beta_old);
		
				// Updating the values of the parameters 'beta' with just 
				// the index-th component of the Gradient
				var Delta_params = mulScalarVector(LearningRate, GradientOld_v);
				beta = addVectors(beta_old, Delta_params);
			
			}
		
			//  Checking convergence
			if ( i%1000==0) {
				CostFunctionOld = CostFunctionNew;
				CostFunctionNew = LogRegBinaryCostFunction(beta);
				console.log(AlgorithmKind, i, CostFunctionNew , CostFunctionOld, CostFunctionNew - CostFunctionOld);
			}     
			var GradientNorm = norm(GradientOld_v);
			
			//var PrmtRelativeChange = norm(subVectors(beta,beta_old))/norm(beta_old); 
			// if (i%100 == 0) 
				//console.log( AlgorithmKind, i, Math.sqrt(GradientNormSq), PrmtRelativeChange );

			i++;

		}  while (GradientNorm > MaxError ); // &&  PrmtRelativeChange > 1e-2);
	
		var w = getSubVector(beta, range(0, beta.length-1));
		var b = get(beta, beta.length - 1);
	}
	else if ( AlgorithmKind == "Deterministic Gradient Ascent" ) {

	 	// Deterministic Gradient Optimization Algorithm
	 	// to compute the Y-intercept 'b' and a p-dimensional vector 'w',
		// where p = X.n is the feature-space dimensionality
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b
	
		// Initial guess for values of parameters beta = w,b
		var beta = divScalarVector(1,transpose(norm(X,1))); 
	

		// For LearningRate optimization via a Newton-Raphson algorithm
		var LambdaMin = 1e-12, LambdaMax = 1, MaxErrorL = 1e-9;
	
		var i = 0;
		var CostFunctionOld;
		var CostFunctionNew = LogRegBinaryCostFunction(beta);
		do {
			var beta_old = matrixCopy(beta);
		
			var GradientOld_v = LogRegBinaryDetGradient(beta_old);
		
			//  LearningRate optimization
			// if (LearningRate > 1e-12) LearningRate *= 0.9999;

			// Newton-Raphson algorithm for LearningRate
			LearningRate = LogRegBinaryLearningRate(beta_old, GradientOld_v, LearningRate, LambdaMin, LambdaMax, MaxErrorL, MaxError);
			
			// Updating the values of the parameters 'beta'
			var Delta_params = mulScalarVector(LearningRate, GradientOld_v);
			beta = addVectors(beta_old, Delta_params);
		
			// Checking convergence		 
			if ( i%100==0) {
				CostFunctionOld = CostFunctionNew;
				CostFunctionNew = LogRegBinaryCostFunction(beta);
				console.log(AlgorithmKind, i, CostFunctionNew , CostFunctionOld, CostFunctionNew - CostFunctionOld);
			} 
			  
			var GradientNorm = norm(GradientOld_v);
			//var PrmtRelativeChange = norm(subVectors(beta,beta_old))/norm(beta_old); 
			// if (i%100 == 0) 
			//	console.log( AlgorithmKind, i, Math.sqrt(GradientNormSq), PrmtRelativeChange );

			i++;
	
		} while ( GradientNorm > MaxError ); // &&  PrmtRelativeChange > 1e-2);
	
		var w = getSubVector(beta, range(0, beta.length-1));
		var b = get(beta, beta.length - 1);
	
	}
	else {
	
		// Newton-Raphson Optimization Algorithm
	 	// to compute the y-intercept 'b' and a p-dimensional vector 'w',
		// where p = X.n-1 is the feature-space dimensionality.
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b
	
	  	// Initial guess for values of parameters beta = w,b
		var beta = divScalarVector(1,transpose(norm(X,1))); 
	
		var i = 0;
		var CostFunctionOld;
		var CostFunctionNew = LogRegBinaryCostFunction(beta);
		do {

			var beta_old = vectorCopy(beta);
		
			var GradientOld_v = LogRegBinaryDetGradient(beta_old);
		
			var HessianOld = LogRegBinaryHessian(beta_old);
				
			//   Updating the values of the parameters 'beta' 
			// var Delta_params = mul(inv(HessianOld),GradientOld_v);
			var Delta_params = solve(HessianOld, GradientOld_v);
			beta = subVectors(beta_old, Delta_params);

			// Checking convergence
			CostFunctionOld = CostFunctionNew;
			CostFunctionNew = LogRegBinaryCostFunction(beta);
			console.log(AlgorithmKind, i, CostFunctionNew , CostFunctionOld, CostFunctionNew - CostFunctionOld);
				                      
			var GradientNorm = norm(GradientOld_v);
			// var PrmtRelativeChange = norm(subVectors(beta,beta_old))/norm(beta_old); 
			//if (i%100 == 0) 
				// console.log( AlgorithmKind + " algorithm:", i, Math.sqrt(GradientNormSq), PrmtRelativeChange );

			i++;
		
		} while ( GradientNorm  > MaxError ); // &&  PrmtRelativeChange > 1e-2);
	
		var w = getSubVector(beta, range(0, beta.length-1));
		var b = get(beta, beta.length - 1);
			
	}

    return {w: w, b : b};
}




// LogReg Multi-class classification ////////////////////
LogReg.prototype.trainMulticlass = function (x, y) {
	
	//Adding a column of 'ones' to 'X' to accommodate the y-intercept 'b' 
	var X = mat([x,ones(x.m)]);
	
	var LearningRate = 1e-6;
	var MaxError = 1e-3;
	const Q = this.labels.length;
	const AlgorithmKind = "NewtonRaphson";
	

	// Building a concatenated block-diagonal input matrix "X_conc",
		// by repeating Q-1 times "X" as blocks in the diagonal.
		// Order of X_conc is X.m*(Q-1) x X.n*(Q-1), where X.m = N (size of training set),
		// X.n - 1 = p (feature-space dimensionality) and Q the quantity of classes.
	
	var X_conc = zeros(X.m*(Q-1),X.n*(Q-1));
	for (var i_class = 0; i_class < Q-1; i_class++)
	    set(X_conc, range(i_class*X.m, (i_class+1)*X.m), range(i_class*X.n, (i_class+1)*X.n), X);
  
  	var X_concT = transposeMatrix(X_conc);
  	
	// Building a concatenated column-vector of length (y.length)*(Q-1)
		// with the Indicator functions for each class-otput

	var Y_conc = zeros(y.length*(Q-1));
	for (var i_class = 0; i_class < Q-1; i_class++)
	   set(Y_conc, range(i_class*y.length, (i_class+1)*y.length), isEqual(y,i_class)); 
	

	///////////// Utility functions ///////////// 

	function LogRegMultiDetGradient(beta) {

		// Computing the Deterministic Gradient of the Cost Function
		// p = X.n-1 is the feature-space dimensionality.
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b
	
		// Computing the conditional probabilities for each Class and input vector
		var pi = LogRegProbabilities(beta);
			
		return mulMatrixVector(X_concT, subVectors(Y_conc,pi));	
	}

	function LogRegProbabilities(beta) {
		// Building a concatenated column-vector of length X.m*(Q-1) with
		// the posterior probabilities for each Class and input vector
		
		var p = zeros((Q-1)*X.m);

		var InvProbClassQ = LogRegInvProbClassQ(beta);
		for ( var i = 0; i < X.m; i++ )
		    for ( i_class = 0; i_class < Q-1; i_class++ ) {
				var betaClass = getSubVector(beta, range(i_class*X.n,(i_class+1)*X.n));
		        p[i_class*X.m + i] = Math.exp(dot(betaClass,X.row(i)))/InvProbClassQ[i];
			}
		return p;		
	}

	function LogRegInvProbClassQ(beta) {
	
		// Computes the inverse of the Posterior Probability that each input vector 
		// is labeled with the reference Class (the last one is chosen here)
	
		var InvProbClass_Q = ones(X.m);	
		for ( var i = 0; i < X.m; i++)
			for ( var i_class = 0; i_class < Q-1; i_class++ ) {	
				var betaClass = getSubVector(beta, range(i_class*X.n,(i_class+1)*X.n));
				InvProbClass_Q[i] += Math.exp(dot(betaClass,X.row(i)));
			}					
		return InvProbClass_Q;
	}

	function LogRegMultiHessian(beta) {
	   	// Computing the Hessian matrix of the Cost Function

		
		// Computing the conditional probabilities for each Class and input vector
		var p = LogRegProbabilities(beta);

		// Building the Hessian matrix: a concatenated block-diagonal input matrix "W_conc"
		// of order X.m*(Q-1) x X.m*(Q-1), whose blocks W_jk are diagonal matrices as well
		var W_conc = zeros(X.m*(Q-1),X.m*(Q-1));
		for ( var j_class = 0; j_class < Q-1; j_class++ )
		    for ( var k_class = 0; k_class < Q-1; k_class++ ) {
		        var v_diag = zeros(X.m);
		        for ( var i = 0; i < X.m; i++ ) 
					v_diag[i] = p[j_class*X.m + i]*( (j_class == k_class? 1 : 0) - p[k_class*X.m + i] );
				
		        var W_jk = diag(v_diag);
		        set(W_conc, range(j_class*X.m, (j_class+1)*X.m), range(k_class*X.m, (k_class+1)*X.m), W_jk);
		    }
		
		var Hessian = mulMatrixMatrix(transposeMatrix(X_conc), mulMatrixMatrix(W_conc,X_conc));
		
		return minusMatrix(Hessian);
	}

	function LogRegMultiCostFunction(beta) {
	
		var InvProbClassQ = LogRegInvProbClassQ(beta);
	
		// Contribution from all the Classes but the reference Class (the last one is chosen here)
		var L = 0;
		for ( var i_class = 0; i_class < Q-1; i_class++ )
			for ( var i = 0; i < X.m; i++ ) {
				var betaClass = getSubVector(beta, range(i_class*X.n,(i_class+1)*X.n));
				L += (y[i]==i_class? 1: 0)*(dot(betaClass,X.row(i)) - Math.log(InvProbClassQ[i]) );
			}
		
		// Contribution from the reference Class (the last one is chosen here)		
		for ( i = 0; i < X.m; i++ )
			if ( y[i]==Q ) 
				L -= Math.log(InvProbClassQ[i]);
		
		return L;
	}

	// --- end of utility functions ---

	
	if ( AlgorithmKind == "DetGradAscent" ) {
	 	// Deterministic Gradient Optimization Algorithm
	 	// to compute the (Q-1)-dimensional vector of y-intercept 'b' and a px(Q-1)-dimensional matrix 'w',
		// where p = X.n-1 is the feature-space dimensionality and Q the quantity of classes.
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b'
	
	  	// Initial guess for values of parameters beta
	  	var beta = zeros((Q-1)*X.n);
	  	for (var i_class = 0; i_class < Q-1; i_class++)
			set(beta, range(i_class*X.n,(i_class+1)*X.n), divScalarVector(1,transpose(norm(X,1))));  
	
		var i = 0;
		var CostFunctionOld;
		var CostFunctionNew = LogRegMultiCostFunction(beta);
		do {
			var beta_old = vectorCopy(beta);
//			var CostFunctionOld = LogRegMultiCostFunction(beta_old);
			
			var GradientOld_v = LogRegMultiDetGradient( beta_old);
		
			//   LearningRate optimization 
			if (LearningRate > 1e-12) LearningRate *= 0.9999;
		
			//  Updating the values of the parameters 'beta'  
		
			var Delta_params = mulScalarVector(LearningRate, GradientOld_v);
			beta = addVectors(beta_old, Delta_params);
	
			//  Checking convergence		
			if ( i%100==0) {
				CostFunctionOld = CostFunctionNew;
				CostFunctionNew = LogRegMultiCostFunction(beta);
				console.log(AlgorithmKind, i, CostFunctionNew , CostFunctionOld, CostFunctionNew - CostFunctionOld);
			}
			// var CostFunctionDiff = LogRegMultiCostFunction(X, y, Q, beta) - LogRegMultiCostFunction(X, y, Q, beta_old);                        
		
			var GradientNormSq = norm(GradientOld_v)*norm(GradientOld_v); 
			//var PrmtRelativeChange = norm(subVectors(beta,beta_old))/norm(beta_old); 
			//if (i%100 == 0) 
			//	console.log( AlgorithmKind + " algorithm:", i, Math.sqrt(GradientNormSq), PrmtRelativeChange );
		
			i++;
		
		} while (Math.abs(CostFunctionNew - CostFunctionOld) > 1e-3); //Math.sqrt(GradientNormSq) > MaxError ); // &&  PrmtRelativeChange > 1e-2);
	
		var betaMatrix = reshape(beta, Q-1, X.n);
		var w_new = get(betaMatrix, range(), range(0, betaMatrix.n-1));
		var b_new = get(betaMatrix, range(), betaMatrix.n-1);
		console.log(betaMatrix,w_new);
	}
	// else if (AlgorithmKind == "Newton-Raphson")
	else {
	     
	 	// Newton-Raphson Optimization Algorithm
	 	// to compute the (Q-1)-dimensional vector of y-intercept 'b' and a px(Q-1)-dimensional matrix 'w',
		// where p = X.n-1 is the feature-space dimensionality and Q the quantity of classes.
		// Note that input matrix 'X' contains a last column of 'ones' to accommodate the y-intercept 'b'
	
	  	// Initial guess for values of parameters beta
	  	var beta = zeros((Q-1)*X.n);
	  	for (var i_class = 0; i_class < Q-1; i_class++)
			set(beta, range(i_class*X.n,(i_class+1)*X.n), divScalarVector(1,transpose(norm(X,1))));  

		var i = 0;
		var CostFunctionOld;
		var CostFunctionNew = LogRegMultiCostFunction(beta);
		do {
		
			var beta_old = vectorCopy(beta);			
			var CostFunctionOld = LogRegMultiCostFunction(beta_old);
			
			var GradientOld_v = LogRegMultiDetGradient(beta_old);
		
			var HessianOld = LogRegMultiHessian(beta_old);
		
			//  Updating the values of the parameters 'beta' 
			// var Delta_params = solveWithQRcolumnpivoting(HessianOld, GradientOld_v);
			var Delta_params = solve(HessianOld, GradientOld_v);
			beta = subVectors(beta_old, Delta_params);

			// Checking convergence
			CostFunctionOld = CostFunctionNew;
			CostFunctionNew = LogRegMultiCostFunction(beta);
			console.log(AlgorithmKind, i, CostFunctionNew , CostFunctionOld, CostFunctionNew - CostFunctionOld);
				                      
		
			var GradientNorm = norm(GradientOld_v);
			//var PrmtRelativeChange = norm(subVectors(beta,beta_old))/norm(beta_old); 
			//if (i%100 == 0) 
			//	console.log( AlgorithmKind + " algorithm:", i, Math.sqrt(GradientNormSq), PrmtRelativeChange );

			i++;
		
		} while ( GradientNorm > MaxError ); // &&  PrmtRelativeChange > 1e-2);
	
		var betaMatrix = reshape(beta, Q-1, X.n);
		var w_new = get(betaMatrix, range(), range(0, betaMatrix.n-1));
		var b_new = get(betaMatrix, range(), betaMatrix.n-1);
	}        
	
			
	this.w = w_new;
	this.b = b_new;
	
}
