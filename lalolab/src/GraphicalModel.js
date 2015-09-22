///////////////////////////////////////////
/// Generic class for Graphical Models
//////////////////////////////////////////
function GraphicalModel (algorithm, params ) {

	this.type = "GraphicalModel:" + algorithm.name;

	this.graphicalModel = algorithm.name;
	this.userParameters = params;

	// Functions that depend on the algorithm:
	this.construct = algorithm.prototype.construct; 
	
	// Training functions
	this.train = algorithm.prototype.train; 
	
	// Prediction functions
	this.forward = algorithm.prototype.forward;
	this.backward = algorithm.prototype.forward;
	this.smoothing = algorithm.prototype.forward;		

	this.annotate = algorithm.prototype.annotate;
	
	if ( algorithm.prototype.sample ) 
		this.sample = algorithm.prototype.sample;
	if ( algorithm.prototype.train ) 
		this.train = algorithm.prototype.train;
	if ( algorithm.prototype.uniform ) 
		this.uniform = algorithm.prototype.uniform;
	if ( algorithm.prototype.random ) 
		this.random = algorithm.prototype.random;
	
	// Initialization depending on algorithm
	this.construct(params);

}

GraphicalModel.prototype.construct = function ( params ) {
	// Read this.params and create the required fields for a specific algorithm
}

GraphicalModel.prototype.tune = function ( X, labels ) {
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

GraphicalModel.prototype.train = function (X, labels) {
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
	return (1 - this.test(X, labels));

}
/**
 * @return {string}
 */
GraphicalModel.prototype.info = function () {
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
				str += i + ": vector of size " + this[i].length + "<br>";
				break;
			case "matrix":
				str += i + ": matrix of size " + this[i].m + "-by-" + this[i].n + "<br>";
				break;
			case "Array":
				str += i + ": array of length " + this[i].length + " : " + this[i].toString() + "<br>";
				break;
			case "function": 
				if ( this[i].name.length == 0 )
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

////////////////////////////////////
///   HMM
////////////////////////////////////
function HMM ( params ) {
	var that = new GraphicalModel(HMM, params);
	return that;
}
HMM.prototype.construct = function ( params ) {
	
	if ( typeof(params.states) == "undefined" ) {
		this.Nstates = params.Nstates; 
		this.states = "ABCDEFGHIJKLMNOPWRSTUVWXYZ".slice(0,params.Nstates);
	}
	else {
		this.Nstates = params.states.length;
		this.states = params.states;	// string or Array of states
	}
	if ( typeof(params.outputs) == "undefined" ) {
		this.Noutputs = params.Noutputs; 
		this.outputs = "ABCDEFGHIJKLMNOPWRSTUVWXYZ".slice(0,params.Noutputs);
	}
	else {
		this.Noutputs = params.outputs.length;
		this.outputs = params.outputs;
	}
	
	this.prior = zeros(this.Nstates);
	this.T = zeros(this.Nstates, this.Nstates);
	this.E = zeros(this.Nstates, this.Noutputs);
	this.logP = zeros(this.Nstates);
	this.logT = zeros(this.Nstates, this.Nstates);
	this.logE = zeros(this.Nstates, this.Noutputs);
	
}

HMM.prototype.uniform = function () {
	// initialize all probabilities uniformly
	const Nstates = this.Nstates;
	const Noutputs = this.Noutputs;
	var i;
	var j;
	
	for(i=0;i<Nstates;i++) {
		this.prior[i] = 1 / Nstates; 
		for(j=0;j< Nstates;j++)
			this.T.val[i * Nstates + j] = 1 / Nstates; 
		for(j=0;j<Noutputs;j++)
			this.E.val[i * Noutputs + j] = 1 / Noutputs;
	}
	// precompute logs
	this.logP = log(this.prior);
	this.logT = log(this.T);
	this.logE = log(this.E);
}



HMM.prototype.random = function () {
/*
	Initialize parameters of HMM randomly
	(uniform distribution)
*/
	const N = this.Nstates;
	const Noutputs = this.Noutputs;
	var i,j;
		
	var normalize =  function(v) {
		// Normalize vector to sum to 1
		var s = sumVector(v);
		for ( var i=0; i < v.length; i++)
			v[i] /= s;
	}
		
	for(i=0;i<N;i++) {
		this.prior[i] = Math.random();
				
		for(j=0;j< N; j++) {
			this.T.val[i * N + j] = Math.random(); 
		}
		normalize( this.T.row(i) );

		for(j=0;j<Noutputs;j++) {
			this.E.val[i * Noutputs + j] = Math.random();
		}
		normalize(this.E.row(i));
	}
	normalize(this.prior);	
	
	// precompute logs
	this.logP = log(this.prior);
	this.logT = log(this.T);
	this.logE = log(this.E);
}


HMM.prototype.normalize = function(v) {
	// Normalize vector to sum to 1
	var s = sumVector(v);
	for ( var i=0; i < v.length; i++)
		v[i] /= s;
}


HMM.prototype.normalizeLogspace = function(v) {
/*	IN PLACE operation!!
	Normalize in log space:
	from a vector v of v_i = log (p_i)
	we want new v_i such that sum_i p_i = 1	
	
*/
	var v_max = maxVector(v);
	// scale = v_max + log( sum_i exp ( v_i - v_max) )	
	var scale = v_max + Math.log( sumVector(exp(subVectorScalar(v, v_max))) );
	
	// v_i <- v_i - scale
	// return subVectorScalar(v, scale); // below is in place operation
	for ( var i = 0; i < v.length; i++)
		v[i] -= scale;
}

HMM.prototype.train = function ( stateSequences, obsSequences, labels ) {

	const N = this.Nstates;
	const Noutputs = this.Noutputs;
	
	var i,j,t,k;
	var x;
	var length;
	
	var countObservation = zeros(N, Noutputs); 
	var countTransition = zeros(N,N);
	var countState = zeros(N);
	var Q = new Array(stateSequences.length);
	
	// Emissions + parse sequences
	for(i=0;i < sequences.length ;i++) {
		
		x = new Array(obsSequences[i].length);
		Q[i] = new Array(stateSequences[i].length);
		for ( t = 0; t < obsSequences[i].length; t++) {
		
			x[t] = this.outputs.indexOf(obsSequences[i][t]);
			Q[i][t] = this.states.indexOf(stateSequences[i][t]);
			if ( x[t] == -1 || Q[i][t] == -1)
				return "Unknown states orobservations in the sequences.";
			
			countObservation.val[ Q[i][t] * Noutputs + x[t] ]++;
			countState[ Q[i][t] ]++;				
		}
	}
	// Ei = countobs_i / countstate_i
	this.E = entrywisediv( countObservation, outerprod(countState, ones(Noutputs)) );
	
	// Transitions
	for(i=0;i<Q.length;i++) {
		for(t=0;t< Q[i].length - 1 ;t++) {
			countTransition[ Q[i][t] * N + q[t+1]  ]++;
		}
		countState[ Q[i][Q[i].length-1] ] --; 
	}
	this.T = entrywisediv( countTransition, outerprod(countState, ones(N)) );	

	// Priors
	countState = zeros(N);
	for(i=0;i<Q.length;i++) {
		countState[ Q[i][0] ]++;
	}
	this.prior = entrywisediv(countState, Q.length ) ;


	// precompute logs
	this.logP = log(this.prior);
	this.logT = log(this.T);
	this.logE = log(this.E);
}

/*
	Sample Nseq sequences of length T with an HMM 	
*/
HMM.prototype.sample = function(Nseq, T) {
	var i,xt,t;
	const N = this.N;
	const Noutputs = this.Noutputs;	
	
	// Output sequences
	var X = new Array(Nseq); // numerical format
	var Xstr = new Array(Nseq); // string (or list of strings) format

	// State sequences
	var Q = new Array(Nseq);
	var Qstr = new Array(Nseq);
	
	for(i=0;i<Nseq;i++) {
		X[i] = new Array(T);
		Q[i] = new Array(T); 
		Xstr[i] = new Array(T);
		Qstr[i] = new Array(T); 
	}
	
	var sampleDiscrete = function ( P ) {
		var v=0;
		var u = Math.random(); // uniform in [0,1]
		var sumP = P[0];	
		while(u > sumP) {
			v++; 
			sumP += P[v];
		}	
		return v;
	}
	
	for(i=0;i<Nseq;i++) {
		// Draw initial states	
		Q[i][0] = sampleDiscrete(this.prior);
		Qstr[i][0] = this.states[Q[i][0]];
	
		// and first observation
		X[i][0] = sampleDiscrete(this.E.row( Q[i][0] ) );
		Xstr[i][0] = this.outputs[X[i][0]]; 
		
		// Draw state and observation sequence
		for(t=1;t<T;t++) {
			// Draw state from transition probabilities
			Q[i][t] = sampleDiscrete( this.T.row( Q[i][t-1] ) );
			Qstr[i][t] = this.states[Q[i][t]];
			
			// Draw observation from emission probabilities
			X[i][t] = sampleDiscrete(this.E.row ( Q[i][t] ) );
			Xstr[i][t] = this.outputs[X[i][t]];
		}	
	}
	
	return {stateSequences: Qstr, numstateSequences: Q, obsSequences: Xstr, numobsSequences: X};
}


HMM.prototype.forward = function (x) {

	const N = this.Nstates;
	const Noutputs = this.Noutputs;	
	
	var i,j,t;
	var sum_i_alpha_t_Tij;
	var c = zeros(x.length);
	var alpha = zeros(x.length * N);
	
	// 1) Initialization 		
	c[0] = 1.0;
	xt = this.outputs.indexOf(x[0]);
	for(i=0;i < N; i++) { 		
		alpha[i] = this.prior[i] * this.E.val[i * Noutputs + xt] ; 
	}
	
	/*
		2) Induction
		 loop for all time steps
	*/	
	for(t=1;t<x.length; t++) {
		
		c[t] = 0.0;
		xt = this.outputs.indexOf(x[t]);
		/*
			Loop over all states
		*/		
		for(j=0; j<N; j++) {

			// Compute sum_i
			sum_i_alpha_t_Tij = 0;
			for(i=0; i<N; i++)
				sum_i_alpha_t_Tij += alpha[(t-1) * N + i] * this.T.val[i*N + j];

			// Compute alpha_t(j)
			alpha[t * N + j] = sum_i_alpha_t_Tij * this.E.val[j * Noutputs + xt];
		
			// Compute c[t] = scaling coefficient			
			c[t] += alpha[t * N + j];
		}
		// Scale alpha
		if ( c[t] > 0) {
			c[t] = 1./c[t];
			for(j=0;j < N; j++) {
				alpha[t * N + j] *= c[t];
			}
		}
		else
			c[t] = 1.0;
	}

	// 3) Termination
	var logPsequence = 0;
	for(t=1;t<x.length;t++) 
		logPsequence -= Math.log(c[t]);	
		
	return logPsequence;
}

/*
	VITERBI for HMM: maximize the log-likelihood wrt the state sequence

	Return max_Q log P(Q, sequence | hmm) 
	
	X: the observation sequence
	Q: a state sequence	
	
	delta_t(i) = max log of probability of state and observation sequences
			that end in state i at time t
	
	delta_t(i) = max_Q log P(q0...qt-1, qt = i, x0,...,xt | hmm)
	
	1)  delta_0(i) = log P(q0=i, x0 | hmm) = log P(q0 = i) + log P(x0 | q0 = i) 

	2)  delta_t(j,1) = max_i [ delta_(t-1)(i) + log P(qt = j | qt_1 = i) ] 
				+ log  P(xt | qt = j)
	    	
	    psi_t(j) = index of the max_i (argmax)  (as a state)
	    
	3)  max_Q log P(Q, X | hmm) = max_i delta_T(i)  
	
	4) Backtracking
*/
HMM.prototype.annotate =
HMM.prototype.viterbi = function (sequence) {
/*
h = new GraphicalModel(HMM, {states:"ABC",outputs: "abc"})
h.prior = [0.3;0.3;0.4]
h.T = [ [ 0.1, 0.8, 0.1]; [0.1, 0.1, 0.8]; [0.3,0.3, 0.4] ]
h.E = [ [ 0.9, 0.05, 0.05]; [0.1, 0.8, 0.1]; [0.15,0.05, 0.8] ]
s = h.sample(2, 100)
p = h.annotate(s.obsSequences[0])
*/

	var Qopt = zeros(sequence.length);
	var Qoptstr = new Array(sequence.length);
	
	const N = this.Nstates;
	const Noutputs = this.Noutputs;	
	var i,j,t;
	var xt;
	
	var delta = zeros(N * sequence.length);
	
	var max_i_delta_t_Tij;
	var psi = zeros(sequence.length * N );
	
	var delta_Tij;

	// 0) precompute logs 
	// (this must be done unless we guarantee that it is whenvever a parameter has changed)
	this.logP = log(this.prior);
	this.logT = log(this.T);
	this.logE = log(this.E);


	// 1) Initialization (for t=0)
	xt = this.outputs.indexOf( sequence[0] );
	for(i=0;i < N; i++) { 	
		delta[i] = this.logP[i] + this.logE.val[i * Noutputs + xt] ; 		
		psi[i] = 0;
	}
	
	/*
		2) Recursion
		 loop for all time steps
	*/
	for(t=1;t<sequence.length; t++) {
		/*
			Loop over all states
		*/		
		for(j=0; j<N; j++) {

			// Compute max_tau max_i  for delta_t(j,1)
			max_i_delta_t_Tij = -Infinity;	
			for(i = 0; i<N; i++) {
				if( i != j) {
					var delta_Tij = delta[(t-1) * N + i] + this.logT.val[i*N+j];

					if( delta_Tij > max_i_delta_t_Tij ) {
						max_i_delta_t_Tij = delta_Tij;
						psi[t * N + j] = i;
					}					
				}					
			}
			
			// Compute delta_t(j)
			xt = this.outputs.indexOf( sequence[t] );
			delta[t * N + j] = max_i_delta_t_Tij + this.logE.val[j * Noutputs + xt] ;						
		}
	}

	// 3) Termination
		
	// MAX_i
	var logLikelihood = delta[(sequence.length-1)*N]; 	//assume best i=0
	Qopt[sequence.length-1] = 0;
	for(i=1;i<N;i++) { 				// then update by looking for best i
		if( delta[(sequence.length-1)*N + i]  > logLikelihood ) {
			logLikelihood = delta[(sequence.length-1)*N + i];
			Qopt[sequence.length-1] = i;
		}
	}
	
	// 4) Backtracking
	var t = sequence.length-2;
	while(t >= 0) {
		// effective state transition
		Qopt[t] = psi[ (t+1) * N + Qopt[t+1] ];
		t--;
	}
	
	return {logLikelihood: logLikelihood, Q: Qopt};
}
