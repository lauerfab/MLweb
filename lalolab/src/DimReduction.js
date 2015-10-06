// Generic class for Dimensionality Reduction
function DimReduction (algorithm, params ) {
	
	if ( typeof(algorithm) == "undefined" ) {
		var algorithm = PCA;
	}

	this.type = "DimReduction:" + algorithm.name;

	this.algorithm = algorithm.name;
	this.userParameters = params;

	// Functions that depend on the algorithm:
	this.construct = algorithm.prototype.construct; 

	this.train = algorithm.prototype.train; 
	if (  algorithm.prototype.reduce ) 
		this.reduce = algorithm.prototype.reduce; // otherwise use default function for linear model
	if (  algorithm.prototype.unreduce ) 
		this.unreduce = algorithm.prototype.unreduce; // otherwise use default function for linear model	
	
	// Initialization depending on algorithm
	this.construct(params);
}

DimReduction.prototype.construct = function ( params ) {
	// Read params and create the required fields for a specific algorithm
	
	// Default parameters:

	this.dimension = undefined; 
	
	// Set parameters:
	if ( typeof(params) == "number")
		this.dimension = params;
	else {
		var i;
		if ( params) {
			for (i in params)
				this[i] = params[i]; 
		}		
	}	

}

DimReduction.prototype.train = function ( X ) {
	// Training function: should set trainable parameters of the model and return reduced X
	var Xreduced;
	
	// Return X reduced:
	return Xreduced;
}

DimReduction.prototype.reduce = function (X) {
	// Default prediction function : apply linear dimensionality reduction to X 
	if ( type(X) == "matrix") {
		var Xc = zeros(X.m, X.n);
		var i;
		for ( i=0; i< X.length; i++) 
			Xc.row(i).set( subVectors(X.row(i) , this.means) );
		
		return mul(Xc, this.Q);
	}
	else {
		var Xc = sub(X, this.means);
		return transpose(mul(transpose(Xc), this.Q));
	}	
}

DimReduction.prototype.unreduce = function (Xr) {
	// Recover (up to compression level) the original X from a reduced Xr
	if ( type(Xr) == "matrix") {
		var X = mul(Xr, transpose(this.Q));
		var i;
		var j;
		for ( i=0; i< X.length; i++) 
			for(j=0; j < X.n; j++)
				X.val[i*X.n+j] += this.means[j];
		
		return X;
	}
	else {
		if ( this.dimension > 1 ) 
			return add(mul(this.Q, Xr) , this.means); // single data 
		else {
			var X = outerprod(Xr, this.Q);	// multiple data in dimension 1
			var i;
			var j;
			for ( i=0; i< X.length; i++) 
				for(j=0; j < X.n; j++)
					X.val[i*X.n+j] += this.means[j];
			return X;
		}
	}	
}


DimReduction.prototype.info = function () {
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
				str += i + ": matrix of size " + this[i].length + "-by-" + this[i].n + "<br>";
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

/////////////////////////////////////
/// Pincipal Component Analysis (PCA)
//
//	main parameter: { dimension: integer} 
/////////////////////////////////////

function PCA ( params) {
	var that = new DimReduction ( PCA, params);
	return that;
}
PCA.prototype.construct = function (params) {
	// Default parameters:

	this.dimension = undefined; // automatically set by the method by default
	this.energy = 0.8;	// cumulative energy for PCA automatic tuning
	
	// Set parameters:
	if ( typeof(params) == "number")
		this.dimension = params;
	else {
		var i;
		if ( params) {
			for (i in params)
				this[i] = params[i]; 
		}		
	}	

}

PCA.prototype.train = function ( X ) {
	// Training function: set trainable parameters of the model and return reduced X
	var Xreduced;
	var i;
	
	// center X :
	this.means = mean(X, 1).val;
	var Xc = zeros(X.m, X.n);
	for ( i=0; i< X.length; i++) 
		Xc.row(i).set ( subVectors(X.row(i), this.means) );
	
	if ( this.dimension && this.dimension < X.n / 3) {
		// Small dimension => compute a few eigenvectors
		var C = mul(transpose(Xc), Xc);	// C = X'X = covariance matrix
		var eigendecomposition = eigs(C, this.dimension);
		this.Q = eigendecomposition.U;
		this.energy = sum(eigendecomposition.V); // XXX divided by total energy!!
		
		Xreduced = mul(Xc, this.Q);
	}
	else {
		// many or unknown number of components => Compute SVD: 
		var svdX = svd(Xc, true);
		var totalenergy = mul(svdX.s,svdX.s);
		
		if ( typeof(this.dimension) == "undefined" ) {
			// set dimension from cumulative energy
			
			var cumulativeenergy = svdX.s[0] * svdX.s[0];
			i = 1;
			while ( cumulativeenergy < this.energy * totalenergy) {
				cumulativeenergy += svdX.s[i] * svdX.s[i];
				i++;
			}
			this.dimension = i;
			this.energy = cumulativeenergy / totalenergy;
		}
		else {
			var singularvalues=  get(svdX.s, range(this.dimension));
			this.energy = mul(singularvalues,singularvalues) / totalenergy;
		}		
		
		// take as many eigenvectors as dimension: 
		this.Q = get(svdX.V, [], range(this.dimension) );		
		
		Xreduced = mul( get(svdX.U, [], range(this.dimension)), get( svdX.S, range(this.dimension), range(this.dimension)));
	}
	
	// Return X reduced:
	return Xreduced;
}



/////////////////////////////////////
/// Locally Linear Embedding (LLE)
//
//	main parameter: { dimension: integer, K: number of neighbors} 
/////////////////////////////////////

function LLE ( params) {
	var that = new DimReduction ( LLE, params);
	return that;
}
LLE.prototype.construct = function (params) {
	// Default parameters:

	this.dimension = undefined; // automatically set by the method by default
	this.K = undefined // auto set to min(dimension + 2, original dimension); 
	
	// Set parameters:
	if ( typeof(params) == "number")
		this.dimension = params;
	else {
		var i;
		if ( params) {
			for (i in params)
				this[i] = params[i]; 
		}		
	}	

}

LLE.prototype.train = function ( X ) {
	// Training function: set trainable parameters of the model and return reduced X

	var i,j,k;
	
	const N = X.m;
	const d = X.n;
	
	if ( typeof(this.dimension) == "undefined")
		this.dimension = Math.floor(d/4);
	
	if(typeof(this.K ) == "undefined" )
		this.K = Math.min(d, this.dimension+2); 
		
	const K = this.K; 
	
	// Compute neighbors and weights Wij 
	var I_W = eye(N); // (I-W)
	var neighbors; 
	
	for ( i= 0; i < N; i++) {
		neighbors = getSubVector(knnsearch(K+1, X.row(i), X).indexes, range(1,K+1)); // get K-NN excluding the point Xi
	
		// matrix G
		var G = zeros(K,K); 
		for (j=0; j < K; j++) {
			for ( k=0; k < K; k++) {
				G.val[j*K+k] = dot(subVectors(X.row(i) , X.row(neighbors[j])),subVectors(X.row(i) , X.row(neighbors[k])));
			}
		}
		
		// apply regularization?
		if ( K > d ) {
			const delta2 = 0.01;
			const traceG = trace(G);
			var regul = delta2 / K;
			if ( traceG > 0 )
				regul *= traceG;
				
			for (j=0; j < K; j++) 
				G.val[j*K+j] += regul;
		}
		
		var w = solve(G, ones(K));

		// rescale w and set I_W = (I-W):
		var sumw = sumVector(w);
		var ri = i*N;		
		for ( j=0; j < K; j++) 
			I_W.val[ri + neighbors[j] ] -= w[j] / sumw;	
	}
	
	var usv = svd(I_W, "V"); // eigenvectors of M=(I-W)'(I-W) are singular vectors of (I-W)
	var Xreduced = get(usv.V, [], range(N-this.dimension-1, N-1) ); // get d first eigenvectors in the d+1 last (bottom) eigenvectors
	
	/* should do this faster as below, but eigs fails due to difficulties with inverse iterations to yield orthogonal eigenvectors for the bottom eigenvalues that are typically very close to each other
	
	var M = xtx(I_W);
	var Xreduced = get(eigs(M, this.dimension+1, "smallest").U, [], range( this.dimension) ); // get d first eigenvectors in the d+1 last (bottom) eigenvectors
	*/
	
	// Set model parameters
	this.W = I_W; // XXX
	
	// Return X reduced:
	return Xreduced;
}

////////////////////////////////
// LTSA: Local Tangent Space Alignment
///////////////////////////////
function ltsa( X , dim, K ) {
	const N = X.m;
	const d = X.n;
	
	if ( typeof(K) == "undefined")
		var K = 8;
	
	var B = zeros(N, N);
    var usesvd = (K > d);
	var neighbors;
	var U;
    for (var i=0; i < N; i++) {
   		neighbors = knnsearchND(K+1,X.row(i), X).indexes;
		var Xi = getRows(X, neighbors );
	    
        Xi = sub( Xi, mul(ones(K+1), mean(Xi,1)) );

        // Compute dim largest eigenvalues of Xi Xi'
		if (usesvd) 
    	    U = getCols(svd(Xi, "U").U, range(dim));
	    else 
	        U = eigs(mulMatrixMatrix(Xi,transposeMatrix(Xi)) , dim);
        
        Gi = zeros(K+1, dim+1);
        set(Gi, [], 0, 1/Math.sqrt(K+1));
        set(Gi, [], range(1,Gi.n), U);
        
        GiGit = mulMatrixMatrix(Gi, transposeMatrix(Gi));
        // local sum into B
        for ( var j=0; j < K+1; j++) {
        	for ( var k=0; k < K+1; k++)
        		B.val[neighbors[j]*N + neighbors[k]] -= GiGit.val[j*(K+1)+k];
        	B.val[neighbors[j]*N + neighbors[j]] += 1;
        }
	}
	var usv = svd(B, "V"); // eigenvectors of B are also singular vectors...
	var Xreduced = get(usv.V, [], range(N-dim-1, N-1) ); // get d first eigenvectors in the d+1 last (bottom) eigenvectors
	return Xreduced;
}

//////////////////////////
/// General tools
/////////////////////////
/** compute the distance matrix for the points 
 *  stored as rows in X
 * @param {Matrix}
 * @return {Matrix}
 */
function distanceMatrix( X ) {
	const N = X.m;
	var i,j;
	var D = zeros(N,N);
	var ri = 0;
	var Xi = new Array(N);
	for ( i=0; i<N; i++) {
		Xi[i] = X.row(i);
		for ( j=0; j < i; j++) {
			D.val[ri + j] = norm(subVectors(Xi[i], Xi[j]));
			D.val[j*N + i] = D.val[ri + j];
		}
		ri += N;
	}
	return D;	
}
