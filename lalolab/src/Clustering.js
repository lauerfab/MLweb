////////////////////////////////////////////
///	Clustering functions for the ML library
/////////////////////////////////////////////

function spectralclustering ( X , n, sigma ) {
	const N = X.length;
	switch( type(sigma) ) {
		case "undefined": 
			var A = kernelMatrix(X, "rbf", 1) ; // Affinity Matrix with sigma = 1 by default
			break;
		case "number": 
			var A = kernelMatrix(X, "rbf", sigma) ; // Affinity Matrix
			break;
		case "matrix": 
			var A = sigma ; // custom Affinity Matrix provided
			break;
		case "function": 	
			// custom Affinity function provided
			var A = zeros(N,N);
			for ( var i = 1; i< N; i++) {
				for ( var j=0; j<i; j++) {
					A.val[i*A.n + j] = sigma(X.row(i), X.row(j));
					A.val[j*A.n + i] = A.val[i*A.n + j];
				}
			}
			break;
		default:
			return "Invalid 3rd parameter value.";
			break;
	}

	if ( typeof(sigma) != "function") 
		A = sub(A, diag(diag(A))) ; // zero diagonal of A
	
	// normalize  as in A. Ng et al., 2002:
	var D = sum(A, 2); 
	var D2 = diag(entrywisediv(1, sqrt(D))); // diag(D)^(-1/2);    
	var Ap = mul(D2, mul(A, D2));

	/*
	// Shi-Malik: 
	// in place A <- L = D^-1 A  (should be I - D^-A , but we use largest egienvalues instead of smallest)
	for ( var i=0; i < N; i++) {
		for ( var j=0; j < N; i++) 
			A.val[i*A.n + j] /= D[i];
	}
	*/

	// find eigenvectors
	var eigen = eigs(Ap,n);

    
    // Normalize rows of U
    var i;
    var U = matrixCopy(eigen.U);
    var normU = norm(U,2); 
	for ( i=0; i< U.m ; i++) {
		for ( var j=0; j < U.n; j++)
			U.val[i*U.n+j] /= normU[i]; 
 	}

 	// Kmeans clustering
	var labels = kmeans(U , n).labels;
	return labels;	
}

function kmeans(X, n, restarts) {
	
	if ( type ( X) == "vector") {
		// Exact algorithm for one-dimensional data:
		return kmeansexact1D(X, n, 0);
	}
	if( typeof(restarts) == "undefined" ) {
		var restarts = 100;
	}
	
	// box bounds for centers:
	var minX = transpose(min(X,1));
	var maxX = transpose(max(X,1));
	
	// do multiple kmeans restarts
	var r;
	var res = {};
	res.cost = Infinity;
	var clustering;
	for ( r=0; r<restarts; r++) {
		clustering = kmeans_single(X, n, minX, maxX); // single shot kmeans
		
		//console.log(clustering.cost);
		
		if ( clustering.cost < res.cost ) {
			res.labels = vectorCopy(clustering.labels);
			res.centers = matrixCopy(clustering.centers);
			res.cost = clustering.cost
		}
	}
	
	return res;
}

function kmeans_single(X, n, minX, maxX ) {
	var i;
	var j;
	var k;

	const N = X.m;
	const d = X.n;

	if ( typeof(minX) == "undefined")
		var minX = transpose(min(X,1));
	if( typeof(maxX) == "undefined")
		var maxX = transpose(max(X,1));
		
	var boxwidth = sub(maxX, minX);
	
	// initialize centers:
	var centers = zeros(n,d); // rand(n,d);
	for ( k=0; k < n; k++) {
		set(centers, k, [], add( entrywisemul(rand(d), boxwidth) , minX) );
	}

	var labels = new Array(N);
	var diff;
	var distance = zeros(n);
	var Nk = new Array(n);
	var normdiff;
	var previous;
	var updatecenters = zeros(n,d);
	do {
		// Zero number of points in each class
		for ( k=0; k< n; k++) {
			Nk[k] = 0;
			for ( j=0; j<d; j++)
				updatecenters.val[k*d + j] = 0;
		}
		
		// Classify data
		for ( i=0; i < N; i++) {
			var Xi = X.val.subarray(i*d, i*d+d); 
			
			for ( k=0; k < n; k++) {
				diff = sub ( Xi , centers.val.subarray(k*d, k*d+d) ) ;
				distance[k] = dot(diff,diff); // = ||Xi - ck||^2
			}
			labels[i] = findmin(distance);

			// precompute update of centers			
			for ( j=0; j < d; j++)
				updatecenters.val[ labels[i] * d + j] += Xi[j]; 
			
			Nk[labels[i]] ++; 
		}

		// Update centers:
		previous = matrixCopy(centers);
		for (k=0;k < n; k++) {
			if ( Nk[k] > 0 ) {			
				for ( j= 0; j < d; j++) 
					centers.val[k*d+j] = updatecenters.val[k*d+j] / Nk[k]  ;								
								
			}
			else {
				//console.log("Kmeans: dropped one class");				
			}
		}	
		normdiff = norm( sub(previous, centers) );

	} while ( normdiff > 1e-8 );
	 
	// Compute cost
	var cost = 0;
	for ( i=0; i < N; i++) {
		var Xi = X.val.subarray(i*d, i*d+d); 		
		for ( k=0; k < n; k++) {
			diff = sub ( Xi , centers.val.subarray(k*d, k*d+d) ) ;
			distance[k] = dot(diff,diff); // = ||Xi - ck||^2
		}
		labels[i] = findmin(distance);

		cost += distance[labels[i]];
	}

	return {"labels": labels, "centers": centers, "cost": cost};
}

function kmeansexact1D( X, n, start) {

	if ( n <= 1 || start >= X.length -1 ) {
		var cost = variance(get(X,range(start,X.length)));
		return {"cost":cost, "indexes": []};
	}
	else {
		var i;
		var cost;
		var costmin = Infinity;
		var mini = 0;
		var nextcut;
		for ( i= start+1; i < X.length-1; i++) {
			// cut between start and end at i :
			// cost is variance of first half + cost from the cuts in second half
			cost = variance ( get(X, range(start, i) ) ) ;
			nextcut = kmeansexact1D( X, n-1, i)
			cost += nextcut.cost ;
			
			if ( cost < costmin ) {
				costmin = cost;
				mini = i;
			}

		}

		var indexes = nextcut.indexes;
		indexes.push(mini);
		return {"cost": costmin, "indexes": indexes } ; 
	}
}
