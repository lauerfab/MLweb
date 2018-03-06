////////////////////////////////////////////
///	Clustering functions for the ML.js library
/////////////////////////////////////////////

/****************************************
		Spectral clustering

	Implemented as in Ng. et al, NIPS, 2002

	Possible affinity functions (choose via type of sigma): 
	- Gaussian RBF (default) (sigma is the bandwidth)
	- custom function (sigma) computing affinity between two points
	- custom affinity matrix (sigma) computed elsewhere 
*****************************************/
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
	var D2 = spdiag(entrywisediv(1, sqrt(D))); // diag(D)^(-1/2);    
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

/**************************************
		K-means
		
	- including multiple random restarts 
	- exact algorithm for dim(x_i) = 1 tests all linear cuts
		(if X is a vector instead of a matrix)
	
***************************************/
function kmeans(X, n, restarts) {
	
	if ( type ( X) == "vector") {
		// Exact algorithm for one-dimensional data:
		return kmeansexact1D(X, n, 0);
	}
	
	if ( typeof(n) == "undefined" ) {
		// Determine number of clusters using cluster separation measure
		var nmax = 10;
		var validity = new Float64Array(nmax+1);
		validity[0] = Infinity;
		validity[1] = Infinity;
		var n;
		var clustering = new Array(nmax+1); 
		for ( n=2; n <= nmax; n++) {
			clustering[n] = kmeans(X,n,restarts);
			validity[n] = clusterseparation(X,clustering[n].centers,clustering[n].labels);
		}
		var best_n = findmin(validity);
		// go on increasing n while getting better results
		while ( best_n == nmax ) {
			nmax++; 
			clustering[nmax] = kmeans(X,nmax,restarts);
			validity[nmax] = clusterseparation(X,clustering[n].centers,clustering[n].labels);
			if ( validity[nmax] < validity[best_n] )
				best_n = nmax;
		}
		console.log("Clustering validity intra/inter = " , validity, "minimum is at n = " + best_n);
		
		return clustering[best_n];
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

/*
	Compute cluster separation index as in Davis and Bouldin, IEEE PAMI, 1979
	using Euclidean distances (p=q=2).
	
	centers has nGroups rows
*/
function clusterseparation(X, centers, labels) {
	var n = centers.m;
	var dispertions = zeros(n);
	var inter = zeros(n,n);
	for ( var k=0; k<n; k++) {
		var idx = find(isEqual(labels, k));
		
		if ( idx.length == 0 ) {
			return Infinity;
		}
		else if ( idx.length == 1 ) {
			dispertions[k] = 0;
		}
		else {
			var Xk = getRows(X, idx);
			var diff = sub(Xk, outerprod(ones(Xk.length), centers.row(k)));
			diff = entrywisemul(diff,diff);
			var distances = sum(diff, 2);
			dispertions[k] = Math.sqrt(sum(distances) / idx.length );	 
		}
		
		for ( j=0; j < k; j++) {
			inter.val[j*n+k] = norm(sub(centers.row(j), centers.row(k)));
			inter.val[k*n+j] = inter.val[j*n+k];
		}
	}
	var Rkj = zeros(n,n);
	for ( k=0; k < n; k++) {
		for ( j=0; j < k; j++) {
			Rkj.val[k*n+j] = ( dispertions[k] + dispertions[j] ) / inter.val[j*n+k];
			Rkj.val[j*n+k] = Rkj.val[k*n+j];
		}
	}
	var Rk = max(Rkj, 2); 
	var R = mean(Rk);
	return R;
}

function voronoi (x, centers) {
	// Classify x according to a Voronoi partition given by the centers
	var t = type(x);
	if ( t == "matrix" && x.n == centers.n ) {
		var labels = new Float64Array(x.m);
		for (var i=0; i < x.m; i++) {
			labels[i] = voronoi_single(x.row(i), centers);
		}
		return labels;
	}
	else if ( t == "vector" && x.length == centers.n ) {
		return voronoi_single(x, centers);
	}
	else 
		return undefined;
}
function voronoi_single(x, centers) {
	var label;
	var mindist = Infinity; 
	for (var k=0; k < centers.m; k++) {
		var diff = subVectors(x, centers.row(k) );
		var dist = dot(diff,diff); 
		if ( dist < mindist ) {
			mindist = dist;
			label = k;
		}
	}
	return label;
}
/*******************************
	Clustering with Stability analysis for tuning the nb of groups
	(see von Luxburg, FTML 2010)

	X: data matrix
	nmax: max nb of groups
	method: clustering function for fixed nb of groups
			(either kmeans or spectralclustering for now)
	params: parameters of the clustering function
********************************/
function cluster(X, nmax, method, params, nbSamples ) {
	var auto = false;
	if ( typeof(nmax) != "number" ) {
		var nmax = 5;
		var auto = true;
	}
	if ( typeof(nbSamples) != "number" )
		var nbSamples = 3; 
	if ( typeof(method) != "function" ) 
		var method = kmeans; 
	
	if ( method.name != "kmeans" ) {
		error("stability analysis of clustering only implemented for method=kmeans yet");
		return undefined;
	}
	
	var Xsub = new Array(nbSamples);
	var indexes = randperm(X.m); 
	var subsize = Math.floor(X.m / nbSamples);
	for (var b=0; b < nbSamples ; b++ ) {
		Xsub[b] = getRows(X, get(indexes, range(b*subsize, (b+1)*subsize)));	
	}
	
	var best_n;
	var mininstab = Infinity;
	var instab = new Array(nmax+1); 
	
	var clusterings = new Array(nbSamples); 
	
	var compute_instab = function ( n ) {
		for (var b=0; b < nbSamples ; b++ ) {
			clusterings[b] = method( Xsub[b], n, params);			
		} 
		
		// reorder centers and labels to match order of first clustering
		//	(otherwise distances below are subject to mere permutations and meaningless 
		var orderedCenters = new Array(nbSamples);
		orderedCenters[0] = clusterings[0].centers;
		for ( var b = 1; b < nbSamples; b++) {
			var idxtemp = range(n); 
			var idx = new Array(n);
			for ( var c = 0; c < n; c++) {
				var k = 0; 
				var mindist = Infinity; 
				for ( var c2 = 0; c2 < idxtemp.length; c2++) {
					var dist = norm(subVectors (clusterings[b].centers.row(idxtemp[c2]), clusterings[0].centers.row(c) ));
					if ( dist < mindist ) {
						mindist = dist;
						k = c2;
					}
				} 
				idx[c] = idxtemp.splice(k,1)[0];
			}
			orderedCenters[b] = getRows(clusterings[b].centers, idx);  
		}
		// Compute instability as average distance between clusterings
		
		instab[n] = 0;
		for (var b=0; b < nbSamples ; b++ ) {
			for (var b2=0; b2 < b; b2++) {
				if ( method.name == "kmeans" ) {
					for ( var i=0; i < subsize; i++) {
						instab[n] += ( voronoi_single(Xsub[b].row(i) , orderedCenters[b2] ) != voronoi_single(Xsub[b].row(i) , orderedCenters[b]) )?2:0 ; // 2 because sum should loop over symmetric cases in von Luxburg, 2010.
					}
					for ( var i=0; i < subsize; i++) {
						instab[n] += ( voronoi_single(Xsub[b2].row(i) , orderedCenters[b] ) != voronoi_single(Xsub[b2].row(i) ,orderedCenters[b2]) )?2:0 ;
					}
				}
			}
		} 
		instab[n] /= (nbSamples*nbSamples);
		if ( instab[n] < mininstab ) {
			mininstab = instab[n];
			best_n = n;
		}			
		console.log("For n = " + n + " groups, instab = " + instab[n] + " (best is " + mininstab + " at n = " + best_n + ")");
	};
	
	for ( var n = 2; n <= nmax; n++ ) {
	
		compute_instab(n);
	
		if ( isZero(instab[n]) ) 
			break;	// will never find a lower instab than this one			
	}
	
	// go on increasing n while stability increases
	while ( auto && best_n == nmax && !isZero(mininstab) ) {
		nmax++; 
		compute_instab(nmax);
	}
		
	return {clustering: method(X,best_n,params) , n: best_n, instability: instab}; 
}
