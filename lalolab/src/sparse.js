/////////////////////////////
//// Sparse matrix and vectors 
/////////////////////////////

/**
 *
 * new spVector(n) => allocate for n nonzeros with dim n
 * new spVector(n, nnz) => allocate for nnz nonzeros out of n
 * new spVector(n,values,indexes) => allocate for values.length nonzeros
 *
 * @constructor
 * @struct
 */
function spVector(n, values, indexes) {
	
	/** @const */ this.length = n;
	/** @const */ this.size = [n,1];
	/** @const */ this.type = "spvector";
	
	if ( arguments.length <= 2) {
		if ( arguments.length == 1)
			var nnz = n;		// too large but more efficient at some point...
		else
			var nnz = values;
			
		/** @type{Float64Array} */ this.val = new Float64Array(nnz);  // nz values
		/** @type{Uint32Array} */ this.ind = new Uint32Array(nnz);   // ind[k] = index of val[k]
	}
	else {
		var nnz = values.length;
		/** @type{Float64Array} */ this.val = new Float64Array(values);  // nz values
		/** @type{Uint32Array} */ this.ind = new Uint32Array(indexes);   // ind[k] = index of val[k]
	}
	
	/** @const */ this.nnz = nnz;	
}
/*
 * @param{number}
 * @return{number}
 */
spVector.prototype.get = function ( i ) {
	var k = this.ind.indexOf(i);
	if ( k < 0 )
		return 0;
	else
		return this.val[k];
}
/*
 * @param{number}
 * @param{number}
 */
spVector.prototype.set = function ( i, value ) {
	// Inefficient do not use this, use sparse(x) instead
	if ( i > this.n ) {
		error( "Error in spVector.set(i,value): i > this.length)");
		return undefined;
	}
	var k = this.ind.indexOf(i);
	if ( k < 0 ) {
		var ind = new Uint32Array(this.nnz + 1);
		var val = new Float64Array(this.nnz + 1);
		k = 0; 
		while ( this.ind[k] < i ) { // copy values until i
			ind[k] = this.ind[k];	// making sure this.ind remains sorted
			val[k] = this.val.ind[k];
			k++;
		}
		ind[k] = i;// insert value
		val[k] = value;
		ind.set(this.ind.subarray(k), k+1);// copy rest of vector
		val.set(this.val.subarray(k), k+1);
		this.nnz++;
	}
	else 
		this.val[k] = value;
		
	return value;
}
/*
 * @return{spVector}
 */
spVector.prototype.copy = function () {
	return new spVector(this.n, this.val, this.ind);	
}

/**
 *
 * new spMatrix(m,n) => allocate for m*n nonzeros
 * new spMatrix(m,n, nnz) => allocate for nnz nonzeros
 * new spMatrix(m,n,values,cols,rows) => allocate for values.length nonzeros
 *
 * @constructor
 * @struct
 */
function spMatrix(m,n, values, cols, rows) {
	
	/** @const */ this.length = m;
	/** @const */ this.m = m;
	/** @const */ this.n = n;
	/** @const */ this.size = [m,n];
	/** @const */ this.type = "spmatrix";
	
	if ( arguments.length <= 3) {
		if ( arguments.length == 2)
			var nnz = m*n;		// too large but more efficient at some point...
		else
			var nnz = values;
			
		/** @type{boolean} */ this.rowmajor = true;
		/** @type{Float64Array} */ this.val = new Float64Array(nnz);  // nnz values
		/** @type{Uint32Array} */ this.cols = new Uint32Array(nnz); // cols[j] = starting index of col j in val and rows
		/** @type{Uint32Array} */ this.rows = new Uint32Array(m+1);   // rows[k] = row of val[k]
	}
	else {
		var nnz = values.length;
		if ( rows.length == nnz && cols.length == n+1 && cols[cols.length-1] == nnz ) {
			/** @type{boolean} */ this.rowmajor = false;
			/** @type{Float64Array} */ this.val = new Float64Array(values);  // nz values
			/** @type{Uint32Array} */ this.cols = new Uint32Array(cols); // cols[j] = starting index of col j in val and rows
			/** @type{Uint32Array} */ this.rows = new Uint32Array(rows);   // rows[k] = row of val[k]		
		}
		else {
			/** @type{boolean} */ this.rowmajor = true;
			/** @type{Float64Array} */ this.val = new Float64Array(values);  // nz values
			/** @type{Uint32Array} */ this.cols = new Uint32Array(cols); // cols[k] = col of val[k]	
			/** @type{Uint32Array} */ this.rows = new Uint32Array(rows);   // rows[i] = starting index of row i in val and cols
		}
	}
	
	/** @const */ this.nnz = nnz;
	
}
/*
 * @return{spMatrix}
 */
spMatrix.prototype.copy = function () {
	return new spMatrix(this.m, this.n, this.val, this.cols, this.rows);	
}
/*
 * @return{spMatrix}
 */
spMatrix.prototype.toRowmajor = function () {
	if ( this.rowmajor ) 
		return this.copy();
	else {
		return sparseMatrixRowMajor( fullMatrix(this) );
	}
}
/*
 * Get a pointer to the spVector for row i
 * @return{spVector}
 */
spMatrix.prototype.row = function ( i ) {
	if ( this.rowmajor ) {
		return new spVector(this.n, this.val.subarray(this.rows[i], this.rows[i+1]), this.cols.subarray(this.rows[i], this.rows[i+1]));	
	/*
		var s = this.rows[i];
		var e = this.rows[i+1];
		var vec = new spVector(this.n);
		vec.val.set(this.val.subarray(s,e));
		vec.ind.set(this.cols.subarray(s,e));
		return vec;*/
	}
	else {
		error ("Cannot extract sparse column from a sparse matrix in row major format.");
		return undefined;
	}
}
/*
 * Get a pointer to the spVector for column j
 * @return{spVector}
 */
spMatrix.prototype.col = function ( j ) {
	if ( ! this.rowmajor )
		return new spVector(this.m, this.val.subarray(this.cols[j], this.cols[j+1]), this.rows.subarray(this.cols[j], this.cols[j+1]));	
	else {
		error ("Cannot extract sparse column from a sparse matrix in row major format.");
		return undefined;
	}
}

/*
 * @param{number}
 * @param{number} 
 * @return{number}
 */
spMatrix.prototype.get = function ( i, j ) {
	if ( this.rowmajor ) {
		var rowind =  this.cols.subarray(this.rows[i], this.rows[i+1]);
		var k = rowind.indexOf(j);
		if ( k < 0 )
			return 0;
		else
			return this.val[this.rows[i] + k];	
	}
	else {
		var colind =  this.rows.subarray(this.cols[j], this.cols[j+1]);
		var k = colind.indexOf(i);
		if ( k < 0 )
			return 0;
		else
			return this.val[this.cols[j] + k];
	}
}

function spgetRows(A, rowsrange) {
	var n = rowsrange.length;
	if ( A.rowmajor) {
		if ( n > 1 ) {

			var rowsidx = sort(rowsrange);
			var Ai = new Array(n);
			var nnz = 0;
			for ( var i = 0; i < n; i++) {
				Ai[i] = A.row(rowsidx[i]);
				nnz += Ai[i].val.length;
			}
			var val = new Float64Array( nnz );
			var cols = new Uint32Array( nnz );
			var rows = new Uint32Array( n+1 );
			var k = 0;
			for ( var i = 0; i < n; i++) {
				rows[i] = k;
				val.set(Ai[i].val, k);
				cols.set(Ai[i].ind, k);
				k += Ai[i].val.length;
			}
			rows[i] = k;
			return new spMatrix(n, A.n, val, cols, rows);
		}
		else
			return A.row( rowsrange[0] ) ;
	}
	else {
		return getRows(fullMatrix(A), rowsrange);
	}
}

/**
 * Return the full/dense version of the vector
 * @param{spVector} 
 * @return{Float64Array}
 */
function fullVector (x) {
	var k;
	const n = x.length;
	const nnz = x.val.length;
	var a = new Float64Array(n);
	
	for ( k=0; k < nnz; k++) 
		a[x.ind[k]] = x.val[k];
	
	return a;
}
/**
 * Return the full/dense version of the matrix
 * @param{spMatrix} 
 * @return{Matrix}
 */
function fullMatrix (S) {
	const n = S.n;
	if ( S.rowmajor ) {
		var k;
		const m = S.m;
		var A = new Float64Array(m * n);
		var ri = 0;
		for (var i = 0; i < m; i++) {
			var s = S.rows[i];
			var e = S.rows[i+1];
			for ( k=s; k < e; k++) {
				A[ri + S.cols[k] ] = S.val[k];
			}
			ri += n;
		}
		return new Matrix(m, n, A, true);
	}
	else {
		var k;
		var A = new Float64Array(S.m * n);
		for (var j = 0; j < n; j++) {
			var s = S.cols[j];
			var e = S.cols[j+1];
			for ( k=s; k < e; k++) {
				var i = S.rows[k];
				A[i*n + j] = S.val[k];
			}
		}
		return new Matrix(S.m, n, A, true);
	}
}
function full( A ) {
	switch(type(A)) {
	case "spvector": 
		return fullVector(A);
		break;
	case "spmatrix":
		return fullMatrix(A);
		break;
	default:
		return A;
		break;
	}
}

/**
 * @param{Float64Array}
 * @return{spVector}
 */
function sparseVector( a ) {
	var i,k;
	const n = a.length;
	var val = new Array();
	var ind = new Array();
	for ( i=0; i < n; i++) {
		if (!isZero(a[i]) ) {
			val.push(a[i]);
			ind.push(i);
		}
	}		
	return new spVector(n,val,ind);
}
/**
 * @param{Matrix}
 * @return{spMatrix}
 */
function sparseMatrix( A ) {
	var i,j;
	const m = A.m;
	const n = A.n;
	var val = new Array();
	var rows = new Array();
	var cols = new Uint32Array(n+1);
	var k;
	for ( j=0; j< n; j++) {
		k = j;
		for ( i=0; i < m; i++) {
			// k = i*n+j;
			if (!isZero(A.val[k]) ) {
				val.push(A.val[k]);
				rows.push(i);
				cols[j+1]++;
			}	
			k += n;	
		}		
	}	
	for ( j=1; j< n; j++) 
		cols[j+1] += cols[j];
	
	return new spMatrix(m,n,val,cols,rows);
}
/**
 * @param{Matrix}
 * @return{spMatrix}
 */
function sparseMatrixRowMajor( A ) {
	var i,j;
	const m = A.m;
	const n = A.n;
	var val = new Array();
	var cols = new Array();
	var rows = new Uint32Array(m+1);
	var k = 0;
	for ( i=0; i < m; i++) {
		for ( j=0; j< n; j++) {
			// k = i*n+j;
			if (!isZero(A.val[k]) ) {
				val.push(A.val[k]);
				rows[i+1]++;
				cols.push(j); 
			}		
			k++;
		}		
	}	
	for ( i=1; i< m; i++) 
		rows[i+1] += rows[i];
	
	return new spMatrix(m,n,val,cols,rows);
}

function sparse( A , rowmajor ) {
	if(typeof(rowmajor) == "undefined" ) 
		var rowmajor = true;
		
	switch(type(A)) {
	case "vector": 
		return sparseVector(A);
		break;	
	case "matrix":
		if ( rowmajor )
			return sparseMatrixRowMajor(A);
		else
			return sparseMatrix(A);
		break;
	case "spvector":
	case "spmatrix":
		return A.copy();
		break;
	default:
		return A;
		break;
	}
}

/**
 * @param{number}
 * @return{spMatrix}
 */
function speye(n) {
	if ( n == 1)
		return 1;
	var val = ones(n);
	var rows = range(n+1);
	var cols = rows.slice(0,n);
	return new spMatrix(n,n,val,cols,rows);
}
/**
 * @param{Float64Array}
 * @return{spMatrix}
 */
function spdiag(val) {
	var n = val.length;
	var rows = range(n+1);
	var cols = rows.slice(0,n);
	var tv = type(val);
	if ( tv == "vector")
		return new spMatrix(n,n,val,cols,rows);
	else {
		error("Error in spdiag( x ): x is a " + tv + " but should be a vector.");
		return undefined;
	}
}

/**
 * @param{spVector}
 * @return{Matrix}
 */
function transposespVector (a) {
	return new Matrix(1,a.length, fullVector(a), true);
}
/**
 * @param{spMatrix}
 * @return{spMatrix}
 */
function transposespMatrix (A) {
	return new spMatrix(A.n, A.m, A.val, A.rows, A.cols);
	/*
	const m = A.m;
	const n = A.n;
	
	var At = zeros(n, m);	
	for ( var j=0; j < n; j++) {
		var s = A.cols[j];
		var e = A.cols[j+1];

		for ( var k=s;k < e; k++) {
			At[ rj + A.rows[k] ] = A.val[k];
		}
		rj += m;
	}
	return sparseMatrix(At);
	*/
}



/** Concatenate sparse matrices/vectors
 * @param {Array} 
 * @param {boolean}
 * @return {spMatrix}
 */
function spmat( elems, rowwise ) {
	var k;
	var elemtypes = new Array(elems.length);
	for ( k=0; k < elems.length; k++) {
		elemtypes[k] = type(elems[k]);
	}
		
	if ( typeof(rowwise) == "undefined")
		var rowwise = true;
		
	if ( elems.length == 0 ) {
		return []; 
	}

	var m = 0;
	var n = 0;
	var nnz = 0;
	var i;
	var j;
	if ( rowwise ) {
		var res = new Array( ) ;
		
		for ( k= 0; k<elems.length; k++) {
			switch( elemtypes[k] ) {

			case "vector": // vector (auto transposed)
				var v = sparseVector(elems[k]);
				res.push ( v ) ;
				m += 1;
				n = elems[k].length;
				nnz += v.val.length;
				break;			
			
			case "spvector":
				res.push(elems[k]);
				n = elems[k].length;
				m += 1;
				nnz += elems[k].val.length;
				break;
				
			case "spmatrix":
				for ( var r=0; r < elems[k].m; r++)
					res.push(elems[k].row(r));
				res.push(elems[k]);
				n = elems[k].length;
				m += 1;
				nnz += elems[k].val.length;
				
				break;
				
			default:
				return undefined;
				break;
			}
		}
		
		var M = new spMatrix( m , n , nnz ) ;
		var p = 0;
		M.rows[0] = 0;		
		for (k=0; k < res.length ; k++) {
			if ( res[k].val.length > 1 ) {
				M.val.set( new Float64Array(res[k].val), p);
				M.cols.set( new Uint32Array(res[k].ind), p);
				M.rows[k+1] = M.rows[k] + res[k].val.length;
				p += res[k].val.length;
			}
			else if (res[k].val.length == 1) {
				M.val[p] = res[k].val[0];
				M.cols[p] = res[k].ind[0];
				M.rows[k+1] = M.rows[k] + 1;
				p += 1;			
			}
				
		}
		return M;
	}
	else {
		// not yet...
		
		error("spmat(..., false) for columnwise concatenation of sparse vectors not yet implemented");
		
		return res;
	}
}



/**
 * @param{number}
 * @param{spVector}
 * @return{spVector}
 */
function mulScalarspVector (a, b) {
	const nnz = b.val.length;
	var c = b.copy();
	for ( var k=0;k < nnz; k++) 
		c.val[k] *= a;	
	return c;
}
/**
 * @param{number}
 * @param{spMatrix}
 * @return{spMatrix}
 */
function mulScalarspMatrix (a, B) {
	const nnz = B.nnz;
	var C = B.copy();
	for ( var k=0;k < nnz; k++) 
		C.val[k] *= a;	
	return C;
}

/**
 * @param{spVector}
 * @param{spVector}
 * @return{number}
 */
function spdot (a, b) {
	const nnza = a.val.length;
	const nnzb = b.val.length;
	var c = 0;
	var ka = 0;
	var kb = 0;	
	while ( ka < nnza && kb < nnzb ){
		var i = a.ind[ka]; 
		while ( b.ind[kb] < i && kb < nnzb)
			kb++;
		if(b.ind[kb] == i)
			c += a.val[ka] * b.val[kb];	
		ka++;
	}
	return c;
}
/**
 * @param{spVector}
 * @param{Float64Array}
 * @return{number}
 */
function dotspVectorVector (a, b) {
	const nnza = a.val.length;
	var c = 0;
	for ( var ka=0;ka < nnza; ka++) 
		c += a.val[ka] * b[a.ind[ka]];
	
	return c;
}
/**
 * @param{Matrix}
 * @param{spVector}
 * @return{Float64Array}
 */
function mulMatrixspVector (A, b) {
	const m = A.m;
	const n = A.n;
	const nnz = b.val.length;
	var c = zeros(m);
	var ri = 0;
	for ( var i=0;i < n; i++) {
		for ( var k=0; k < nnz; k++) 
			c[i] += A.val[ri + b.ind[k]] * b.val[k];
		ri+=n;
	}
	return c;
}
/**
 * @param{spMatrix}
 * @param{Float64Array}
 * @return{Float64Array}
 */
function mulspMatrixVector (A, b) {
	const m = A.m;
	const n = A.n;
	var c = zeros(m);
	if ( A.rowmajor) {
		for(var i=0; i < m; i++) {
			var s = A.rows[i];
			var e = A.rows[i+1];
			for(var k = s; k < e; k++) {
				c[i] += A.val[k] * b[A.cols[k]];
			}
		}
	}
	else {
		for ( var j=0;j < n; j++) {
			var s = A.cols[j];
			var e = A.cols[j+1];
			var bj = b[j];
			for ( var k= s; k < e; k++) {
				c[A.rows[k]] += A.val[k] * bj;
			}
		}
	}
	return c;
}
/**
 * @param{spMatrix}
 * @param{Float64Array}
 * @return{Float64Array}
 */
function mulspMatrixTransVector (A, b) {
	const m = A.m;
	const n = A.n;
	var c = zeros(n);
	if ( A.rowmajor ) {
		for ( var j=0;j < m; j++) {
			var s = A.rows[j];
			var e = A.rows[j+1];
			var bj = b[j];
			for ( var k= s; k < e; k++) {
				c[A.cols[k]] += A.val[k] * bj;
			}
		}
	}
	else {
		for ( var j=0;j < n; j++) {
			var s = A.cols[j];
			var e = A.cols[j+1];
			for ( var k= s; k < e; k++) {
				c[j] += A.val[k] * b[A.rows[k]];
			}
		}
	}
	return c;
}
/**
 * @param{spMatrix}
 * @param{spVector}
 * @return{Float64Array}
 */
function mulspMatrixspVector (A, b) {
	const m = A.m;
	const n = A.n;
	var c = zeros(m);
	const nnzb = b.val.length;
	if ( A.rowmajor) {
		for(var i=0; i < m; i++) {
			c[i] = spdot(A.row(i), b);
		}
	}
	else {
		for ( var kb=0;kb < nnzb; kb++) {
			var j = b.ind[kb];		
			var bj = b.val[kb];
			var s = A.cols[j];
			var e = A.cols[j+1];

			for ( var k= s; k < e; k++) {
				c[A.rows[k]] += A.val[k] * bj;
			}
		}
	}
	return c;
}
/**
 * @param{spMatrix}
 * @param{spVector}
 * @return{Float64Array}
 */
function mulspMatrixTransspVector (A, b) {
	const m = A.m;
	const n = A.n;
	var c = zeros(n);
	const nnzb = b.val.length;
	if (A.rowmajor) {
		for ( var kb=0;kb < nnzb; kb++) {
			var j = b.ind[kb];		
			var bj = b.val[kb];
			var s = A.rows[j];
			var e = A.rows[j+1];
			for ( var k= s; k < e; k++) {
				c[A.cols[k]] += A.val[k] * bj;
			}
		}
	}
	else {
		for ( var i= 0; i < n; i++) {
			var kb = 0;
			var s = A.cols[i];
			var e = A.cols[i+1];

			for ( var ka=s;ka < e; ka++) {
				var j = A.rows[ka]; 
				while ( b.ind[kb] < j && kb < nnzb)
					kb++;
				if(b.ind[kb] == i)
					c[i] += A.val[ka] * b.val[kb];	
			}
		}
	}
	return c;
}
/**
 * @param{spMatrix}
 * @param{spMatrix} 
 * @return{Matrix}
 */
function mulspMatrixspMatrix (A, B) {
	const m = A.m;
	const n = A.n;
	const n2 = B.n;
	var c = zeros(m, n2);

	if ( A.rowmajor ) {
		if ( B.rowmajor ) {
			for ( var ic = 0; ic < m; ic++) {
				var sa = A.rows[ic];
				var ea = A.rows[ic+1];
	
				for ( var ka = sa; ka < ea; ka++) {
					var j = A.cols[ka];
					var aj = A.val[ka];
		
					var s = B.rows[j];
					var e = B.rows[j+1];

					var rc = ic * n2 ;
					for (var k= s; k < e; k++) {						
						c.val[rc + B.cols[k] ] += aj * B.val[k] ;
					}
				}
			}
		}
		else {
			var kc = 0;
			for ( var i=0; i < m; i++) {
				for ( var j=0; j < n2; j++) {
					c.val[kc] = spdot(A.row(i), B.col(j));
					kc++;
				}
			}
		}
	}
	else {
		if ( B.rowmajor ) {
			for (var ja=0;ja < n; ja++) {
				var sa = A.cols[ja];
				var ea = A.cols[ja+1];
				var sb = B.rows[ja];
				var eb = B.rows[ja+1];					
				for ( var ka = sa; ka < ea; ka++) {
					var rc = A.rows[ka] * n2;
					var aij = A.val[ka];
					
					for(var kb = sb; kb < eb; kb++) {
						c.val[rc  + B.cols[kb]] += aij * B.val[kb];	
					}										
				} 
			}
		}
		else {
			for ( var jc = 0; jc < n2; jc++) {
				var sb = B.cols[jc];
				var eb = B.cols[jc+1];
	
				for ( var kb = sb; kb < eb; kb++) {
					var j = B.rows[kb];
					var bj = B.val[kb];
		
					var s = A.cols[j];
					var e = A.cols[j+1];

					for (var k= s; k < e; k++) {
						c.val[A.rows[k] * n2 + jc] += A.val[k] * bj;
					}
				}
			}
		}
	}
	return c;
}
/**
 * @param{Matrix}
 * @param{spMatrix} 
 * @return{Matrix}
 */
function mulMatrixspMatrix (A, B) {
	const m = A.m;
	const n = A.n;
	const n2 = B.n;
	var c = zeros(m, n2);
	
	if ( B.rowmajor ) {
		for (var ja=0;ja < n; ja++) {
			var sb = B.rows[ja];
			var eb = B.rows[ja+1];					
			for ( var i = 0; i < m; i++) {
				var rc = i * n2;
				var aij = A.val[i * n + ja];
				
				for(var kb = sb; kb < eb; kb++) {
					c.val[rc  + B.cols[kb]] += aij * B.val[kb];	
				}										
			}
		}
	}
	else {
		for ( var jc = 0; jc < n2; jc++) {
			var sb = B.cols[jc];
			var eb = B.cols[jc+1];
	
			for ( var kb = sb; kb < eb; kb++) {
				var j = B.rows[kb];
				var bj = B.val[kb];
		
				for ( i= 0; i < m; i++) {
					c.val[i * n2 + jc] += A.val[i*n + j] * bj;
				}
			}
		}
	}
	return c;
}

/**
 * @param{spMatrix}
 * @param{Matrix} 
 * @return{Matrix}
 */
function mulspMatrixMatrix (A, B) {
	const m = A.m;
	const n = A.n;
	const n2 = B.n;
	var c = zeros(m, n2);

	if ( A.rowmajor ) {
		for(var i=0; i < m; i++) {
			var sa = A.rows[i];
			var ea = A.rows[i+1];
			for(var ka = sa; ka < ea; ka++) {
				var ai = A.val[ka];
				var rb = A.cols[ka] * n2;
				var rc = i*n2;
				for ( j=0; j < n2; j++) {
					c.val[rc + j] += ai * B.val[rb + j];
				}				
			}
		}
	}
	else {
		for(var j=0; j < n; j++) {
			var s = A.cols[j];
			var e = A.cols[j+1];

			for ( var k= s; k < e; k++) {
				var i = A.rows[k];
				for ( var jc = 0; jc < n2; jc++) 
					c.val[i*n2 + jc ] += A.val[k] * B.val[j*n2 + jc];
			}
		}
	}
	return c;
}

/**
 * @param{spVector}
 * @param{spVector}
 * @return{spVector}
 */
function entrywisemulspVectors (a, b) {
	const nnza = a.val.length;
	const nnzb = b.val.length;
	var val = new Array();
	var ind = new Array();
	
	var ka = 0;
	var kb = 0;	
	while ( ka < nnza && kb < nnzb ){
		var i = a.ind[ka]; 
		while ( b.ind[kb] < i && kb < nnzb)
			kb++;
		if(b.ind[kb] == i) {
			var aibi = a.val[ka] * b.val[kb];
			if ( !isZero(aibi) ) {
				val.push(aibi);	
				ind.push(i);
			}
		}
		ka++;
	}
	return new spVector(a.length, val, ind);
}
/**
 * @param{spVector}
 * @param{Float64Array}
 * @return{spVector}
 */
function entrywisemulspVectorVector (a, b) {
	// fast operation but might not yield optimal nnz:
	var c = a.copy();	
	const nnz = a.val.length;
	for ( var k = 0; k< nnz; k++) {
		c.val[k] *= b[a.ind[k]];
	}
	return c;
}
/**
 * @param{spMatrix}
 * @param{spMatrix}
 * @return{spMatrix}
 */
function entrywisemulspMatrices (A, B) {
	if ( A.rowmajor ) {
		if ( B.rowmajor ) {
			var val = new Array();
			var cols = new Array();
			var rows = new Uint32Array(A.m+1);
			var ka;
			var kb;
			var i;	
			for ( i=0; i < A.m; i++) {
				ka = A.rows[i];
				kb = B.rows[i];
				var ea = A.rows[i+1];
				var eb = B.rows[i+1];
				while ( ka < ea & kb < eb ){
					var j = A.cols[ka]; 
					while ( B.cols[kb] < j && kb < eb)
						kb++;
					if(B.cols[kb] == j) {
						val.push(A.val[ka] * B.val[kb]);	
						cols.push(j);
						rows[i+1]++;
					}
					ka++;
				}
			}
			for(i=1; i < A.m; i++)
				rows[i+1] += rows[i];
				
			return new spMatrix(A.m, A.n, val, cols, rows);
		}
		else {
			return entrywisemulspMatrixMatrix(B, fullMatrix(A)); // perhaps not the fastest
		}
	}
	else {
		if ( B.rowmajor ) {
			return entrywisemulspMatrixMatrix(A, fullMatrix(B)); // perhaps not the fastest
		}
		else {
			var val = new Array();
			var cols = new Uint32Array(A.n+1);
			var rows = new Array();
			var ka;
			var kb;	
			var j;
			for ( j=0; j < A.n; j++) {
				ka = A.cols[j];
				kb = B.cols[j];
				var ea = A.cols[j+1];
				var eb = B.cols[j+1];
				while ( ka < ea & kb < eb ){
					var i = A.rows[ka]; 
					while ( B.rows[kb] < i && kb < eb)
						kb++;
					if(B.rows[kb] == i) {
						val.push(A.val[ka] * B.val[kb]);	
						rows.push(i);
						cols[j+1]++;
					}
					ka++;
				}
			}
			for ( j=1; j< A.n; j++) 
				cols[j+1] += cols[j];
	
			return new spMatrix(A.m, A.n, val, cols, rows);
		}
	}
}
/**
 * @param{spMatrix}
 * @param{Matrix}
 * @return{spMatrix}
 */
function entrywisemulspMatrixMatrix (A, B) {
	var c = A.copy();	
	const nnz = A.val.length;
	const n = A.n;
	const m = A.m;
	if ( A.rowmajor ) {
		for ( i=0;i< m; i++) {
			var s = c.rows[i];
			var e = c.rows[i+1];
			var r = i*n;
			for ( var k = s; k< e; k++) {
				c.val[k] *= B.val[r + c.cols[k] ];
			}
		}
	}
	else {
		for ( j=0;j< n; j++) {
			var s = c.cols[j];
			var e = c.cols[j+1];
			for ( var k = s; k< e; k++) {
				c.val[k] *= B.val[c.rows[k] * n + j];
			}
		}
	}
	return c;
}

/**
 * @param{number}
 * @param{spVector}
 * @return{Float64Array}
 */
function addScalarspVector (a, b) {
	const nnzb = b.val.length;
	const n = b.length;
	var c = zeros(n);
	var k;
	for ( k=0;k < n; k++) 
		c[k] = a;
	for ( k=0;k < nnzb; k++) 
		c[b.ind[k]] += b.val[k];
			
	return c;
}
/**
 * @param{Float64Array}
 * @param{spVector}
 * @return{Float64Array}
 */
function addVectorspVector (a, b) {
	const nnzb = b.val.length;
	const n = b.length;
	var c = new Float64Array(a);
	for (var k=0;k < nnzb; k++) 
		c[b.ind[k]] += b.val[k];
			
	return c;
}
/**
 * @param{spVector}
 * @param{spVector}
 * @return{spVector}
 */
function addspVectors (a, b) {
	const nnza = a.val.length;
	const nnzb = b.val.length;
	var c = zeros(a.length);
	var k;
	for ( k=0;k < nnza; k++) 
		c[a.ind[k]] = a.val[k];
	for ( k=0;k < nnzb; k++) 
		c[b.ind[k]] += b.val[k];
			
	return sparseVector(c);
}

/**
 * @param{number}
 * @param{spMatrix}
 * @return{Matrix}
 */
function addScalarspMatrix (a, B) {
	const nnzb = B.val.length;
	const m = B.m;
	const n = B.n;
	const mn = m*n;
	
	var C = zeros(m,n); 
	var i;
	for (i = 0; i < mn; i++)
		C.val[i] = a;
	if ( B.rowmajor ) {
		var ri = 0;
		for (i = 0; i < m; i++) {
			var s = B.rows[i];
			var e = B.rows[i+1];
			for (var k= s; k < e; k++)
				C.val[ri + B.cols[k]] += B.val[k];
			ri += n;
		}
	}
	else {
		for (i = 0; i < n; i++) {
			var s = B.cols[i];
			var e = B.cols[i+1];
			for (var k= s; k < e; k++)
				C.val[B.rows[k] * n + i] += B.val[k];
		}
	}
	return C;
}
/**
 * @param{Matrix}
 * @param{spMatrix}
 * @return{Matrix}
 */
function addMatrixspMatrix (A, B) {
	const nnzb = B.val.length;
	const m = B.m;
	const n = B.n;
	const mn = m*n;
	
	var C = matrixCopy(A);
	var i;	
	if ( B.rowmajor ) {
		var ri = 0;
		for (i = 0; i < m; i++) {
			var s = B.rows[i];
			var e = B.rows[i+1];
			for (var k= s; k < e; k++)
				C.val[ri + B.cols[k]] += B.val[k];
			ri += n;
		}
	}
	else {
		for (i = 0; i < n; i++) {
			var s = B.cols[i];
			var e = B.cols[i+1];
			for (var k= s; k < e; k++)
				C.val[B.rows[k] * n + i] += B.val[k];
		}
	}
	return C;
}
/**
 * @param{spMatrix}
 * @param{spMatrix}
 * @return{spMatrix}
 */
function addspMatrices (A, B) {
	const nnza = A.val.length;
	const nnzb = B.val.length;
	const m = A.m;
	const n = A.n;
	
	var C = fullMatrix(A); 
	var i;	
	if ( B.rowmajor ) {
		var ri = 0;
		for (i = 0; i < m; i++) {
			var s = B.rows[i];
			var e = B.rows[i+1];
			for (var k= s; k < e; k++)
				C.val[ri + B.cols[k]] += B.val[k];
			ri += n;
		}
	}
	else {
		for (i = 0; i < n; i++) {
			var s = B.cols[i];
			var e = B.cols[i+1];
			for (var k= s; k < e; k++)
				C.val[B.rows[k] * n + i] += B.val[k];
		}
	}
	return sparseMatrixRowMajor(C);
}

/** sparse SAXPY : y = y + ax with x sparse and y dense
 * @param {number}
 * @param {spVector}
 * @param {Float64Array}
 */
function spsaxpy ( a, x, y) {
	const nnz = x.val.length;	
	for (var k=0;k < nnz; k++) 
		y[x.ind[k]] += a * x.val[k];			
}

/**
 * @param{number}
 * @param{spVector}
 * @return{Float64Array}
 */
function subScalarspVector (a, b) {
	const nnzb = b.val.length;
	const n = b.length;
	var c = zeros(n);
	var k;
	for ( k=0;k < n; k++) 
		c[k] = a;
	for ( k=0;k < nnzb; k++) 
		c[b.ind[k]] -= b.val[k];
			
	return c;
}
/**
 * @param{Float64Array}
 * @param{spVector}
 * @return{Float64Array}
 */
function subVectorspVector (a, b) {
	const nnzb = b.val.length;
	const n = b.length;
	var c = new Float64Array(a);
	for (var k=0;k < nnzb; k++) 
		c[b.ind[k]] -= b.val[k];
			
	return c;
}
/**
 * @param{spVector}
 * @param{Float64Array}
 * @return{Float64Array}
 */
function subspVectorVector (a, b) {
	return subVectors(fullVector(a), b);
}
/**
 * @param{spVector}
 * @param{spVector}
 * @return{spVector}
 */
function subspVectors (a, b) {
	const nnza = a.val.length;
	const nnzb = b.val.length;
	var c = zeros(a.length);
	var k;
	for ( k=0;k < nnza; k++) 
		c[a.ind[k]] = a.val[k];
	for ( k=0;k < nnzb; k++) 
		c[b.ind[k]] -= b.val[k];
			
	return sparseVector(c);
}

/**
 * @param{number}
 * @param{spMatrix}
 * @return{Matrix}
 */
function subScalarspMatrix (a, B) {
	const nnzb = B.val.length;
	const m = B.m;
	const n = B.n;
	const mn = m*n;
	
	var C = zeros(m,n); 
	var i;
	for (i = 0; i < mn; i++)
		C.val[i] = a;
	if ( B.rowmajor ) {
		var ri = 0;
		for (i = 0; i < m; i++) {
			var s = B.rows[i];
			var e = B.rows[i+1];
			for (var k= s; k < e; k++)
				C.val[ri + B.cols[k]] -= B.val[k];
			ri += n;
		}
	}
	else {
		for (i = 0; i < n; i++) {
			var s = B.cols[i];
			var e = B.cols[i+1];
			for (var k= s; k < e; k++)
				C.val[B.rows[k] * n + i] -= B.val[k];
		}
	}
	return C;
}
/**
 * @param{spMatrix}
 * @param{Matrix}
 * @return{Matrix}
 */
function subspMatrixMatrix (A, B) {
	return subMatrices(fullMatrix(A), B);
}
/**
 * @param{Matrix}
 * @param{spMatrix}
 * @return{Matrix}
 */
function subMatrixspMatrix (A, B) {
	const nnzb = B.val.length;
	const m = B.m;
	const n = B.n;
	const mn = m*n;
	
	var C = matrixCopy(A);
	var i;	
	if ( B.rowmajor ) {
		var ri = 0;
		for (i = 0; i < m; i++) {
			var s = B.rows[i];
			var e = B.rows[i+1];
			for (var k= s; k < e; k++)
				C.val[ri + B.cols[k]] -= B.val[k];
			ri += n;
		}
	}
	else {
		for (i = 0; i < n; i++) {
			var s = B.cols[i];
			var e = B.cols[i+1];
			for (var k= s; k < e; k++)
				C.val[B.rows[k] * n + i] -= B.val[k];
		}
	}
	return C;
}
/**
 * @param{spMatrix}
 * @param{spMatrix}
 * @return{spMatrix}
 */
function subspMatrices (A, B) {
	const nnza = A.val.length;
	const nnzb = B.val.length;
	const m = A.m;
	const n = A.n;
	
	var C = fullMatrix(A); 
	var i;	
	if ( B.rowmajor ) {
		var ri = 0;
		for (i = 0; i < m; i++) {
			var s = B.rows[i];
			var e = B.rows[i+1];
			for (var k= s; k < e; k++)
				C.val[ri + B.cols[k]] -= B.val[k];
			ri += n;
		}
	}
	else {
		for (i = 0; i < n; i++) {
			var s = B.cols[i];
			var e = B.cols[i+1];
			for (var k= s; k < e; k++)
				C.val[B.rows[k] * n + i] -= B.val[k];
		}
	}
	return sparseMatrixRowMajor(C);
}

/**
 * @param{function}
 * @param{spVector}
 * @return{Float64Array}
 */
function applyspVector( f, x ) {
	const nnz = x.val.length;
	const n = x.length;
	var res = new Float64Array(n);
	var i;
	const f0 = f(0);
	for ( i=0; i< n; i++) 
		res[i] = f0;
	for ( i=0; i< nnz; i++) 
		res[x.ind[i]] = f(x.val[i]);
	return res;
}
/**
 * @param{function}
 * @param{spMatrix}
 * @return{Matrix}
 */
function applyspMatrix( f, X ) {
	const nnz = X.val.length;
	const m = X.m;
	const n = X.n;
	const mn = m*n;
	const f0 = f(0);
	var C = zeros(m,n); 
	var i;
	if ( !isZero(f0) ) {
		for (i = 0; i < mn; i++)
			C.val[i] = f0;
	}
	if ( X.rowmajor ) {
		var ri = 0;
		for (i = 0; i < m; i++) {
			var s = X.rows[i];
			var e = X.rows[i+1];
			for (var k= s; k < e; k++)
				C.val[ri + X.cols[k]] = f(X.val[k]);
			ri += n;
		}
	}
	else {
		for (i = 0; i < n; i++) {
			var s = X.cols[i];
			var e = X.cols[i+1];
			for (var k= s; k < e; k++)
				C.val[X.rows[k] * n + i] += f(X.val[k]);
		}
	}
	return C;
}
/**
 * @param{spVector}
 * @return{number}
 */
function sumspVector( a ) {
	return sumVector(a.val);
}
/**
 * @param{spMatrix}
 * @return{number}
 */
function sumspMatrix( A ) {
	return sumVector(A.val);
}
/**
 * @param{spMatrix}
 * @return{Matrix}
 */
function sumspMatrixRows( A ) {
	var res = zeros(A.n);
	if ( A.rowmajor ) {
		for ( var k=0; k < A.val.length; k++)
			res[A.cols[k]] += A.val[k];
	}
	else {
		for ( var i=0; i<A.n; i++)
			res[i] = sumspVector(A.col(i));
	}
	return new Matrix(1,A.n, res, true);
}
/**
 * @param{spMatrix}
 * @return{Float64Array}
 */
function sumspMatrixCols( A ) {	
	var res = zeros(A.m);
	if ( A.rowmajor ) {
		for ( var i=0; i<A.m; i++)
			res[i] = sumspVector(A.row(i));			
	}
	else {
		for ( var k=0; k < A.val.length; k++)
			res[A.rows[k]] += A.val[k];
	}
	return res;
}
/**
 * @param{spMatrix}
 * @return{Matrix}
 */
function prodspMatrixRows( A ) {
	if ( A.rowmajor ) {
		var res = ones(A.n);	
		for ( var i=0; i < A.m; i++) {
			var s = A.rows[i];
			var e = A.rows[i+1];
			for ( var j=0; j < A.n; j++) 
				if ( A.cols.subarray(s,e).indexOf(j) < 0 )
					res[j] = 0;
			for ( var k=s; k < e; k++)
				res[A.cols[k]] *= A.val[k];
		}
	}
	else {
		var res = zeros(A.n);
		for ( var i=0; i<A.n; i++) {
			var a = A.col(i);
			if ( a.val.length == a.length )
				res[i] = prodVector(a.val);
		}
	}
	return new Matrix(1,A.n, res, true);
}
/**
 * @param{spMatrix}
 * @return{Float64Array}
 */
function prodspMatrixCols( A ) {	
	if ( A.rowmajor ) {
		var res = zeros(A.m);
		for ( var i=0; i<A.m; i++) {
			var a = A.row(i);
			if ( a.val.length == a.length )
				res[i] = prodVector(a.val);
		}
	}
	else {
		var res = ones(A.m);	
		for ( var j=0; j < A.n; j++) {
			var s = A.cols[j];
			var e = A.cols[j+1];
			for ( var i=0; i < A.m; i++) 
				if ( A.rows.subarray(s,e).indexOf(i) < 0 )
					res[i] = 0;
			for ( var k=s; k < e; k++)
				res[A.rows[k]] *= A.val[k];
		}
	}
	return res;
}


///////////////////////////
/// Sparse linear systems 
///////////////////////////
/** Sparse Conjugate gradient method for solving the symmetric positie definite system Ax = b
 * @param{spMatrix}
 * @param{Float64Array}
 * @return{Float64Array}
 */
function spsolvecg ( A, b) {

	const n = A.n;	
	const m = A.m;

	var x = randn(n); 
	var r = subVectors(b, mulspMatrixVector(A, x));
	var rhoc = dot(r,r);
	const TOL = 1e-8;
	var delta2 = TOL * norm(b);
	delta2 *= delta2;
	
	// first iteration:
	var p = vectorCopy(r);
	var w = mulspMatrixVector(A,p);
	var mu = rhoc / dot(p, w);
	saxpy( mu, p, x);
	saxpy( -mu, w, r);
	var rho_ = rhoc;
	rhoc = dot(r,r);

	var k = 1;

	var updateP = function (tau, r) {
		for ( var i=0; i < m; i++)
			p[i] = r[i] + tau * p[i];
	}
	
	while ( rhoc > delta2 && k < n ) {
		updateP(rhoc/rho_, r);
		w = mulspMatrixVector(A,p);
		mu = rhoc / dot(p, w);
		saxpy( mu, p, x);
		saxpy( -mu, w, r);
		rho_ = rhoc;
		rhoc = dot(r,r);
		k++;
	}
	return x;
}
/** Sparse Conjugate gradient normal equation residual method for solving the rectangular system Ax = b
 * @param{spMatrix}
 * @param{Float64Array}
 * @return{Float64Array}
 */
function spcgnr ( A, b) {
/*
TEST
A = randnsparse(0.3,10000,1000)
x = randn(1000)
b = A*x + 0.01*randn(10000)
tic()
xx = cgnr(A,b)
t1 = toc()
ee = norm(A*xx - b)
tic()
xh=spcgnr(sparse(A), b)
t2 = toc()
e = norm(A*xh - b)
*/
	
	const n = A.n;	
	const m = A.m;

	var x = randn(n); 
	var r = subVectors(b, mulspMatrixVector(A, x));	
	const TOL = 1e-8;
	var delta2 = TOL * norm(b);
	delta2 *= delta2;
	
	// first iteration:
	var z = mulspMatrixTransVector(A, r);
	var rhoc = dot(z,z);	
	var p = vectorCopy(z);
	var w = mulspMatrixVector(A,p);
	var mu = rhoc / dot(w, w);
	saxpy( mu, p, x);
	saxpy( -mu, w, r);	
	z = mulspMatrixTransVector(A, r);
	var rho_ = rhoc;
	rhoc = dot(z,z);

	var k = 1;

	var updateP = function (tau, z) {
		for ( var i=0; i < m; i++)
			p[i] = z[i] + tau * p[i];
	}
	
	while ( rhoc > delta2 && k < n ) {
		updateP(rhoc/rho_, z);
		w = mulspMatrixVector(A,p);
		mu = rhoc / dot(w, w);
		saxpy( mu, p, x);
		saxpy( -mu, w, r);
		z = mulspMatrixTransVector(A, r);
		rho_ = rhoc;
		rhoc = dot(z,z);
		k++;
	}
	return x;
}


