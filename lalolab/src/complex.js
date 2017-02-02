const Complex_I = new Complex(0, 1);

/**
 * @constructor
 * @struct
 */
function Complex(a, b, polar) {
	/** @const */ this.type = "Complex";
	
	if ( typeof(a) == "undefined") {
		this.re = 0.0;
		this.im = 0.0;
	}
	else if ( a instanceof Complex ) {
		this.re = a.re;
		this.im = a.im;
	}
	else if ( typeof(a) == "number" && !polar ) {
		this.re = a;
		this.im = b;
	}
	else {
		this.re = a * Math.cos(b);
		this.im = a * Math.sin(b);
	}
}

Complex.prototype.toString = function () {
	return this.re + (this.im >= 0 ? " + " : " - ") + Math.abs(this.im) + "i";
}
Complex.prototype.info = function () {
	return this.re + (this.im >= 0 ? " + " : " - ") + Math.abs(this.im) + "i";
}
/**
 * @param {Complex}
 * @param {Complex}
 * @return {Complex} 
 */
function addComplex(a,b) {
	var z = new Complex(a);
	z.re += b.re; 
	z.im += b.im;
	return z;
}
/**
 * @param {Complex}
 * @param {number}
 * @return {Complex} 
 */
function addComplexReal(a,b) {
	var z = new Complex(a);
	z.re += b;
	return z;
}
/**
 * @param {Complex}
 * @param {Complex}
 * @return {Complex} 
 */
function subComplex(a,b) {
	var z = new Complex(a);
	z.re -= b.re;
	z.im -= b.im;
	return z;
}
/**
 * @param {Complex}
 * @return {Complex} 
 */
function minusComplex(a) {
	return new Complex(-a.re, -a.im);
}

function mulComplex(a,b) {
	return new Complex(a.re*b.re - a.im*b.im, a.im * b.re + a.re*b.im);
}
function mulComplexReal(a,b) {
	return new Complex(a.re*b, a.im * b);
}
function divComplex(a,b) {
	var denom = b.re*b.re + b.im*b.im;
	return new Complex( (a.re*b.re + a.im*b.im) / denom, (a.im * b.re - a.re*b.im) / denom );
}

function conj(z) {
	if (z instanceof Complex)
		return new Complex(z.re, -z.im);
	else if (z instanceof ComplexVector) {
		var r = new ComplexVector(z);
		for (var i=0; i < z.length; i++)
			r.im[i] = -r.im[i];
		return r;
	}	
	else if (z instanceof ComplexMatrix) {
		var r = new ComplexMatrix(z);
		for (var i=0; i < z.length; i++)
			r.im[i] = -r.im[i];
		return r;
	}		
	else 
		return new Complex(z);	// for a real
}

function modulus(z) {
	if ( z instanceof Complex ) 
		return Math.sqrt(z.re*z.re + z.im*z.im);
	else if (z instanceof ComplexVector)
		return sqrt(addVectors( entrywisemulVector(z.re, z.re), entrywisemulVector(z.im, z.im) )); 
	else if (z instanceof ComplexVector)
		return new Matrix(z.m, z.n, sqrt(addVectors( entrywisemulVector(z.re, z.re), entrywisemulVector(z.im, z.im) ) , true)); 
}
var absComplex = modulus;

function expComplex(z) {
	return new Complex(Math.exp(z.re), z.im, true); 	
}



/**
 * @constructor
 * @struct
 */
function ComplexVector(a, b, dontcopy) {
	/** @const */ this.type = "ComplexVector";
	
	
	if ( a instanceof ComplexVector) {
		/** @const */ this.length = a.length;
		this.re = vectorCopy(a.re);
		this.im = vectorCopy(a.im);
	}
	else if (typeof(a) == "number") {
		/** @const */ this.length = a;
		this.re = new Float64Array(a);
		this.im = new Float64Array(a);		
	}
	else if ( a instanceof Float64Array && b instanceof Float64Array ) {
		/** @const */ this.length = a.length;
		if ( typeof(dontcopy) == "undefined" || !dontcopy ){
			this.re = vectorCopy(a);
			this.im = vectorCopy(b);
		}
		else {
			this.re = a;
			this.im = b;
		}
	}
	else {
		error("Bad arguments to new ComplexVector()");
	}
}

/**
 * @constructor
 * @struct
 */
function ComplexMatrix(a, b, values, valuesimag) {
	/** @const */ this.type = "ComplexMatrix";
		
	if ( a instanceof ComplexMatrix) {
		/** @const */ this.length = a.length;
		/** @const */ this.m = a.m;
		/** @const */ this.n = a.n;
		/** @const */ this.size = [a.m, a.n]; 
		this.re = vectorCopy(a.re);
		this.im = vectorCopy(a.im);
	}
	else if (typeof(a) == "number" && typeof(b) == "number") {
		/** @const */ this.length = a;
		/** @const */ this.m = a;
		/** @const */ this.n = b;
		/** @const */ this.size = [a, b]; 
		if ( typeof(values) == "undefined") {
			this.re = new Float64Array(a*b);
			this.im = new Float64Array(a*b);
		}
		else if ( values instanceof ComplexVector ) {
			this.re = vectorCopy(values.re);
			this.im = vectorCopy(values.im);
		}
		else if ( values instanceof Float64Array && typeof(valuesimag) != "undefined" &&  valuesimag instanceof Float64Array) {
			this.re = values;
			this.im = valuesimag;	// !! no copy!
		}
	}
	else if ( a instanceof Matrix && b instanceof Matrix) {
		/** @const */ this.length = a.length;
		/** @const */ this.m = a.m;
		/** @const */ this.n = a.n;
		/** @const */ this.size = [a.m, a.n]; 
		this.re = vectorCopy(a.val);
		this.im = vectorCopy(b.val);
	}
	else 
		error("Bad arguments to new ComplexMatrix()");
}


ComplexVector.prototype.toString = function () {
	return "[" + this.type + " of size " + this.length + "]";
}
ComplexMatrix.prototype.toString = function () {
	return "[" + this.type + " of size " + this.m + " x " + this.n + "]";
}
ComplexVector.prototype.get = function (i) {
	return new Complex(this.re[i], this.im[i]);
}
ComplexMatrix.prototype.get = function (i,j) {
	return new Complex(this.re[i*this.n + j], this.im[i*this.n + j]);
}
ComplexVector.prototype.set = function (i, z) {
	if ( typeof(z) == "number" )  {
		this.re[i] = z;
		this.im[i] =	0;
	}
	else {
		this.re[i] = z.re;
		this.im[i] = z.im;
	}
}
ComplexMatrix.prototype.set = function (i, j, z) {
	if ( typeof(z) == "number" )  {
		this.re[i*this.n + j] = z;
		this.im[i*this.n + j] =	0;
	}
	else {
		this.re[i*this.n + j] = z.re;
		this.im[i*this.n + j] = z.im;
	}
}
ComplexVector.prototype.getSubVector = function (rowsrange) {
	const n = rowsrange.length;
	var res = new ComplexVector( n );
	for (var i = 0; i< n; i++) {
		res.re[i] = this.re[rowsrange[i]];
		res.im[i] = this.im[rowsrange[i]];
	}
	return res;
}
ComplexVector.prototype.setVectorScalar = function (rowsrange, B) {
	var i;
	for (i = 0; i< rowsrange.length; i++) 
		A.set ( rowsrange[i], B);
}
ComplexVector.prototype.setVectorVector = function (rowsrange, B) {
	var i;
	for (i = 0; i< rowsrange.length; i++) 
		A.set(rowsrange[i], B[i]);
}



function real(z) {
	if (z instanceof Complex) 
		return z.re;
	else if (z instanceof ComplexVector)
		return vectorCopy(z.re);
	else if (z instanceof ComplexMatrix)
		return new Matrix(z.m, z.n, z.re);
	else
		return copy(z);		
}
function imag(z) {
	if (z instanceof Complex) 
		return z.im;
	else if (z instanceof ComplexVector)
		return vectorCopy(z.im);
	else if (z instanceof ComplexMatrix)
		return new Matrix(z.m, z.n, z.im);
	else
		return 0;		
}

/**
 * @param {MatrixComplex} 
 */
function transposeComplexMatrix ( A ) {
	// Hermitian transpose = conjugate transpose
	const m = A.m;
	const n = A.n;
	if ( m > 1 ) {
		var i;
		var j;
		var res = new ComplexMatrix( n,m);
		var Aj = 0;
		for ( j=0; j< m;j++) {
			var ri = 0;
			for ( i=0; i < n ; i++) {
				res.re[ri + j] = A.re[Aj + i];
				res.im[ri + j] = -A.im[Aj + i];
				ri += m;
			}
			Aj += n;
		}
		return res;
	}
	else {
		return new ComplexVector(A.re,minusVector(A.im));
	}
}
/**
 * @param {MatrixComplex} 
 */
ComplexMatrix.prototype.transpose = function ( ) {
	// simple Transpose without conjugate
	const m = A.m;
	const n = A.n;
	if ( m > 1 ) {
		var i;
		var j;
		var res = new ComplexMatrix( n,m);
		var Aj = 0;
		for ( j=0; j< m;j++) {
			var ri = 0;
			for ( i=0; i < n ; i++) {
				res.re[ri + j] = A.re[Aj + i];
				res.im[ri + j] = A.im[Aj + i];
				ri += m;
			}
			Aj += n;
		}
		return res;
	}
	else {
		return new ComplexVector(A.re,A.im);
	}
}


/**
 * @param {ComplexVector}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function addComplexVectors(a, b) {
	var z = new ComplexVector(a);
	const n = a.length;
	for ( var i=0; i< n; i++) {
		z.re[i] += b.re[i];
		z.im[i] += b.im[i];
	}
	return z;
}
/**
 * @param {ComplexVector}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function subComplexVectors(a, b) {
	var z = new ComplexVector(a);
	const n = a.length;
	for ( var i=0; i< n; i++) {
		z.re[i] -= b.re[i];
		z.im[i] -= b.im[i];
	}
	return z;
}

/**
 * @param {ComplexMatrix}
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function addComplexMatrices(a, b) {
	var z = new ComplexMatrix(a);
	const mn = a.m * a.n;
	for ( var i=0; i< mn; i++) {
		z.re[i] += b.re[i];
		z.im[i] += b.im[i];
	}
	return z;
}

/**
 * @param {ComplexMatrix}
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function subComplexMatrices(a, b) {
	var z = new ComplexMatrix(a);
	const mn = a.m * a.n;
	for ( var i=0; i< mn; i++) {
		z.re[i] -= b.re[i];
		z.im[i] -= b.im[i];
	}
	return z;
}
/**
 * @param {ComplexVector}
 * @param {Float64Array}
 * @return {ComplexVector} 
 */
function addComplexVectorVector(a, b) {
	var z = new ComplexVector(a);
	const n = a.length;
	for ( var i=0; i< n; i++) {
		z.re[i] += b[i];
	}
	return z;
}
/**
 * @param {ComplexVector}
 * @param {Float64Array}
 * @return {ComplexVector} 
 */
function subComplexVectorVector(a, b) {
	var z = new ComplexVector(a);
	const n = a.length;
	for ( var i=0; i< n; i++) {
		z.re[i] -= b[i];		
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @param {Matrix}
 * @return {ComplexMatrix} 
 */
function addComplexMatrixMatrix(a, b) {
	var z = new ComplexMatrix(a);
	const n = a.m * a.n;
	for ( var i=0; i< n; i++) {
		z.re[i] += b.val[i];
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @param {Matrix}
 * @return {ComplexMatrix} 
 */
function subComplexMatrixMatrix(a, b) {
	var z = new ComplexMatrix(a);
	const n = a.m * a.n;
	for ( var i=0; i< n; i++) {
		z.re[i] -= b.val[i];
	}
	return z;
}

/**
 * @param {number}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function addScalarComplexVector(a, b) {
	var z = new ComplexVector(b);
	const n = b.length;
	for ( var i=0; i< n; i++) {
		z.re[i] += a;
	}
	return z;
}
/**
 * @param {number}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function subScalarComplexVector(a, b) {
	var z = minusComplexVector(b);
	const n = b.length;
	for ( var i=0; i< n; i++) {
		z.re[i] += a;
	}
	return z;
}

/**
 * @param {number}
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function addScalarComplexMatrix(a, b) {
	var z = new ComplexMatrix(b);
	const n = b.m * b.n;
	for ( var i=0; i< n; i++) {
		z.re[i] += a;
	}
	return z;
}



/**
 * @param {ComplexVector}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function entrywisemulComplexVectors(a, b) {
	const n = a.length;
	var z = new ComplexVector(n);
	for ( var i=0; i< n; i++) {
		z.re[i] = a.re[i] * b.re[i] - a.im[i] * b.im[i];
		z.im[i] = a.im[i] * b.re[i] + a.re[i] * b.im[i];
	}
	return z;
}
/**
 * @param {ComplexVector}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function entrywisedivComplexVectors(a, b) {
	const n = a.length;
	var z = new ComplexVector(n);
	for ( var i=0; i< n; i++) {
		var bre = b.re[i];
		var bim = b.im[i]; 
		var denom = bre*bre + bim*bim;
		z.re[i] = (a.re[i]*bre + a.im[i]*bim) / denom;
		z.im[i] = (a.im[i]*bre - a.re[i]*bim) / denom;		
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function entrywisemulComplexMatrices(a, b) {
	const n = a.m * a.n;
	var z = new ComplexMatrix(a.m, a.n);
	for ( var i=0; i< n; i++) {
		z.re[i] = a.re[i] * b.re[i] - a.im[i] * b.im[i];
		z.im[i] = a.im[i] * b.re[i] + a.re[i] * b.im[i];
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function entrywisedivComplexMatrices(a, b) {
	const n = a.m * a.n;
	var z = new ComplexMatrix(a.m, a.n);
	for ( var i=0; i< n; i++) {
		var bre = b.re[i];
		var bim = b.im[i]; 
		var denom = bre*bre + bim*bim;
		z.re[i] = (a.re[i]*bre + a.im[i]*bim) / denom;
		z.im[i] = (a.im[i]*bre - a.re[i]*bim) / denom;		
	}
	return z;
}

/**
 * @param {ComplexVector}
 * @param {Float64Array}
 * @return {ComplexVector} 
 */
function entrywisemulComplexVectorVector(a, b) {
	const n = a.length;
	var z = new ComplexVector(n);
	for ( var i=0; i< n; i++) {
		z.re[i] = a.re[i] * b[i];
		z.im[i] = a.im[i] * b[i];
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @param {Matrix}
 * @return {ComplexMatrix} 
 */
function entrywisemulComplexMatrixMatrix(a, b) {
	const n = a.m * a.n;
	var z = new ComplexMatrix(a.m, a.n);
	for ( var i=0; i< n; i++) {
		z.re[i] = a.re[i] * b.val[i];
		z.im[i] = a.im[i] * b.val[i];
	}
	return z;
}

/**
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function minusComplexVector(a) {
	const n = a.length;
	var z = new ComplexVector(n);
	for ( var i=0; i< n; i++) {
		z.re[i] = -a.re[i];
		z.im[i] = -a.im[i];
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function minusComplexMatrix(a) {
	var z = new ComplexMatrix(a.m, a.n);
	const n = a.m * a.n;
	for ( var i=0; i< n; i++) {
		z.re[i] = -a.re[i];
		z.im[i] = -a.im[i];
	}
	return z;
}
/**
 * @param {ComplexVector}
 * @return {number} 
 */
function sumComplexVector(a) {
	var z = new Complex();
	const n = a.length;
	for ( var i=0; i< n; i++) {
		z.re += a.re[i];
		z.im += a.im[i];
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @return {number} 
 */
function sumComplexMatrix(a) {
	var z = new Complex();
	const n = a.m * a.n;
	for ( var i=0; i< n; i++) {
		z.re += a.re[i];
		z.im += a.im[i];
	}
	return z;
}
/**
 * @param {ComplexVector}
 * @return {number} 
 */
function norm1ComplexVector(a) {
	var r = 0.0;
	const n = a.length;
	for ( var i=0; i< n; i++) {
		r += Math.sqrt(a.re[i] * a.re[i] + a.im[i]*a.im[i]);		
	}
	return r;
}
/**
 * @param {ComplexVector}
 * @return {number} 
 */
function norm2ComplexVector(a) {
	var r = 0.0;
	const n = a.length;
	for ( var i=0; i< n; i++) {
		r += a.re[i] * a.re[i] + a.im[i]*a.im[i];
	}
	return Math.sqrt(r);
}
/**
 * @param {ComplexMatrix}
 * @return {number} 
 */
function normFroComplexMatrix(a) {
	var r = 0.0;
	const n = a.m * a.n;
	for ( var i=0; i< n; i++) {
		r += a.re[i] * a.re[i] + a.im[i]*a.im[i];
	}
	return Math.sqrt(r);
}
/**
 * @param {ComplexVector}
 * @param {ComplexVector}
 * @return {Complex} 
 */
function dotComplexVectors(a, b) {
	// = b^H a = conj(b)^T a
	var z = new Complex(); 
	const n = a.length;
	for ( var i=0; i< n; i++) {
		z.re += a.re[i] * b.re[i] + a.im[i] * b.im[i];
		z.im += a.im[i] * b.re[i] - a.re[i] * b.im[i]
	}
	return z;
}
/**
 * @param {ComplexVector}
 * @param {Float64Array}
 * @return {Complex} 
 */
function dotComplexVectorVector(a, b) {
	// = b^T a
	var z = new Complex(); 
	const n = a.length;
	for ( var i=0; i< n; i++) {
		z.re += a.re[i] * b[i];
		z.im += a.im[i] * b[i];
	}
	return z;
}
/**
 * @param {number}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function mulScalarComplexVector(a, b) {
	var re = mulScalarVector(a, b.re);
	var im = mulScalarVector(a, b.im);
	return new ComplexVector(re,im, true);
}
/**
 * @param {Complex}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function mulComplexComplexVector(a, b) {
	const n = b.length;
	var z = new ComplexVector(n);
	var are = a.re;
	var aim = a.im; 
	for ( var i=0; i< n; i++) {
		z.re[i] = are * b.re[i] - aim * b.im[i];
		z.im[i] = aim * b.re[i] + are * b.im[i];
	}
	return z;
}
/**
 * @param {Complex}
 * @param {Float64Array}
 * @return {ComplexVector} 
 */
function mulComplexVector(a, b) {
	const n = b.length;
	var z = new ComplexVector(n);
	var are = a.re;
	var aim = a.im; 
	for ( var i=0; i< n; i++) {
		z.re[i] = are * b[i];
		z.im[i] = aim * b[i];
	}
	return z;
}
/**
 * @param {number}
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function mulScalarComplexMatrix(a, b) {
	var re = mulScalarVector(a, b.re);
	var im = mulScalarVector(a, b.im);
	return new ComplexMatrix(b.m, b.n, re, im);
}
/**
 * @param {Complex}
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function mulComplexComplexMatrix(a, b) {
	const n = b.m*b.n;
	var z = new ComplexMatrix(b.m,b.n);
	var are = a.re;
	var aim = a.im; 
	for ( var i=0; i< n; i++) {
		z.re[i] = are * b.re[i] - aim * b.im[i];
		z.im[i] = aim * b.re[i] + are * b.im[i];
	}
	return z;
}
/**
 * @param {Complex}
 * @param {Matrix}
 * @return {ComplexMatrix} 
 */
function mulComplexMatrix(a, b) {
	const n = b.m * b.n;
	var z = new ComplexMatrix(b.m, b.n);
	var are = a.re;
	var aim = a.im; 
	for ( var i=0; i< n; i++) {
		z.re[i] = are * b.val[i];
		z.im[i] = aim * b.val[i];
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @param {Float64Array}
 * @return {ComplexVector} 
 */
function mulComplexMatrixVector(a, b) {
	const m = a.m;
	const n = a.n;
	var z = new ComplexVector(m); 
	var ai = 0;
	for ( var i=0; i< m; i++) {
		for ( j=0; j < n ; j++) {
			z.re[i] += a.re[ai+j] * b[j];
			z.im[i] += a.im[ai+j] * b[j];
		}
		ai += n;
	}
	return z;
}
/**
 * @param {ComplexMatrix}
 * @param {ComplexVector}
 * @return {ComplexVector} 
 */
function mulComplexMatrixComplexVector(a, b) {
	const m = a.m;
	const n = a.n;
	var z = new ComplexVector(m); 
	var ai = 0;
	for ( var i=0; i< m; i++) {
		for ( j=0; j < n ; j++) {
			z.re[i] += a.re[ai+j] * b.re[j] - a.im[ai+j] * b.im[j];
			z.im[i] += a.im[ai+j] * b.re[j] + a.re[ai+j] * b.im[j];
		}
		ai += n;
	}
	return z;
}

/**
 * @param {ComplexMatrix}
 * @param {ComplexMatrix}
 * @return {ComplexMatrix} 
 */
function mulComplexMatrices(A, B) {
	const m = A.length;
	const n = B.n;
	const n2 = B.length;
	
	var Are = A.re; 
	var Aim = A.im;
	var Bre = B.re;
	var Bim = B.im;
	
	var Cre = new Float64Array(m*n);
	var Cim = new Float64Array(m*n);	
	var aik;
	var Aik = 0;
	var Ci = 0;
	for (var i=0;i < m ; i++) {		
		var bj = 0;
		for (var k=0; k < n2; k++ ) {
			aikre = Are[Aik];
			aikim = Aim[Aik];
			for (var j =0; j < n; j++) {
				Cre[Ci + j] += aikre * Bre[bj] - aikim * Bim[bj];
				Cim[Ci + j] += aikre * Bim[bj] + aikim * Bre[bj];
				bj++;
			}	
			Aik++;					
		}
		Ci += n;
	}
	return  new ComplexMatrix(m,n,Cre, Cim);
}
/**
 * @param {ComplexMatrix}
 * @param {Matrix}
 * @return {ComplexMatrix} 
 */
function mulComplexMatrixMatrix(A, B) {
	const m = A.m;
	const n = B.n;
	const n2 = B.m;
	
	var Are = A.re; 
	var Aim = A.im;
	var Bre = B.val;
	
	var Cre = new Float64Array(m*n);
	var Cim = new Float64Array(m*n);	
	var aik;
	var Aik = 0;
	var Ci = 0;
	for (var i=0;i < m ; i++) {		
		var bj = 0;
		for (var k=0; k < n2; k++ ) {
			aikre = Are[Aik];
			aikim = Aim[Aik];
			for (var j =0; j < n; j++) {
				Cre[Ci + j] += aikre * Bre[bj];
				Cim[Ci + j] += aikim * Bre[bj];
				bj++;
			}	
			Aik++;					
		}
		Ci += n;
	}
	return  new ComplexMatrix(m,n,Cre, Cim);
}



/**
 * @param {Float64Array|ComplexVector}
 * @return {ComplexVector} 
 */
function fft(x) {
	const n = x.length;
	const s = Math.log2(n);
	const m = n/2;
	
	if ( s % 1 != 0 ) {
		error("fft(x) only implemented for x.length = 2^m. Use dft(x) instead.");
		return undefined;
	}

	var X = new ComplexVector(x,zeros(n));
		
	// bit reversal:	
	var j = 0;
	for (var i = 0; i < n-1 ; i++) {
		if (i < j) {
			// swap(X[i], X[j])
			var Xi = X.re[i];
			X.re[i] = X.re[j];
			X.re[j] = Xi;
			Xi = X.im[i];
			X.im[i] = X.im[j];
			X.im[j] = Xi;
		}
		
		var k = m;
		while (k <= j) {
			j -= k;
			k /= 2;
		}
		j += k;
	}
	
	// FFT:
	var l2 = 1;
	var c = new Complex(-1,0);
	var u = new Complex();
	for (var l = 0; l < s; l++) {
		var l1 = l2;
		l2 *= 2;
		u.re = 1;
		u.im = 0;
      	for (var j = 0; j < l1; j++) {
        	for (var i = j; i < n; i += l2) {
		        var i1 = i + l1;
		        //var t1 = mulComplex(u, X.get(i1) );
		        var t1re = u.re * X.re[i1] - u.im * X.im[i1]; // t1 = u * X[i1]
		        var t1im = u.im * X.re[i1] + u.re * X.im[i1];

		        X.re[i1] = X.re[i] - t1re;
		        X.im[i1] = X.im[i] - t1im;
		        	        
		        X.re[i] += t1re;
		        X.im[i] += t1im;
	        }

			u = mulComplex(u, c);
		}

		c.im = -Math.sqrt((1.0 - c.re) / 2.0);
		c.re = Math.sqrt((1.0 + c.re) / 2.0);
	}
	return X;
}
/**
 * @param {ComplexVector}
 * @return {ComplexVector|Float64Array} 
 */
function ifft(x) {
	const n = x.length;
	const s = Math.log2(n);
	const m = n/2;
	
	if ( s % 1 != 0 ) {
		error("ifft(x) only implemented for x.length = 2^m. Use idft(x) instead.");
		return undefined;
	}
	
	
	var X = new ComplexVector(x,zeros(n));
		
	// bit reversal:	
	var j = 0;
	for (var i = 0; i < n-1 ; i++) {
		if (i < j) {
			// swap(X[i], X[j])
			var Xi = X.re[i];
			X.re[i] = X.re[j];
			X.re[j] = Xi;
			Xi = X.im[i];
			X.im[i] = X.im[j];
			X.im[j] = Xi;
		}
		
		var k = m;
		while (k <= j) {
			j -= k;
			k /= 2;
		}
		j += k;
	}
	
	// iFFT:
	var l2 = 1;
	var c = new Complex(-1,0);
	var u = new Complex();
	for (var l = 0; l < s; l++) {
		var l1 = l2;
		l2 *= 2;
		u.re = 1;
		u.im = 0;
      	for (var j = 0; j < l1; j++) {
        	for (var i = j; i < n; i += l2) {
		        var i1 = i + l1;
		        //var t1 = mulComplex(u, X.get(i1) );
		        var t1re = u.re * X.re[i1] - u.im * X.im[i1]; // t1 = u * X[i1]
		        var t1im = u.im * X.re[i1] + u.re * X.im[i1];

		        X.re[i1] = X.re[i] - t1re;
		        X.im[i1] = X.im[i] - t1im;
		        	        
		        X.re[i] += t1re;
		        X.im[i] += t1im;
	        }

			u = mulComplex(u, c);
		}

		c.im = Math.sqrt((1.0 - c.re) / 2.0);		
		c.re = Math.sqrt((1.0 + c.re) / 2.0);
	}
	var isComplex = false;
	for(var i=0; i < n; i++) {
		X.re[i] /= n;
		X.im[i] /= n;
		if ( Math.abs(X.im[i]) > 1e-6 )
			isComplex = true;
	}
	if (isComplex)
		return X;
	else
		return X.re;
}

function dft(x) {
	// DFT of a real signal
	if ( typeof(x) == "number")
		return new Complex(x, 0); 
	
	const n = x.length;
	if ( n == 1) 
		return new Complex(x[0], 0); 
	else if ( Math.log2(n) % 1 == 0 )
		return fft(x);
	else {
		var X = new ComplexVector(n);
		var thet = 0.0;
		for ( var i=0; i < n; i++) {
			var theta = 0.0;
			for ( var t=0; t < n; t++)  {
				// theta = -2 pi i * t / n;
				X.re[i] += x[t] * Math.cos(theta);
				X.im[i] += x[t] * Math.sin(theta);
				theta += thet; 
			}
			thet -= 2*Math.PI / n;
		}
		return X;
	}
}
function idft(X) {
	// Only recovers real part 
	/*
importScripts("src/experimental/complex.js")
t = 0:512
x = sin(t)
X = dft(x)
plot(modulus(X))
s = idft(X)
plot(s)	
	*/
	if ( !(X instanceof ComplexVector) ) {
		if ( X instanceof Complex)
			return X.re;	
		else if (typeof(X) == "number")
			return X;
		else if ( X instanceof Float64Array)
			return idft(new ComplexVector(X, zeros(X.length), true)); 
		else
			return undefined;
	}
	const n = X.length;
	if ( n == 1) 
		return X.re[0];
	else if ( Math.log2(n) % 1 == 0 )
		return ifft(X);
	else {
		var x = new Float64Array(n);
		var thet = 0.0;
		for ( var t=0; t < n; t++) {
			var theta = 0.0;
			var re = 0.0;
			//var im = 0.0;
			for ( var i=0; i < n; i++)  {
				// theta = 2 pi i * t / n;
				re += X.re[i] * Math.cos(theta) - X.im[i] * Math.sin(theta);
				// im += X[i].im * Math.sin(theta) + X[i].re * Math.cos(theta); // not used for real signals
				theta += thet; 
			}
			x[t] = re / n;
			thet += 2*Math.PI / n;
		}
		return x;
	}
}

function spectrum(x) {
	if ( x instanceof Float64Array ) {
		return absComplex(dft(x));
	}
	else 
		return undefined;
}
