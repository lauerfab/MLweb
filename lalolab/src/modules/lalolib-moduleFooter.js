
return {
	// lalolibbase.js
	// constants?
	laloprint: laloprint,
	//plot: plot, 
	//colorplot: colorplot,
	//plot3: plot3,
	//image: image,
	lalo: lalo,
	Lalolab: Lalolab,
	load_data: load_data,
	
	/* laloplots.js (not available in module because need access to global scope and html document)
	Diagram: Diagram,
	Plot: Plot,
	ColorPlot: ColorPlot,
	Plot2D: Plot2D,
	Plot3D: Plot3D,
	*/
	
	// Constants in linalg.js
	LALOLIB_ERROR: LALOLIB_ERROR,
	EPS: EPS,
	
	
	// linalg.js
	isZero: isZero,
	tic: tic,
	toc: toc,
	type: type,
	isArrayOfNumbers: isArrayOfNumbers,
	isScalar: isScalar,
	printVector: printVector,
	Matrix: Matrix,
	array2mat: array2mat,
	array2vec: array2vec, 
	size: size,
	ones: ones,
	zeros: zeros,
	eye: eye,
	diag: diag,
	vec: vec,
	matrixCopy: matrixCopy,
	vectorCopy: vectorCopy,
	vectorCopyInto: vectorCopyInto,
	arrayCopy: arrayCopy,
	appendRow: appendRow,
	reshape: reshape,
	get: get,
	getSubMatrix: getSubMatrix,
	getRows: getRows,
	getCols: getCols,
	getSubVector: getSubVector,
	getSubArray: getSubArray,
	getrowref: getrowref,
	set: set,
	setVectorScalar: setVectorScalar,
	setVectorVector: setVectorVector,
	setMatrixScalar: setMatrixScalar,
	setMatrixMatrix: setMatrixMatrix,
	setMatrixColVector: setMatrixColVector,
	setMatrixRowVector: setMatrixRowVector,
	setRows: setRows,
	setCols: setCols,
	dense: dense,
	supp: supp,
	range: range,
	swaprows: swaprows,
	swapcols: swapcols,
	randnScalar: randnScalar,
	randn: randn,
	randVector: randVector,
	randMatrix: randMatrix,
	rand: rand,
	randnsparse: randnsparse,
	randsparse: randsparse,
	randperm: randperm,
	// missing: MathFunctions + MathFunctionsVector...
	apply: apply,
	aaplyVector: applyVector,
	applyMatrix: applyMatrix,
	applyComplexVector: applyComplexVector,
	applyComplexMatrix: applyComplexMatrix,
	mul: mul,
	mulScalarVector: mulScalarVector,
	mulScalarMatrix: mulScalarMatrix,
	dot: dot,
	mulMatrixVector: mulMatrixVector,
	mulMatrixTransVector: mulMatrixTransVector,
	mulMatrixMatrix: mulMatrixMatrix,
	entrywisemulVector: entrywisemulVector,
	entrywisemulMatrix: entrywisemulMatrix,
	entrywisemul: entrywisemul,
	saxpy: saxpy,
	gaxpy: gaxpy,
	divVectorScalar: divVectorScalar,
	divScalarVector: divScalarVector,
	divVectors: divVectors,
	divMatrixScalar: divMatrixScalar,
	divScalarMatrix: divScalarMatrix,
	divMatrices: divMatrices,
	entrywisediv: entrywisediv,
	outerprodVectors: outerprodVectors,
	outerprod: outerprod,
	addScalarVector: addScalarVector,
	addScalarMatrix: addScalarMatrix,
	addVectors: addVectors,
	addMatrices: addMatrices,
	add: add,
	subScalarVector: subScalarVector,
	subVectorScalar: subVectorScalar,
	subScalarMatrix: subScalarMatrix,
	subMatrixScalar: subMatrixScalar,
	subVectors: subVectors,
	subMatrices: subMatrices,
	sub: sub,
	pow: pow,
	minus: minus,
	minusVector: minusVector,
	minusMatrix: minusMatrix,
	minVector: minVector,
	minMatrix: minMatrix,
	minVectorScalar: minVectorScalar,
	minMatrixScalar: minMatrixScalar,
	minMatrixRows: minMatrixRows,
	minMatrixCols: minMatrixCols,
	minVectorVector: minVectorVector,
	minMatrixMatrix: minMatrixMatrix,
	min: min,
	maxVector: maxVector,
	maxMatrix: maxMatrix,
	maxVectorScalar: maxVectorScalar,
	maxMatrixScalar: maxMatrixScalar,
	maxMatrixRows: maxMatrixRows,
	maxMatrixCols: maxMatrixCols,
	maxVectorVector: maxVectorVector,
	maxMatrixMatrix: maxMatrixMatrix,
	max: max, 
	transposeMatrix: transposeMatrix,
	transposeVector: transposeVector,
	transpose: transpose,
	det: det,
	trace: trace,
	triiu: triu,
	tril: tril,
	issymmetric: issymmetric,
	mat: mat,
	isEqual: isEqual,
	isNotEqual: isNotEqual,
	isGreater: isGreater,
	isGreaterOrEqual: isGreaterOrEqual,
	isLower: isLower,
	isLowerOrEqual: isLowerOrEqual,
	find: find,
	argmax: argmax,
	findmax: findmax,
	argmin: argmin,
	findmin: findmin, 
	sort: sort,
	sumVector: sumVector,
	sumMatrix: sumMatrix,
	sumMatrixRows: sumMatrixRows,
	sumMatrixCols: sumMatrixCols,
	sum: sum,
	prodVector: prodVector,
	prodMatrix: prodMatrix,
	prodMatrixRows: prodMatrixRows,
	prodMatrixCols: prodMatrixCols,
	prod: prod,
	mean: mean,
	variance: variance,
	std: std,
	cov: cov,
	xtx: xtx,
	norm: norm,
	norm1: norm1,
	norminf: norminf,
	normp: normp,
	normnuc: normnuc,
	norm0: norm0,
	norm0Vector: norm0Vector,
	solve: solve,
	cholsolve: cholsolve,
	// solveQR...
	inv: inv,
	chol: chol,
	ldlsymmetricpivoting: ldlsymmetricpivoting,
	qr: qr,
	solvecg: solvecg,
	cgnr: cgnr,
	eig: eig,
	eigs: eigs,
	svd: svd,
	rank: rank,
	nullspace: nullspace,
	orth: orth,
	
	// stats.js
	nchoosek: nchoosek,
	mvnrnd: mvnrnd,
	Distribution: Distribution,
	Uniform: Uniform,
	Gaussian: Gaussian,
	mvGaussian: mvGaussian,
	Bernoulli: Bernoulli,
	Poisson: Poisson,
	
	// sparse.js
	spVector: spVector,
	spMatrix: spMatrix,
	spgetRows: spgetRows,
	fullVector: fullVector,
	fullMatrix: fullMatrix,
	full: full,
	sparseVector: sparseVector,
	sparseMatrix: sparseMatrix,
	sparseMatrixRowMajor: sparseMatrixRowMajor,
	sparse: sparse,
	speye: speye,
	spdiag: spdiag,
	transposespVector: transposespVector,
	transposespMatrix: transposespMatrix,
	spmat: spmat,
	mulScalarspVector: mulScalarspVector,
	mulScalarspMatrix: mulScalarspMatrix,
	spdot: spdot,
	dotspVectorVector: dotspVectorVector,
	mulMatrixspVector: mulMatrixspVector,
	mulspMatrixVector: mulspMatrixVector,
	mulspMatrixTransVector: mulspMatrixTransVector,
	mulspMatrixspVector: mulspMatrixspVector,
	mulspMatrixTransspVector: mulspMatrixTransspVector,
	mulspMatrixspMatrix: mulspMatrixspMatrix,
	mulMatrixspMatrix: mulMatrixspMatrix,
	mulspMatrixMatrix: mulspMatrixMatrix,
	entrywisemulspVectors: entrywisemulspVectors,
	entrywisemulspVectorVector: entrywisemulspVectorVector,
	entrywisemulspMatrices: entrywisemulspMatrices,
	entrywisemulspMatrixMatrix: entrywisemulspMatrixMatrix,
	addScalarspVector: addScalarspVector,
	addVectorspVector: addVectorspVector,
	addspVectors: addspVectors,
	addScalarspMatrix: addScalarspMatrix,
	addMatrixspMatrix: addMatrixspMatrix,
	addspMatrices: addspMatrices,
	spsaxpy: spsaxpy,
	subScalarspVector: subScalarspVector,
	subVectorspVector: subVectorspVector,
	subspVectorVector: subspVectorVector,
	subspVectors: subspVectors,
	subScalarspMatrix: subScalarspMatrix,
	subspMatrixMatrix: subspMatrixMatrix,
	subMatrixspMatrix: subMatrixspMatrix,
	subspMatrices: subspMatrices,
	applyspVector: applyspVector,
	applyspMatrix: applyspMatrix,
	sumspVector: sumspVector,
	sumspMatrix: sumspMatrix,
	sumspMatrixRows: sumspMatrixRows,
	sumspMatrixCols: sumspMatrixCols,
	prodspMatrixRows: prodspMatrixRows,
	prodspMatrixCols: prodspMatrixCols,
	
	// complex.js
	Complex: Complex,
	addComplex: addComplex,
	addComplexReal: addComplexReal,
	subComplex: subComplex,
	minusComplex: minusComplex,
	mulComplex: mulComplex,
	mulComplexReal: mulComplexReal,
	divComplex: divComplex,
	conj: conj,
	modulus: modulus,
	absComplex: absComplex,
	expComplex: expComplex,
	ComplexVector: ComplexVector,
	ComplexMatrix: ComplexMatrix,
	real: real,
	imag: imag,
	transposeComplexMatrix: transposeComplexMatrix,
	addComplexVectors: addComplexVectors,
	subComplexVectors: subComplexVectors,
	addComplexMatrices: addComplexMatrices,
	subComplexMatrices: subComplexMatrices,
	addComplexVectorVector: addComplexVectorVector,
	subComplexVectorVector: subComplexVectorVector,
	addComplexMatrixMatrix: addComplexMatrixMatrix,
	subComplexMatrixMatrix: subComplexMatrixMatrix,
	addScalarComplexVector: addScalarComplexVector,
	subScalarComplexVector: subScalarComplexVector,
	addScalarComplexMatrix: addScalarComplexMatrix,
	entrywisemulComplexVectors: entrywisemulComplexVectors,
	entrywisedivComplexVectors: entrywisedivComplexVectors,
	entrywisemulComplexMatrices: entrywisemulComplexMatrices,
	entrywisedivComplexMatrices: entrywisedivComplexMatrices,
	entrywisemulComplexVectorVector: entrywisemulComplexVectorVector,
	entrywisemulComplexMatrixMatrix: entrywisemulComplexMatrixMatrix,
	minusComplexVector: minusComplexVector,
	minusComplexMatrix: minusComplexMatrix,
	sumComplexVector: sumComplexVector,
	sumComplexMatrix: sumComplexMatrix,
	norm1ComplexVector: norm1ComplexVector,
	norm2ComplexVector: norm2ComplexVector,
	normFroComplexMatrix: normFroComplexMatrix,
	dotComplexVectors: dotComplexVectors,
	dotComplexVectorVector: dotComplexVectorVector,
	mulScalarComplexVector: mulScalarComplexVector,
	mulComplexComplexVector: mulComplexComplexVector,
	mulComplexVector: mulComplexVector,
	mulScalarComplexMatrix: mulScalarComplexMatrix,
	mulComplexComplexMatrix: mulComplexComplexMatrix,
	mulComplexMatrix: mulComplexMatrix,
	mulComplexMatrixVector: mulComplexMatrixVector,
	mulComplexMatrixComplexVector: mulComplexMatrixComplexVector,
	mulComplexMatrices: mulComplexMatrices,
	mulComplexMatrixMatrix: mulComplexMatrixMatrix,
	fft: fft,
	ifft: ifft,
	dft: dft,
	idft: idft,
	spectrum: spectrum,	
	
	
	// laloglpk.js
	lp: lp,
	linprog: linprog,
	minl1: minl1,
	minl0: minl0,
	qp: qp,
	quadprog: quadprog,
	minimize: minimize,
	secant: secant,
	steepestdescent: steepestdescent, 
	bfgs: bfgs
	
}
}));
