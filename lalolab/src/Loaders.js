/*
	TO DO:
		- libsvm model multiclass
		- load msvmpack model
		- load msvmpack data

*/

///////////////////////////////////////////
/// Utilities for loading standard ML models and data
//////////////////////////////////////////
function loadmodel ( url, format ) {
	
	// load a remote file (from same web server only...)
	var xhr = new XMLHttpRequest();
	xhr.open('GET', url, false); // false = synchronous
	xhr.responseType = 'blob';
	xhr.send(null);
	
	var blob = new Blob([xhr.response]);
  	var reader = new FileReaderSync();

	if ( arguments.length < 2 ) 
		var format = "";

  	switch( format.toLowerCase() ) {
  	case "msvmpack":
  		return readMSVMpack(reader.readAsText(blob) );
  		break;
  	case "libsvm":
  	default:
  		return readLibSVM(reader.readAsText(blob) );
  		break;
  	}

}

function loaddata ( url, format ) {
	
	// load a remote file (from same web server only...)
	var xhr = new XMLHttpRequest();
	xhr.open('GET', url, false); // false = synchronous
	xhr.responseType = 'blob';
	xhr.send(null);
	
	var blob = new Blob([xhr.response]);
  	var reader = new FileReaderSync();

	if ( arguments.length < 2 ) 
		var format = "";
  	switch( format.toLowerCase() ) {
  	case "msvmpack":
  		return undefined;
  		break;
  	case "libsvm":
  		return readLibSVMdata(reader.readAsText(blob) );
  		break;
  	default:
  		// Matrix text format
  		return load_data( reader.readAsText(blob) );
  		break;
  	}

}

function load_data ( datastring ) {
	// convert a string into a matrix data 
	var i,j;
	var row;
	var rows = datastring.split("\n");
	if ( rows[rows.length-1] == "" )
		rows.splice(rows.length-1,1);
	var ri ;
	var rowdata;
	ri = removeFirstSpaces(rows[0]);
	row = ri.replace(/,/g," ").replace(/ +/g,",");
	rowdata = row.split(","); 
	const m = rows.length;
	const n = rowdata.length;
	var X = zeros(m, n);

	var k = 0;
	for ( i=0; i< m; i++) {
		ri = removeFirstSpaces(rows[i]);
		if ( ri != "" ) {
			row = ri.replace(/,/g," ").replace(/ +/g,",");
			rowdata = row.split(","); 
			for (j=0;j<n; j++) {
				X.val[k] = parseFloat(rowdata[j]);
				k++;
			}
		}
	}		
	return X;
}

function readLibSVM ( str ) {
	// read a libSVM model from a string (coming from a text file for instance)
	
	// by default libSVM implements one-vs-one decomposition of multi-class problems;
	
	const svm_type_table = ["c_svc","nu_svc","one_class","epsilon_svr","nu_svr"];
	const kernel_type_table = [ ["linear","polynomial","rbf","sigmoid","precomputed"], ["linear","poly","rbf",undefined,undefined]];

	var rows = str.split("\n");

	var i =0;
	while (rows[i] != "SV" && i < rows.length) {
		console.log(rows[i]);
		if(rows[i].indexOf("svm_type")==0) {
			var svm_type_str = rows[i].split(" ")[1];
			if ( svm_type_str != "c_svc" ) 
				return undefined; 						// for now... 	
		}
		else if(rows[i].indexOf("kernel_type")==0) {		
			var kertype_str = rows[i].split(" ")[1];
			var kerneltype = kernel_type_table[1][kernel_type_table[0].indexOf(kertype_str)];			
		}
		else if(rows[i].indexOf("degree")==0)		
			var kernelpar = parseInt(rows[i].split(" ")[1]);
		else if(rows[i].indexOf("gamma")==0) 		
			var kernelpar = Math.sqrt(1 / (2 * parseFloat(rows[i].split(" ")[1]) ) );
		else if(rows[i].indexOf("coef0")==0)  {
			if ( kerneltype =="poly" && parseFloat(rows[i].split(" ")[1]) == 0 )
				kerneltype = "polyh";
		}
		else if(rows[i].indexOf("nr_class")==0) 
			var Nclasses = parseInt(rows[i].split(" ")[1]);
		else if(rows[i].indexOf("total_sv")==0) {		
		}
		else if( rows[i].indexOf("rho")==0) {		
			var rhostr = rows[i].split(" ");
			if (rhostr.length > 2 ) {
				var rho = new Float64Array(rhostr.length - 1 );
				for (var k=1; k < rhostr.length; k++)
					rho[k-1] = parseFloat(rhostr[k]);
			}
			else
				var rho = parseFloat(rhostr[1]);
		}
		else if(rows[i].indexOf("label")==0) {
			var lblstr = rows[i].split(" ");
			var labels = new Array(Nclasses); 
			for(var k=0;k<Nclasses;k++)
				labels[k] = parseInt(lblstr[k+1]);
		}
/*		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probB[i]);
		}
*/
		else if(rows[i].indexOf("nr_sv")==0) {	
			var nSVstr = rows[i].split(" ");	
			var nSV = new Array(Nclasses); 
			for(var k=0;k<Nclasses;k++)
				nSV[k] = parseInt(nSVstr[k+1]);
		}
		i++;		
	}
	i++; // skip "SV" line

	// read sv_coef and SV
	var Nclassifiers = 1;
	if ( Nclasses > 2 ) {
		Nclassifiers = Math.round(Nclasses*(Nclasses-1)/2); 
		
		var alpha = new Array(Nclassifiers); 
		var SV = new Array(Nclassifiers); 
		var SVlabels = new Array(Nclassifiers); 
		var SVindexes = new Array(Nclassifiers); 
	}
	else {
		var totalSVs = sumVector(nSV);
		var alpha = zeros(totalSVs); 
		var SV = new Array(totalSVs);
		var SVlabels = ones(totalSVs); // fake 
		var SVindexes = new Array(); 
	}
	
	var SVi = 0; // counter of SVs
	var current_class = 0;
	var next_class = nSV[current_class]; // at which SVi do we switch to next class
	
	var dim = 1;
	
	while( i < rows.length && rows[i].length > 0) {
		//console.log("SV number : " + SVi);
		
		var row = rows[i].split(" ");
		
		// Read alphai * yi
		
		if ( Nclasses > 2 ) {
			for (var k=current_class; k < Nclasses; k++) {
				var alphaiyi = parseFloat(row[k-current_class]); // do that for all pairwise classifier involving current class
			
				if ( !isZero( alphaiyi ) ) {
					var kpos = current_class;
					var kneg = k-current_class+1;
					var idx = 0; // find indxe of binary classifier
					for (var t = 1; t <= kpos; t++ )
						idx += Nclasses - t;
					idx += kneg-current_class  - 1 ;

					alpha[idx].val[ SVindexes[idx].length ] = alphaiyi;				
				}
			}
		}
		else {
			var alphaiyi = parseFloat(row[0]); // do that for all pairwise classifier involving current class
		
			if ( !isZero( alphaiyi ) ) {				
				alpha[ SVindexes.length ] = alphaiyi; 
				SV[SVindexes.length] = new Array(dim);				
			}
		}
		
		// Read x_i 
		var startk;
		if ( Nclasses == 2)
			startk = 1;
		else
			startk = Nclasses-current_class;
		for ( var k=startk; k < row.length; k++) {
			var indval = row[k].split(":");
			if ( indval[0].length > 0 ) {
				var featureindex = parseInt(indval[0]);
				if (featureindex > dim ) 
					dim = featureindex; 
				featureindex--; // because libsvm indexes start at 1
			
				var featurevalue = parseFloat(indval[1]);
			
				if ( Nclasses > 2 ) {
					for (var c = current_class; c < Nclasses; c++) {
						var kpos = current_class;
						var kneg = c-current_class+1;
						var idx = 0; // find indxe of binary classifier
						for (var t = 1; t <= kpos; t++ )
							idx += Nclasses - t;
						idx += kneg-current_class  - 1 ;
			
						if ( alpha[idx].val[ SVindexes[idx].length ] != 0 ) {
							SV[idx][SVindexes[idx].length * dim + featureindex ] = featurevalue;
						
							SVlabels[idx].push(1); // fake SVlabels = 1 because alpha = alpha_i y_i
						
							SVindexes[idx].push(SVindexes[idx].length); // dummy index (not saved by libsvm)
						}
					}	
				}
				else {
					SV[SVindexes.length][ featureindex ] = featurevalue;						
				}
			}
		}
		// Make SVlabels and SVindexes
		if ( Nclasses > 2 ) {
			for (var c = current_class; c < Nclasses; c++) {
				var kpos = current_class;
				var kneg = c-current_class+1;
				var idx = 0; // find indxe of binary classifier
				for (var t = 1; t <= kpos; t++ )
					idx += Nclasses - t;
				idx += kneg-current_class  - 1 ;
		
				if ( alpha[idx].val[ SVindexes[idx].length ] != 0 ) {
					SVlabels[idx].push(1); // fake SVlabels = 1 because alpha = alpha_i y_i
					SVindexes[idx].push(SVindexes[idx].length); // dummy index (not saved by libsvm)
				}
			}	
		}
		else
			SVindexes.push(SVindexes.length); // dummy index (not saved by libsvm)										
		
		SVi++; 
		if ( SVi >= next_class ) {
			current_class++;
			next_class += nSV[current_class];
		}
		
		
		i++;
	}

	rows = undefined; // free memory
	
	// Build model
	var svm = new Classifier(SVM, {kernel: kerneltype, kernelpar: kernelpar}) ; 
	svm.normalization = "none"; 
	if (Nclasses == 2) {
		svm.labels = labels; 
		svm.numericlabels = labels;
		
		svm.SV = zeros(totalSVs, dim); 
		for ( i=0; i < totalSVs; i++) {
			for ( var j=0; j < dim; j++) {
				if ( SV[i].hasOwnProperty(j) && SV[i][j] ) // some entries might be left undefined...
					svm.SV.val[i*dim+j] = SV[i][j];
			}
		}

	}
	else {
		svm.labels = labels;
		svm.numericlabels = range(Nclasses);
		// SVs..
	}
	svm.dim_input = dim;
	svm.C = undefined;
	
	svm.b = minus(rho);
	svm.alpha = alpha; 
	svm.SVindexes = SVindexes;
	svm.SVlabels = SVlabels; 
	// compute svm.w if linear ?? 
	
	svm.kernelFunc = kernelFunction(kerneltype, kernelpar); // use sparse input vectors?
	
	return svm;	
}

function readLibSVMdata ( str ) {

	// Read a libsvm data file and return {X, y} 
	var i,j;
	var rows = str.split("\n");
	if ( rows[rows.length-1] == "" )
		rows.splice(rows.length-1,1);
	const N = rows.length;
	
	var dim_input = -1;	
	
	var X = new Array(N); 
	var Y = zeros(N);
	for(i = 0; i< N; i++) {	
		// template Array for data values
		X[i] = new Array(Math.max(1,dim_input));  
	
		// Get line as array of substrings
		var row = rows[i].split(" "); 
					
		Y[i] = parseFloat(row[0]);
		for ( j=1; j < row.length; j++) {
			var feature = row[j].split(":");
			var idx = parseInt(feature[0]);
			X[i][idx-1] = parseFloat( feature[1] );		
			
			if ( idx > dim_input)
				dim_input = idx;	
		}		
	}
	rows = undefined; // free memory
	
	var Xmat = zeros(N, dim_input); 
	for(i = 0; i< N; i++) {	
		for ( j in X[i]) {
			if (X[i].hasOwnProperty(j) && X[i][j])
				Xmat.val[i*dim_input + parseInt(j)] = X[i][j];
		}
		X[i] = undefined; // free mem as we go...
	}
	
	return {X: Xmat, y: Y};
}

function readMSVMpack( str ) {
	// read an MSVMpack model from a string

	const MSVMtype = ["WW", "CS", "LLW", "MSVM2"];
	const Ktypes = [undefined, "linear", "rbf", "polyh", "poly"]; 
	
	var rows = str.split("\n");
	var version;
	var inttype; 
	var i;
	if(rows[0].length < 3 ){
		version = 1.0;	// no version number = version 1.0
		inttype = parseInt(rows[0]);
		i = 1;
	}
	else {
		version = parseFloat(rows[0]);
		inttype = parseInt(rows[1]);
		i = 2;
	}
	
	if ( inttype > 3 ) {
		error("Unknown MSVM model type in MSVMpack model file");
		return undefined;
	}
	var model = new Classifier(MSVM, {MSVMtype: MSVMtype[inttype]} ); 
	
	model.MSVMpackVersion = version;
//	model.OptimAlgorithm = FrankWolfe;	// default back to Frank-Wolfe method
					// use -o 1 to retrain with Rosen's method

	var Q = parseInt(rows[i]);
	i++;
	model.labels = range(1,Q+1); 
	model.numericlabels = range(Q);
	
	var intKtype = parseInt(rows[i]);
	i++;
	
	if ( intKtype % 10 <= 4 ) {
		model.kernel = Ktypes[intKtype % 10]; 
	}
	else {
		error( "Unsupported kernel type in MSVMpack model.");
		return undefined;
	}
	
	if ( model.kernel != "linear" ) {
		var rowstr = rows[i].split(" "); 	
		var nKpar = parseInt(rowstr[0]);
		i++;
		if ( nKpar == 1 ) {
			model.kernelpar = parseFloat(rowstr[1]);
		}
		else if (nKpar > 1) {
			error( "MSVMpack with custom kernel cannot be loaded.");
			return undefined;
		}
	}
		
	model.trainingSetName = rows[i];
	i++;
	
	model.trainingError = parseFloat(rows[i]);
	i++;
	
	var nb_data = parseInt(rows[i]);
	i++;

	var dim_input = parseInt(rows[i]);
	i++;
	model.dim_input = dim_input;

	// C hyperparameters
	if (version > 1.0) {
		var Cstr = rows[i].split(" ");
		model.Ck = zeros(Q);
		for(var k=0;k<Q;k++)	
			model.Ck[k] = parseFloat(Cstr[k]);
		model.C = model.Ck[0];
		i++;
	}
	else {
		model.C = parseFloat(rows[i]);
		i++;
	}
	
	var normalization = parseFloat(rows[i]);
	i++;

	if(normalization >= 0.0) {
		var mean = zeros(dim_input);
		var std = zeros(dim_input);
		mean[0] = normalization;
		var meanstr = rows[i].split(" ");
		i++;
		for(var k=1;k<dim_input;k++)
			mean[k] = parseFloat(meanstr[k]);
		var stdstr = rows[i].split(" ");
		i++;
		for(var k=0;k<dim_input;k++)
			std[k] = parseFloat(stdstr[k]);
		model.normalization = {mean:mean, std:std};
	}
	else
		model.normalization = "none";
	
	// Model file format could change with MSVM type
	// The following loads the model accordingly
	var k;
	var ii;
	
	// Allocate memory for model parameters
	var alpha = zeros(nb_data,Q); 
	model.b = zeros(Q); 
	var SVindexes = new Array(); 
	var SV = new Array(); 
	var SVlabels = new Array(); 

	// Read b
	var rowstr = rows[i].split(" ");
	i++;
	for(k=0; k<Q; k++)
		model.b[k] = parseFloat(rowstr[k]);
		
	var nSV = 0;
	for(ii=0; ii < nb_data; ii++) {

		// Read [alpha_i1 ... alpha_iQ]
		var rowstr = rows[i].split(" ");
		rows[i] = undefined;
		i++;
		var isSV = false;
		for(k=0; k<Q; k++) {
			alpha.val[ii*Q + k] = parseFloat(rowstr[k]);
			isSV = isSV || (alpha.val[ii*Q + k] != 0) ;
		}
		if ( isSV ) {
			// Read x_i
			var rowstr = rows[i].split(" ");
			rows[i] = undefined;
			i++;
			SV.push(new Array(dim_input));
			for(k=0; k<dim_input; k++)
				SV[nSV][k] = parseFloat(rowstr[k]);
		
			// Read y_i
			SVlabels.push( parseInt(rowstr[dim_input]) - 1 );
			
			SVindexes.push(ii);
			nSV++;
		}
		else 
			i++;
	}
	model.alpha = transpose(alpha);
	model.SVindexes = SVindexes;
	model.SVlabels = SVlabels;
	model.SV = mat(SV, true);
	
	return model;
}
