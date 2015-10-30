/*
	Utility functions for ML
*/ 

/////////////////////////////////////
/// Text functions
/////////////////////////////////////
function text2vector( str, dictionnary , multi) {
	var countmulti = false;
	if ( typeof(multi) != "undefined" && multi ) 
		countmulti = true;
	
	var words = str.split(/[ ,;:\?\!\(\)\.\n]+/g);
	var vec = zeros(dictionnary.length);
	for ( var w = 0; w < words.length; w++ )  {
		if ( words[w].length > 2 ) {
			var idx = dictionnary.indexOf( words[w].toLowerCase() );
			if ( idx >= 0 && (countmulti || vec[idx] == 0) )
				vec[ idx ]++;
		}
	}

	return vec;
}
function text2sparsevector( str, dictionnary, multi ) {
	var countmulti = false;
	if ( typeof(multi) != "undefined" && multi ) {
		countmulti = true;
		var val = new Array();
	}
	
	var words = str.split(/[ ,;:\?\!\(\)\.\n]+/g);
	var indexes = new Array(); 
	
	for ( var w = 0; w < words.length; w++ )  {
		if ( words[w].length > 2 ) {
			var idx = dictionnary.indexOf(words[w].toLowerCase() );
			if ( idx >= 0 ) {
				if ( countmulti ) {
					if ( indexes.indexOf(idx) < 0) {
						val.push(1);
						indexes.push( idx );
					}
					else {
						val[indexes.indexOf(idx)] ++;
					}
				}
				else {
					if (indexes.indexOf(idx) < 0) 
						indexes.push( idx );
				}
			}
		}
	}
	if ( countmulti ) {
		var idxsorted = sort(indexes, false, true);
		val = get(val,idxsorted);
		var vec = new spVector(dictionnary.length, val, indexes);
	}
	else
		var vec = new spVector(dictionnary.length, ones(indexes.length),sort(indexes));
	return vec;
}
