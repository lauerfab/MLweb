function sumsquaresjs (x) {
	var ss = 0;
	for ( var i = 0; i < x.length; i++)
		ss += x[i] * x[i];
	return ss;
}
