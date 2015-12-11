/////////////////////////////////////
/// Stand alone lalolib base functions
////////////////////////////////////
var printPrecision = 3; // number of digits to print

var LALOLibPlotsIndex = 0;
var LALOLibPlots = new Array();
var LALOLABPLOTMOVING = false;

//////////////////////////
//// Cross-browser compatibility
///////////////////////////

if( typeof(console) == "undefined" ) {
	// for Safari
	var console = {log: function ( ) { } };
}

if( typeof(Math.sign) == "undefined" ) {
	// for IE, Safari
	Math.sign = function ( x ) { return ( x>=0 ? (x==0 ? 0 : 1) : -1 ) ;}
}

//////////////////////////
//// printing
///////////////////////////

function laloprint( x , htmlId, append ) {
	/*
		use print(x) to print to the standard LALOLabOutput

		use print(x, id) to print to another html entity
		
		use str = print(x, true) to get the resulting string
	*/
	
	if ( typeof(htmlId) == "undefined" )
		var htmlId = "LALOLibOutput"; 
	if ( typeof(append) == "undefined" )
		var append = true; 
				
	return printMat(x, size(x), htmlId, append ) ;
}

function printMat(A, size, htmlId, append) {
	if (typeof(append) === "undefined")
		var append = false;
	if ( typeof(htmlId) == "undefined" || htmlId === true ) {
		// return a string as [ [ .. , .. ] , [.. , ..] ]
		if ( type(A) == "matrix" ) {
			var str = "[";
			var i;
			var j;
			var m = size[0];
			var n = size[1];

			for (i=0;i<m; i++) {
				str += "[";
				for ( j=0; j< n-1; j++)
					str += printNumber(A.val[i*A.n+j]) + ",";
				if ( i < m-1)
					str += printNumber(A.val[i*A.n+j]) + "]; ";
				else
					str += printNumber(A.val[i*A.n+j]) + "]";
			}			
			str += "]";
			return str;
		}
		else if (type(A) == "vector" ) {
			var n = A.length;
			var str = "";
			// Vector (one column)
			for (var i=0;i<n; i++) {
				str += "[ " + printNumber(A[i]) + " ]<br>";
			}
			console.log(str);
			return str;
		}
	}
	else {
		// Produce HTML code and load it in htmlId
		
		var html = "";
		var i;
		var j;
		
		/*if (domathjax) {
			html = tex ( A ) ;
		}
		else {*/
			if ( isScalar(A) ) {
				html +=  A + "<br>" ;
			}
			else if (type(A) == "vector" ) {
				var n = size[0];

				// Vector (one column)
				for (i=0;i<n; i++) {
					html += "[ " + printNumber(A[i]) + " ]<br>";
				}
			}
			else {
				// Matrix
				var m = size[0];
				var n = size[1];

				for (i=0;i<m; i++) {
					html += "[ ";
					for(j=0;j < n - 1; j++) {
						html += printNumber(A.val[i*A.n+j]) + ", ";
					}
					html += printNumber(A.val[i*A.n+j]) + " ]<br>";
				}
			}
		//}
		if (append)
			document.getElementById(htmlId).innerHTML += html;
		else
			document.getElementById(htmlId).innerHTML = html;
		/*
		if ( domathjax) 
			MathJax.Hub.Queue(["Typeset",MathJax.Hub,"output"]);			
			*/
	}
}

function printNumber ( x ) {
	switch ( typeof(x) ) {
		case "undefined":
			return "" + 0;// for sparse matrices
			break;
		case "string":
			/*if ( domathjax ) 
				return "\\verb&" + x + "&";
			else*/
				return x;
			break;
		case "boolean":
			return x;
			break;
		default:	
			if ( x == Infinity )
				return "Inf";
			if ( x == -Infinity )
				return "-Inf";
			var x_int = Math.floor(x);
			if ( Math.abs( x - x_int ) < 2.23e-16 ) {
				return "" + x_int;
			} 
			else
				return x.toFixed( printPrecision );
				
			break;
	}
}

//// Error handling

function error( msg ) {
	throw new Error ( msg ) ;	
//	postMessage( {"error": msg} );
}


/////////// 
// Plots
//////////
function plot(multiargs) {
	// plot(x,y,"style", x2,y2,"style",y3,"style",... )
	
	// Part copied from lalolabworker.js
	
	var data = new Array();
	var styles = new Array();	
	var legends = new Array();		
	var minX = Infinity;
	var maxX = -Infinity;
	var minY = Infinity;
	var maxY = -Infinity;
	
	var p=0; // argument pointer
	var x;
	var y;
	var style;
	var i;
	var n;
	var c = 0; // index of current curve
	while ( p < arguments.length)  {
	
		if ( type( arguments[p] ) == "vector" ) {

			if ( p + 1 < arguments.length && type ( arguments[p+1] ) == "vector" ) {
				// classic (x,y) arguments
				x = arguments[p];
				y = arguments[p+1];
			
				p++;
			}
			else {
				// only y provided => x = 0:n
				y = arguments[p];
				x = range(y.length);
			}
		}
		else if ( type( arguments[p] ) == "matrix" ) {
			// argument = [x, y]
			if ( arguments[p].n == 1 ) {
				y = arguments[p].val;
				x = range(y.length);
			}
			else if (arguments[p].m == 1 ) {
				y = arguments[p].val;
				x = range(y.length);
			}
			else if ( arguments[p].n == 2 ) {
				// 2 columns => [x,y]
				x = getCols(arguments[p], [0]);
				y = getCols(arguments[p], [1]);
			}			
			else {
				// more columns => trajectories as rows
				x = range(arguments[p].n);
				for ( var row = 0; row < arguments[p].m; row++) {
					y = arguments[p].row(row);
					data[c] = [new Array(x.length), new Array(x.length)];
					for ( i=0; i < x.length; i++) {
						data[c][0][i] = x[i];
						data[c][1][i] = y[i]; 
						if ( x[i] < minX )
							minX = x[i];
						if(x[i] > maxX ) 
							maxX = x[i];
						if ( y[i] > maxY ) 
							maxY = y[i];
						if ( y[i] < minY ) 
							minY = y[i];

					}
		
					styles[c] = undefined;
					legends[c] = "";
		
					// Next curve
					c++; 
				}
				p++;
				continue;
			}
		}
		else {
			return "undefined";
		}
				
		//Style
		style = undefined;
		if ( p + 1 < arguments.length && type ( arguments[p+1] ) == "string" ) {
			style = arguments[p+1];
			p++;
		}			
		legend = "";	
		if ( p + 1 < arguments.length && type ( arguments[p+1] ) == "string" ) {
			legend = arguments[p+1];
			p++;
		}	

		// Add the curve (x,y, style) to plot		
		data[c] = [new Array(x.length), new Array(x.length)];		
		for ( i=0; i < x.length; i++) {
			data[c][0][i] = x[i];
			data[c][1][i] = y[i]; 
			if ( x[i] < minX )
				minX = x[i];
			if(x[i] > maxX ) 
				maxX = x[i];
			if ( y[i] > maxY ) 
				maxY = y[i];
			if ( y[i] < minY ) 
				minY = y[i];

		}
		styles[c] = style;
		legends[c] = legend;
		
		// Next curve
		c++; 
		p++; // from next argument	
	}	
		
	var widthX = maxX-minX;
	var widthY = Math.max( maxY-minY, 1);

	maxX += 0.1*widthX;
	minX -= 0.1*widthX;
	maxY += 0.1*widthY;
	minY -= 0.1*widthY;
	
	if ( minY > 0 ) 
		minY = -0.1*maxY;
	
	if ( maxY < 0 ) 
		maxY = -0.1*minY;
	
	var scaleY = 0.9 * (maxX-minX) / (2*maxY);

	var plotinfo = {"data" : data, "minX" : minX, "maxX" : maxX, "minY" : minY, "maxY": maxY, "styles" : styles, "legend": legends };

	//////// Part from laloplots.html //////////
	
	var plotid = "LALOLibPlot" + LALOLibPlotsIndex;
	var legendwidth = 50;
	
	LALOLibOutput.innerHTML += "<br><div style='position:relative;left:0px;top:0px;text-align:left;'> <div><a onmousemove='mouseposition(event," + LALOLibPlotsIndex + ");' onmousedown='mousestartmove(event," + LALOLibPlotsIndex + ");' onmouseup='mousestopmove(event);' onmouseleave='mousestopmove(event);' ondblclick='zoomoriginal(" + LALOLibPlotsIndex + ");'><canvas id='" +plotid + "'  width='500' height='500' style='border: 1px solid black;'></canvas></a></div> <label id='lblposition" + LALOLibPlotsIndex + "'></label> <div style='position: absolute;left: 550px;top: -1em;'> <canvas id='legend" + LALOLibPlotsIndex + "' width='" + legendwidth + "' height='500'></canvas></div> <div id='legendtxt" + LALOLibPlotsIndex + "' style='position: absolute;left: 610px;top: 0;'></div> </div>";

	// prepare legend
	var ylegend = 20;
	
	// do plot

	LALOLibPlots[LALOLibPlotsIndex] = new Plot(plotid) ;
		
	LALOLibPlots[LALOLibPlotsIndex].setScalePlot(plotinfo.minX, plotinfo.maxX, 200, plotinfo.scaleY); 
	if ( plotinfo.minY && plotinfo.maxY ) {
		LALOLibPlots[LALOLibPlotsIndex].view(plotinfo.minX, plotinfo.maxX, plotinfo.minY, plotinfo.maxY); 
	}
	
	var colors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0];
	
	var p;
	var color;
	for (p = 0; p<plotinfo.data.length; p++) {
		
		var linestyle = true;
		var pointstyle = true;
		if ( typeof(plotinfo.styles[p]) == "string" ) {
			if ( plotinfo.styles[p].indexOf(".") >= 0 ) {
				linestyle = false;
				plotinfo.styles[p] = plotinfo.styles[p].replace(".","");
			}
			if ( plotinfo.styles[p].indexOf("_") >= 0 ) {
				pointstyle = false;
				plotinfo.styles[p] = plotinfo.styles[p].replace("_","");
			}
			color = parseColor(plotinfo.styles[p]);
		
			if ( color < 0 )
				color = colors.splice(0,1)[0];		// pick next unused color
			else
				colors.splice(colors.indexOf(color),1); // remove this color
		}
		else 
			color = color = colors.splice(0,1)[0];	// pick next unused color
		
		if ( typeof(color) == "undefined")	// pick black if no next unused color
			color = 0;
	
		for ( i=0; i < plotinfo.data[p][0].length; i++) {
			if ( pointstyle )
				LALOLibPlots[LALOLibPlotsIndex].addPoint(plotinfo.data[p][0][i],plotinfo.data[p][1][i], color);	
			if ( linestyle && i < plotinfo.data[p][0].length-1 ) 
				LALOLibPlots[LALOLibPlotsIndex].plot_line(plotinfo.data[p][0][i],plotinfo.data[p][1][i], plotinfo.data[p][0][i+1],plotinfo.data[p][1][i+1], color);				
		}
		
		
		// Legend
		if ( plotinfo.legend[p] != "" ) {		
			var ctx = document.getElementById("legend" +LALOLibPlotsIndex).getContext("2d");
			setcolor(ctx, color);
			ctx.lineWidth = "3";
			if ( pointstyle ) {
				ctx.beginPath();
				ctx.arc( legendwidth/2 , ylegend, 5, 0, 2 * Math.PI , true);
				ctx.closePath();
				ctx.fill();
			}
			if( linestyle) {
				ctx.beginPath();
				ctx.moveTo ( 0,ylegend);
				ctx.lineTo (legendwidth, ylegend);
				ctx.stroke();
			}
			ylegend += 20;
			
			document.getElementById("legendtxt" +LALOLibPlotsIndex).innerHTML += plotinfo.legend[p] + "<br>";						
		}
	}
	for ( var pi=0; pi <= LALOLibPlotsIndex; pi++)
		LALOLibPlots[pi].replot();
	
	// ZOOM	
	if(window.addEventListener)
        document.getElementById(plotid).addEventListener('DOMMouseScroll', this.mousezoom, false);//firefox
 
    //for IE/OPERA etc
    document.getElementById(plotid).onmousewheel = this.mousezoom;
	
	LALOLibPlotsIndex++;
}

// Color plot
function colorplot(multiargs) {
	// colorplot(x,y,z) or colorplot(X) or colorplot(..., "cmapname" )

	// Part copied from lalolabworker.js
	
	var minX = Infinity;
	var maxX = -Infinity;
	var minY = Infinity;
	var maxY = -Infinity;
	var minZ = Infinity;
	var maxZ = -Infinity;
	
	var x;
	var y;
	var z;
	var i;

	var t0 =  type( arguments[0] );
	if ( t0 == "matrix" && arguments[0].n == 3 ) {
		x = getCols(arguments[0], [0]);
		y = getCols(arguments[0], [1]);
		z = getCols(arguments[0], [2]);
	}
	else if ( t0 == "matrix" && arguments[0].n == 2 && type(arguments[1]) == "vector" ) {
		x = getCols(arguments[0], [0]);
		y = getCols(arguments[0], [1]);
		z = arguments[1];
	}
	else if (t0 == "vector" && type(arguments[1]) == "vector" && type(arguments[2]) == "vector") {
		x = arguments[0];
		y = arguments[1];
		z = arguments[2];
	}
	else {
		return "undefined";
	}
	
	var minX = min(x);
	var maxX = max(x);
	var minY = min(y);
	var maxY = max(y);
	var minZ = min(z);
	var maxZ = max(z);
	
	var widthX = maxX-minX;
	var widthY = Math.max( maxY-minY, 1);

	maxX += 0.1*widthX;
	minX -= 0.1*widthX;
	maxY += 0.1*widthY;
	minY -= 0.1*widthY;
	
	if ( minY > 0 ) 
		minY = -0.1*maxY;
	
	if ( maxY < 0 ) 
		maxY = -0.1*minY;

	var plotinfo = {"x" : x, "y": y, "z": z, "minX" : minX, "maxX" : maxX, "minY" : minY, "maxY": maxY,  "minZ" : minZ, "maxZ" : maxZ };

	//////// Part from laloplots.html //////////
		
	var plotid = "LALOLibPlot" + LALOLibPlotsIndex;
	var legendwidth = 50;
	
	LALOLibOutput.innerHTML += "<br><div style='position:relative;left:0px;top:0px;text-align:left;'> <div><a onmousemove='mouseposition(event," + LALOLibPlotsIndex + ");' onmousedown='mousestartmove(event," + LALOLibPlotsIndex + ");' onmouseup='mousestopmove(event);' onmouseleave='mousestopmove(event);' ondblclick='zoomoriginal(" + LALOLibPlotsIndex + ");'><canvas id='" +plotid + "'  width='500' height='500' style='border: 1px solid black;'></canvas></a></div> <label id='lblposition" + LALOLibPlotsIndex + "'></label> <div style='position: absolute;left: 550px;top: -1em;'> <canvas id='legend" + LALOLibPlotsIndex + "' width='" + legendwidth + "' height='500'></canvas></div> <div id='legendtxt" + LALOLibPlotsIndex + "' style='position: absolute;left: 610px;top: 0;'></div> </div>";
	
	LALOLibPlots[LALOLibPlotsIndex] = new ColorPlot(plotid) ;
	LALOLibPlots[LALOLibPlotsIndex].setScale(plotinfo.minX, plotinfo.maxX, plotinfo.minY, plotinfo.maxY,plotinfo.minZ, plotinfo.maxZ); 
	LALOLibPlots[LALOLibPlotsIndex].view(plotinfo.minX, plotinfo.maxX, plotinfo.minY, plotinfo.maxY); 	
	
	for (var i=0; i < plotinfo.x.length; i++)
		LALOLibPlots[LALOLibPlotsIndex].addPoint(plotinfo.x[i],plotinfo.y[i],plotinfo.z[i]);
	
	LALOLibPlots[LALOLibPlotsIndex].replot();
	
	var legendwidth = 50;
//	plotlegend.innerHTML += plotinfo.maxZ.toFixed(3) + "<br><canvas id='legend'  width='" + legendwidth + "' height='500'></canvas><br>" + plotinfo.minZ.toFixed(3);
	var ctx = document.getElementById("legend" +LALOLibPlotsIndex).getContext("2d");

	var y;
	for (var i=0; i< LALOLibPlots[LALOLibPlotsIndex].cmap.length;i++) {
		y = Math.floor(i * legend.height / plot1.cmap.length);
		ctx.fillStyle = "rgb(" + LALOLibPlots[LALOLibPlotsIndex].cmap[i][0] + "," + LALOLibPlots[LALOLibPlotsIndex].cmap[i][1] + "," + LALOLibPlots[LALOLibPlotsIndex].cmap[i][2] + ")";
		ctx.fillRect( 0, legend.height-y, legendwidth , (legend.height / LALOLibPlots[LALOLibPlotsIndex].cmap.length) + 1) ;
	}	
	
		
	if(window.addEventListener)
        plotcanvas.addEventListener('DOMMouseScroll', this.mousezoom, false);//firefox
 
    //for IE/OPERA etc
    plotcanvas.onmousewheel = this.mousezoom;
	
	LALOLibPlotsIndex++;
}

// 3D plot
function plot3(multiargs) {
	// plot3(x,y,z,"style", x2,y2,z2,"style",... )
	
	var data = new Array();
	var styles = new Array();	
	var legends = new Array();		
	
	var p=0; // argument pointer
	var x;
	var y;
	var z;
	var style;
	var i;
	var n;
	var c = 0; // index of current curve
	while ( p < arguments.length)  {
	
		if ( type( arguments[p] ) == "vector" ) {

			if ( p + 2 < arguments.length && type ( arguments[p+1] ) == "vector" && type ( arguments[p+2] ) == "vector" ) {
				// classic (x,y,z) arguments
				x = arguments[p];
				y = arguments[p+1];
				z = arguments[p+2];				
				
				p += 2;
			}
			else {
				return "undefined";
			}
		}
		else if ( type( arguments[p] ) == "matrix" ) {
			// argument = [x, y, z]
			n = arguments[p].length;
			x = new Array(n);
			y = new Array(n);
			z = new Array(n);
			for ( i=0; i < n; i++) {
				x[i] = get(arguments[p], i, 0); 
				y[i] = get(arguments[p], i, 1);				
				z[i] = get(arguments[p], i, 2);					
			}
		}
		else {
			return "undefined";
		}
				
		//Style
		style = undefined;
		if ( p + 1 < arguments.length && type ( arguments[p+1] ) == "string" ) {
			style = arguments[p+1];
			p++;
		}			
		legend = "";	
		if ( p + 1 < arguments.length && type ( arguments[p+1] ) == "string" ) {
			legend = arguments[p+1];
			p++;
		}	

		// Add the curve (x,y,z, style) to plot
		data[c] = new Array();
		for ( i=0; i < x.length; i++) {
			data[c][i] = [x[i], y[i], z[i]]; 			
		}
		styles[c] = style;
		legends[c] = legend;
		
		// Next curve
		c++; 
		p++; // from next argument	
				
	}	
			

	var plotinfo =  { "data" : data, "styles" : styles, "legend": legends };
	
	//////// Part from laloplots.html //////////
	
	var plotid = "LALOLibPlot" + LALOLibPlotsIndex;
	var legendwidth = 50;
	
	LALOLibOutput.innerHTML += '<br><div style="position:relative;left:0px;top:0px;text-align:left;"> <div><a onmousedown="LALOLibPlots[' + LALOLibPlotsIndex + '].mousedown(event);" onmouseup="LALOLibPlots[' + LALOLibPlotsIndex + '].mouseup(event);" onmousemove="LALOLibPlots[' + LALOLibPlotsIndex + '].mouserotation(event);"><canvas id="' + plotid + '" width="500" height="500" style="border: 1px solid black;" title="Hold down the mouse button to change the view and use the mousewheel to zoom in or out." ></canvas></a></div><label id="lblposition' + LALOLibPlotsIndex + '"></label> <div style="position: absolute;left: 550px;top: -1em;"> <canvas id="legend' + LALOLibPlotsIndex + '" width="' + legendwidth + '" height="500"></canvas></div> <div id="legendtxt' + LALOLibPlotsIndex + '" style="position: absolute;left: 610px;top: 0;"></div> </div>';
	
	var ylegend = 20;
	
	// do plot

	LALOLibPlots[LALOLibPlotsIndex] = new Plot3D(plotid) ;
	
	LALOLibPlots[LALOLibPlotsIndex].cameraDistance = 30; 
	LALOLibPlots[LALOLibPlotsIndex].angleX = Math.PI/10;	
	LALOLibPlots[LALOLibPlotsIndex].angleZ = Math.PI/10;
	
	LALOLibPlots[LALOLibPlotsIndex].axisNameX1 = "x";
	LALOLibPlots[LALOLibPlotsIndex].axisNameX2 = "y";		
	LALOLibPlots[LALOLibPlotsIndex].axisNameX3 = "z";		
		
	
	var colors = [1,2,3,4,5,0];
	
	var p;
	var color;

	for (p = 0; p<plotinfo.data.length; p++) {
		
		var linestyle = false;
		var pointstyle = true;
		if ( typeof(plotinfo.styles[p]) == "string" ) {
			if ( plotinfo.styles[p].indexOf(".") >= 0 ) {
				linestyle = false;
				plotinfo.styles[p] = plotinfo.styles[p].replace(".","");
			}
			if ( plotinfo.styles[p].indexOf("_") >= 0 ) {
				pointstyle = false;
				plotinfo.styles[p] = plotinfo.styles[p].replace("_","");
			}
			color = parseColor(plotinfo.styles[p]);
		
			if ( color < 0 )
				color = colors.splice(0,1)[0];		// pick next unused color
			else
				colors.splice(colors.indexOf(color),1); // remove this color
		}
		else 
			color = color = colors.splice(0,1)[0];	// pick next unused color
		
		if ( typeof(color) == "undefined")	// pick black if no next unused color
			color = 0;
	
		for ( i=0; i < plotinfo.data[p].length; i++) {
			if ( pointstyle ) {
				LALOLibPlots[LALOLibPlotsIndex].X.push( plotinfo.data[p][i] );
				LALOLibPlots[LALOLibPlotsIndex].Y.push( color );	
			}
			if ( linestyle && i < plotinfo.data[p].length-1 ) 
				LALOLibPlots[LALOLibPlotsIndex].plot_line(plotinfo.data[p][i], plotinfo.data[p][i+1], "", color);
		}
		
		// Legend
		if ( plotinfo.legend[p] != "" ) {		
			var ctx = document.getElementById("legend" +LALOLibPlotsIndex).getContext("2d");
			setcolor(ctx, color);
			ctx.lineWidth = "3";
			if ( pointstyle ) {
				ctx.beginPath();
				ctx.arc( legendwidth/2 , ylegend, 5, 0, 2 * Math.PI , true);
				ctx.closePath();
				ctx.fill();
			}
			if( linestyle) {
				ctx.beginPath();
				ctx.moveTo ( 0,ylegend);
				ctx.lineTo (legendwidth, ylegend);
				ctx.stroke();
			}
			ylegend += 20;
			
			document.getElementById("legendtxt" +LALOLibPlotsIndex).innerHTML += plotinfo.legend[p] + "<br>";						
		}
	}
	LALOLibPlots[LALOLibPlotsIndex].computeRanges();
	LALOLibPlots[LALOLibPlotsIndex].replot();

	LALOLibPlotsIndex++;
}

// image
function image(X, title) {
	if (type(X) == "vector")  {
		X = mat([X]);
	}
		
	var style;
	var minX = min(X);
	var maxX = max(X);
	var m = X.length;	
	var n = X.n;
	var scale = (maxX - minX) ; 
	
	var i;
	var j;
	var k = 0;
	var data = new Array();
	for ( i=0; i < m; i++) {
		var Xi = X.row(i);
		for ( j=0; j < n; j++) {	// could do for j in X[i] if colormap for 0 is white...
			color =   mul( ( Xi[j] - minX) / scale, ones(3) ) ;
			data[k] = [i/m, j/n, color];
			k++;
		}
	}
	style  = [m,n,minX,maxX];

	var imagedata =  { "data" : data, "style" : style, "title": title };

	////// Part from laloplots.html	
	
	
	var plotid = "LALOLibPlot" + LALOLibPlotsIndex;
	var legendwidth = 50;
	var pixWidth ;
	var pixHeight ;

		// prepare legend
	var ylegend = 20;
	
	// do plot

	
	var i;

	var width = 500; 
	var height = 500; 	

	var title = imagedata.title;
	if(title) {
		LALOLibOutput.innerHTML += "<h3>"+title+"</h3>" + "  ( " + imagedata.style[0] + " by " + imagedata.style[1] + " matrix )";
	}
	
	if ( imagedata.style[1] > width ) {
		width = imagedata.style[1]; 
		plotlegend.style.left = (width+60) +"px";
	}
	if ( imagedata.style[0] > height )
	 	height = imagedata.style[0];
		
	pixWidth = width / imagedata.style[1];
	pixHeight = height / imagedata.style[0];
	
	var legendwidth = 50;
	
	LALOLibOutput.innerHTML += '<div style="position:relative;left:0px;top:0px;text-align:left;"> <div><a onmousemove="mouseimageposition(event,' + LALOLibPlotsIndex + ');"><canvas id="' +plotid + '"  width="' + width + '" height="' + height + '" style="border: 1px solid black;"></canvas></a></div><label id="lblposition' + LALOLibPlotsIndex + '"></label> <div style="position: absolute;left: 550px;top: -1em;">' + imagedata.style[2].toFixed(3) + '<br> <canvas id="legend' + LALOLibPlotsIndex + '" width="' + legendwidth + '" height="500"></canvas> <br>' + imagedata.style[3].toFixed(3) + ' </div>  </div>';

	var x;
	var y;
	var color;
	
	LALOLibPlots[LALOLibPlotsIndex] = imagedata;
	LALOLibPlots[LALOLibPlotsIndex].canvasId = plotid; 
	var canvas = document.getElementById(plotid);
	
  	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");

		for ( i=0; i < imagedata.data.length ; i++) {
			x = canvas.width * LALOLibPlots[LALOLibPlotsIndex].data[i][1];
			y =  canvas.height * LALOLibPlots[LALOLibPlotsIndex].data[i][0] ;
			color = LALOLibPlots[LALOLibPlotsIndex].data[i][2];
		
			ctx.fillStyle = "rgb(" + Math.floor(255*(1-color[0])) + "," + Math.floor(255*(1-color[1])) + "," + Math.floor(255*(1-color[2])) + ")";
			ctx.fillRect( x , y, pixWidth +1,  pixHeight +1); // +1 to avoid blank lines between pixels

		}
	}
	
	// add legend / colormap

	var legend = document.getElementById("legend" +LALOLibPlotsIndex);
	var ctx = legend.getContext("2d");

	for ( i=0; i< 255;i++) {
		y = Math.floor(i * legend.height / 255);
		ctx.fillStyle = "rgb(" + (255-i) + "," + (255-i) + "," + (255-i) + ")";
		ctx.fillRect( 0, y, legendwidth , (legend.height / 255) + 1) ;
	}	
	
	// Prepare mouseposition info
	LALOLibPlots[LALOLibPlotsIndex].pixelWidth = pixWidth; 
	LALOLibPlots[LALOLibPlotsIndex].pixelHeight = pixHeight;
	LALOLibPlotsIndex++;
}



function parseColor( str ) {
	if ( typeof(str) == "undefined") 
		return -1;
		
	var color;
	switch( str ) {
	case "k":
	case "black":
		color = 0;
		break;
	case "blue":
	case "b":
		color = 1;
		break;
	case "r":
	case "red":
		color = 2;
		break;
	case "g":
	case "green":
		color = 3;
		break;
	case "m":
	case "magenta":
		color = 4;
		break;
	case "y":
	case "yellow":
		color = 5;
		break;
	
	default:
		color = -1;
		break;
	}
	return color;
}

function mousezoom ( e, delta , plotidx) {
	if (!e) 
    	e = window.event;
 	
 	e.preventDefault();
	
	if ( typeof(plotidx) == "undefined")
		var plotidx = 0;
	
	if ( typeof(delta) == "undefined") {
		var delta = 0;
		
		// normalize the delta
		if (e.wheelDelta) {
		     // IE and Opera
		    delta = e.wheelDelta / 30;
		} 
		else if (e.detail) { 
		    delta = -e.detail ;
		}
	} 
	else {
		if (e.button != 0 )
			delta *= -1;
	}
		
	var plotcanvas = document.getElementById(LALOLibPlots[plotidx].canvasId);
	var rect = plotcanvas.getBoundingClientRect();
	var x = e.clientX - rect.left;	// mouse coordinates relative to plot
	var y = e.clientY - rect.top;
	LALOLibPlots[plotidx].zoom(1+delta/30,1+delta/30, x, y);	
}
function zoomoriginal(plotidx) {
	LALOLibPlots[plotidx].resetzoom(); 
}
function mouseposition( e , plotidx) {
	var plotcanvas = document.getElementById(LALOLibPlots[plotidx].canvasId);
	var rect = plotcanvas.getBoundingClientRect();

	var xmouse = e.clientX - rect.left;	// mouse coordinates relative to plot
	var ymouse = e.clientY - rect.top;

	if ( LALOLABPLOTMOVING ) {	
		var dx = xmouse - LALOLABPLOTxprev ;
		var dy = ymouse - LALOLABPLOTyprev;
		if ( Math.abs( dx ) > 1 || Math.abs( dy ) > 1 ) {			
			LALOLibPlots[plotidx].translate(dx, dy);
		}
		LALOLABPLOTxprev = xmouse;
		LALOLABPLOTyprev = ymouse;		
	}
	else {		
		var x = xmouse / LALOLibPlots[plotidx].scaleX + LALOLibPlots[plotidx].minX;
		var y = (plotcanvas.height - ymouse ) / LALOLibPlots[plotidx].scaleY + LALOLibPlots[plotidx].minY;
	
		document.getElementById("lblposition" + plotidx).innerHTML = "x = " + x.toFixed(3) + ", y = " + y.toFixed(3);	
	}
}

function mousestartmove( e , plotidx) {
	if ( e.button == 0 ) {
		LALOLABPLOTMOVING = true;
		var plotcanvas = document.getElementById(LALOLibPlots[plotidx].canvasId);
		var rect = plotcanvas.getBoundingClientRect();
		LALOLABPLOTxprev = e.clientX - rect.left;	// mouse coordinates relative to plot
		LALOLABPLOTyprev = e.clientY - rect.top;
	}
	else {
		LALOLABPLOTMOVING = false;
	}
}
function mousestopmove( e ) {
	LALOLABPLOTMOVING = false;
}

function mouseimageposition( e, plotidx ) {
	var plotcanvas = document.getElementById(LALOLibPlots[plotidx].canvasId);
	var rect = plotcanvas.getBoundingClientRect();

	var xmouse = e.clientX - rect.left;	// mouse coordinates relative to plot
	var ymouse = e.clientY - rect.top;

	var n = LALOLibPlots[plotidx].style[1];
	var minX = LALOLibPlots[plotidx].style[2];	
	var maxX = LALOLibPlots[plotidx].style[3];	
	var i = Math.floor(ymouse / LALOLibPlots[plotidx].pixelHeight);
	var j = Math.floor(xmouse / LALOLibPlots[plotidx].pixelWidth );
	if ( j < n ) {
		var val = LALOLibPlots[plotidx].data[i*n + j][2][0]*(maxX - minX) + minX;
	
		document.getElementById("lblposition" + plotidx).innerHTML = "Matrix[ " + i + " ][ " + j + " ] = " + val.toFixed(3);
	}
}
/////////////////////////////////
//// Parser
////////////////////////////////

function lalo( Command ) {
	// Parse command line and execute in current scopes	
	var cmd = laloparse( Command );
	var res = self.eval(cmd); 
	return res; 
}
function laloparse( WorkerCommand ) {
	// Parse Commands
	var WorkerCommandList = WorkerCommand.split("\n");
	var k;
	var cmd = "";
	for (k = 0; k<WorkerCommandList.length; k++) {
		if( WorkerCommandList[k].length > 0 ) {
		  	if ( WorkerCommandList[k].indexOf("{") >= 0 || WorkerCommandList[k].indexOf("}") >= 0) {
		  		// this line includes braces => plain javascript: do not parse it!
		  		cmd += WorkerCommandList[k];
		  		if ( WorkerCommandList[k].indexOf("}") >= 0 ) {
		  			// braces closed, we can end the line
			  		cmd += " ;\n"; 
			  	}				  	
		  	}
		  	else {
		  		// standard lalolab line
		  		cmd += parseCommand(WorkerCommandList[k]) + " ;\n"; 
		  	}
		}
	}
	return cmd; 
}
function parseSplittedCommand( cmd ) {
	//console.log("parsing : " + cmd);
	// !!! XXX should parse unary ops before all the others !!! 
	
	var ops = ["==", "!=", ">=" ,"<=", ">", "<" , "\\" ,":", "+", "-",  ".*", "*", "./" ,  "^", "'"]; // from lowest priority to highest
	var opsFcts = ["isEqual" , "isNotEqual", "isGreaterOrEqual", "isLowerOrEqual", "isGreater" , "isLower", "solve","range", "add", "sub", "entrywisemul", "mul" , "entrywisediv",  "pow", "undefined" ];
	var unaryOpsFcts = ["", "", "", "", "","", "","range","", "minus", "", "" , "",  "", "transpose" ];
	
	var o;
	var i ;
	var k;
	var operandA;
	var operandB;

	for ( o = 0; o < ops.length; o++) {
		
		var splitted_wrt_op = cmd.split(ops[o]);
		
		if ( splitted_wrt_op.length > 1) {			
			if ( removeSpaces(splitted_wrt_op[0]) != "" ) {				
				// there is actually a left-hand side operand
				if( removeSpaces(splitted_wrt_op[1]) != "" ) {
					// and a right-hand side operand
					operandA = parseSplittedCommand(splitted_wrt_op[0]);

					for ( k = 1; k< splitted_wrt_op.length ; k++) {
						operandB = splitted_wrt_op[k];
						operandA =  opsFcts[o] + "(" + operandA +  "," + parseSplittedCommand(operandB) + ")";
					}
					cmd = operandA; 
				}
				else {
					// no right-hand side: like transpose operator
					cmd = unaryOpsFcts[o] + "(" + parseSplittedCommand(splitted_wrt_op[0]) + ")";
				}
			}
			else {
				// no left operand: like minus something...
				
				// Apply unary operator
				operandA = unaryOpsFcts[o] + "(" + parseSplittedCommand(splitted_wrt_op[1]) + ")";
				
				// and then binary operator for the remaining occurences
				for ( k = 2; k< splitted_wrt_op.length ; k++) {
					operandB = splitted_wrt_op[k];
					operandA =  opsFcts[o] + "(" + operandA +  "," + parseSplittedCommand(operandB) + ")";
				}
				cmd = operandA; 
			}
		}
	}
	
	return cmd;
	
}

function parseAssignment ( assignmentStr ) {
	if ( assignmentStr.indexOf("[") < 0 ) {
		// straightforward assignment 
		return assignmentStr; 
	}
	else {
		var assign = removeSpaces(assignmentStr).replace("=","").replace(",","][");
		var middle = assign.indexOf("][");
		var start = assign.indexOf("[");
		var varname = assign.substr(0,start);
		if ( middle >= 0 ) {
			// submatrix assignment
			var rowsrange = assign.substr( start + 1, middle-start-1); 

			// find last "]";
			var end = middle+1;
			while ( assign.indexOf("]",end+1) >= 0)
				end = assign.indexOf("]",end+1);
			
			var colsrange = assign.substr(middle+2, end - (middle+2)); // everything after "]["	and before last "]"	

			// Parse colon ranges
			var rowssplit = rowsrange.split(":");
			if (rowssplit.length == 2 ){
				if ( rowssplit[0] =="" && rowssplit[1] =="" )
					rowsrange = "[]";
				else
					rowsrange = "range(" + rowssplit[0] + "," + rowssplit[1] + ")";
			}
			else if ( rowssplit.length == 3)
				rowsrange = "range(" + rowssplit[0] + "," + rowssplit[2] + "," + rowssplit[1] + ")";
			
			var colssplit = colsrange.split(":");
			if (colssplit.length == 2 ) {
				if ( colssplit[0] =="" && colssplit[1] =="" )
					colsrange = "[]";
				else
					colsrange = "range(" + colssplit[0] + "," + colssplit[1] + ")";
			}
			else if ( colssplit.length == 3)
				colsrange = "range(" + colssplit[0] + "," + colssplit[2] + "," + colssplit[1] + ")";

			return "set( " + varname + "," + rowsrange + "," + colsrange + ", ";
		}
		else {
			// subvector assignment
			
			// find last "]";
			var end = start;
			while ( assign.indexOf("]",end+1) >= 0)
				end = assign.indexOf("]",end+1);
			
			var rowsrange = assign.substr( start + 1, end-start-1); 
			
			// Parse colon ranges
			var rowssplit = rowsrange.split(":");
			if (rowssplit.length == 2 ){
				if ( rowssplit[0] =="" && rowssplit[1] =="" )
					rowsrange = "[]";
				else
					rowsrange = "range(" + rowssplit[0] + "," + rowssplit[1] + ")";
			}
			else if ( rowssplit.length == 3)
				rowsrange = "range(" + rowssplit[0] + "," + rowssplit[2] + "," + rowssplit[1] + ")";

			return "set( " + varname + "," + rowsrange + ", ";
		}
	}
}

function parseBrackets( cmdString ) {
	// Parse brackets => get matrix entries
	
	var delimiters = ["[", "(",",",";",")", "\\", "+", "-", "*", "/", ":", "^", "'", "=", ">", "<", "!"];
	
	cmdString = cmdString.split("][").join(","); // replace ][ by , and ] by )
	
	var cmd = cmdString.split("");	// string to array of char
	
	var i; 
	var j;
	var k;
	var l;
	var lhs;
	
	// For the entire string:	
	i = cmd.length - 1;
	while ( i >= 0 ) {
		// Search for the right-most opening bracket:
		while ( i >= 0 && cmd[i] != "[" ) 
			i--;
		
		if ( i >= 0 ) {
			// found a bracket,  find its corresponding closing bracket
			j = i+1;
			while ( j < cmd.length && cmd[j] != "]" ) 
				j++;

			if ( j < cmd.length ) {		

				// then determine its left-hand side operand:
				l = 0;
				k = 0;
				while ( k < i ) {
					if ( delimiters.indexOf(cmd[k]) >= 0)
						l = k+1;
					k++;
				}
				lhs = cmd.slice(l,i).join(""); // should be LHS as string or "" if l >= i

				if ( removeSpaces(lhs) == "" ) {
					// if the LHS operand is empty, leave the brackets untouched 
					cmd[i] = "#"; // (replace by # and $ re-replace at the end by a matrix creation)
					
					// look for semicolon within brackets: 
					k = i+1; 
					var rowwise = false; 
					var colwise = false; 
					while (  k < j ) {
						if( cmd[k] == "," ) {
							//colwise = true;
						}
						
						if ( cmd[k] == ";" ) {
							rowwise = true; // mark for rowwise mat
							
							if ( colwise ) {
								cmd.splice(k,1, ["@", ","] ); // end previous row vector, replace by comma  
								colwise = false;
							}
							else {
								cmd[k] = ","; // simply replace by comma
							}
						}
						
						
						k++; 
					} 
					
					if ( rowwise ) 
						cmd[j] = "$";
					else
						cmd[j] = "@";
					
				}
				else {						
					// if not empty, implement a GET
					cmd[l]="get(" + lhs ;
					for ( k = l+1; k < i; k++)
						cmd[k] = "";
					cmd[i] = ",";
					cmd[j] = ")";					
				}
			}
			else {
				return undefined; // error no ending bracket;
			}
		}
		i--;
	}
		
	var cmdparsed = cmd.join("").split("#").join("mat([").split("$").join("], true)").split("@").join("])");
	//console.log(cmdparsed);
	return cmdparsed;
}

function parseCommand( cmdString ) {

	// Remove comments at the end of the line
	var idxComments = cmdString.indexOf("//");
	if ( idxComments >= 0 )
		cmdString = cmdString.substr(0,idxComments);
	

	// Parse "=" sign to divide between assignement String and computeString
	var idxEqual = cmdString.split("==")[0].split("!=")[0].split(">=")[0].split("<=")[0].indexOf("=");
	if ( idxEqual > 0 )  {
		var assignmentStr = parseAssignment( cmdString.substr(0,idxEqual + 1) );
		var computeStr = cmdString.substr(idxEqual+1);
		
		// Check for simple assignments like A = B to force copy
		if ( assignmentStr.indexOf("set(") < 0 && typeof(self[removeSpaces(computeStr)]) != "undefined" ) { //self.hasOwnProperty( removeSpaces(computeStr) ) ) { // self.hasOwnProperty does not work in Safari workers....
		
			// computeStr is a varaible name
			if ( !isScalar(self[ removeSpaces(computeStr) ] ) ) { 
				// the variable is a vector or matrix
				var FinalCommand = assignmentStr + "matrixCopy(" + computeStr + ")";
				console.log(FinalCommand);
				return FinalCommand;
			}
		}		
	}
	else {
		var assignmentStr = "";		
		var computeStr = cmdString;
	}
	
	// parse brackets:
	var cmd =  parseBrackets( computeStr ).split(""); // and convert string to Array

	// Parse delimiters 
	var startdelimiters = ["(","[",",",";"];
	var enddelimiters = [")","]",",",";"];
	var i;
	var j;
	var k;
	var parsedContent = "";
	var parsedCommand = new Array(cmd.length);

	var map = new Array(cmd.length ) ;
	for ( k=0;k<cmd.length;k++) {
		map[k] = k;
		parsedCommand[k] = cmd[k];
	}
	
	i = cmd.length - 1; 
	while ( i >= 0 ) {
		// Find the most right starting delimiter
		while ( i >= 0 && startdelimiters.indexOf(cmd[i]) < 0 )
			i--;
		if ( i >= 0 ) {
			// found a delimiter, search for the closest ending delimiter
			j = i+1;
			while ( j < cmd.length && enddelimiters.indexOf(cmd[j] ) < 0 ) {				
				j++;
			}
			if ( j < cmd.length ) {			
				// starting delimiter is at cmd[i] and ending one at cmd[j]
				
				// parse content within delimiters
				parsedContent = parseSplittedCommand( parsedCommand.slice(map[i]+1,map[j]).join("") ) ;
				// and replace the corresponding content in the parsed command
				parsedCommand.splice (map[i]+1, map[j]-map[i]-1, parsedContent ) ;
				
				// remove delimiters from string to be parsed 
				if ( cmd[i] != "," ) 
					cmd[i] = " ";	// except for commas that serve twice (once as start once as end)
				cmd[j] = " ";
								
				// map position in the original cmd to positions in the parsedCommand to track brackets
				for ( k=i+1; k < j;k++)
					map[k] = map[i]+1;
				var deltamap = map[j] - map[i] - 1;
				for ( k=j; k < cmd.length;k++)
					map[k] += 1 - deltamap; 
					
				/*console.log(parsedCommand);
				console.log(cmd.join(""));
				console.log(map);
				console.log(i + " : " + j);*/
			}
			else {
				return "undefined";
			}				
		}
		i--;
	}
	var FinalCommand = assignmentStr + parseSplittedCommand(parsedCommand.join(""));
	
	// Parse brackets => get matrix entries
	//cmdString = cmdString.split("][").join(",").split("]").join(")");	// replace ][ by , and ] by )
	// consider [ as a left-hand unary operator 
//	cmd = "get(" + parseSplittedCommand(splitted_wrt_op[0]) + ")";

	
	
	if ( assignmentStr.substr(0,4) == "set(" ) 
		FinalCommand  += " )";

	FinalCommand = parseRangeRange(	FinalCommand );

	console.log(FinalCommand);
	return FinalCommand;
}

function parseRangeRange( cmd ) {
	// parse complex ranges like 0:0.1:4
	var elems = cmd.split("range(range(");
	var i;
	var j;
	var tmp;
	var args;
	var incargs;
	var endargs;
	for ( i = 0; i< elems.length - 1 ; i++) {
	
//		elems[i+1] = elems[i+1].replace(")","");	
		
		// ivert second and third arguments to get range(start, end, inc) from start:inc:end
		args = 	elems[i+1].split(",");
		tmp = args[2].split(")"); // keep only the content of the range and not the remaining commands
		endargs = tmp[0];
		j = 0;	// deal with cases like end="minus(4)" where the first closing bracket is not at the complete end
		while ( tmp[j].indexOf("(") >= 0 ) {
			endargs = endargs + ")" + tmp[j+1]; 
			j++;
		}
			
		incargs = args[1].substr(0,args[1].length-1); // remove ")" 
		args[1] = endargs;
		//endargs[0] = incargs;
		args[2] = incargs + ")" + tmp.slice(j+1).join(")");
		elems[i+1] = args.join(",");
	}
	return elems.join("range(");//replace range(range( by range(
}

function removeSpaces( str ) {
	return str.split(" ").join("");
}

////////////////////////////
/// Lab 
////////////////////////////
function MLlab ( id , path ) {
	var that = new Lalolab ( id, true, path);	
	return that;
}
function Lalolab ( id, mllab , path ) {
	// constructor for a Lab with independent scope running in a worker
	this.id = id;
	
	this.callbacks = new Array(); 	
	
	// Create worker with a Blob  to avoid distributing lalolibworker.js 
	// => does not work due to importScripts with relative path to the Blob unresolved (or cross-origin)
	
	if ( typeof(path) == "undefined" )
		var path = "http://mlweb.loria.fr/";
	else {
		if (path.length > 0 && path[path.length-1] != "/" )
			path = [path,"/"].join("");
	}
		
	if ( typeof(mllab) != "undefined" && mllab ) {
		this.worker = new Worker(path+"mlworker.js"); // need mlworker.js in same directory as web page
		this.labtype = "ml";
		/* Using a Blob to avoid distributing mlworker.js: 
		 	does not work because of importScripts from cross origin...
		var workerscript = "importScripts(\"ml.js\");\n onmessage = function ( WorkerEvent ) {\n	var WorkerCommand = WorkerEvent.data.cmd;var mustparse = WorkerEvent.data.parse; \n if ( mustparse )\n	var res = lalo(WorkerCommand);\n 	else {\n	if ( WorkerCommand == \"load_mat\" ) {\n	if ( type(WorkerEvent.data.data) == \"matrix\" )\n var res = new Matrix(WorkerEvent.data.data.m,WorkerEvent.data.data.n,WorkerEvent.data.data.val, true);\nelse\n 	var res = mat(WorkerEvent.data.data, true);\n	eval(WorkerEvent.data.varname + \"=res\");\n}\n else\n var res = self.eval( WorkerCommand ) ;\n}\n try {\n	postMessage( { \"cmd\" : WorkerCommand, \"output\" : res } );\n} catch ( e ) {\n try {\n postMessage( { \"cmd\" : WorkerCommand, \"output\" : res.info() } );\n	} catch(e2) { \n postMessage( { \"cmd\" : WorkerCommand, \"output\" : undefined } );\n}\n}\n}";
		var blob = new Blob([workerscript], { "type" : "text/javascript" });
		var blobURL = window.URL.createObjectURL(blob);
		console.log(blobURL);
		this.worker = new Worker(blobURL);*/
	}
	else {
		this.worker = new Worker(path+"lalolibworker.js"); // need lalolibworker.js in same directory as web page
		this.labtype = "lalo";
	}
	this.worker.onmessage = this.onresult; 
	this.worker.parent = this;
}
Lalolab.prototype.close = function ( ) {
	this.worker.terminate();
	this.worker.parent = null;// delete circular reference
}
Lalolab.prototype.onprogress = function ( ratio ) {
	// do nothing by default; 
	// user must set lab.onprogress = function (ratio) { ... } to do something
}
Lalolab.prototype.onresult = function ( WorkerEvent ) {
//	console.log(WorkerEvent, ""+ this.parent.callbacks);
	if ( typeof(WorkerEvent.data.progress) != "undefined" ) {
		this.parent.onprogress( WorkerEvent.data.progress ) ;
	}
	else {
		var cb =  this.parent.callbacks.splice(0,1)[0] ; // take first callback from the list
		if ( typeof(cb) == "function" ) {
			var WorkerCommand = WorkerEvent.data.cmd;
			var WorkerResult = WorkerEvent.data.output;
			cb(	WorkerResult, WorkerCommand, this.parent.id ); // call the callback if present
		}
	}
}
Lalolab.prototype.do = function ( cmd , callback ) {
	// prepare callback, parse cmd and execute in worker
	this.callbacks.push(  callback  ) ;	
	this.worker.postMessage( {cmd: cmd, parse: true} );	 
}
Lalolab.prototype.exec = function ( cmd , callback ) {
	// prepare callback, parse cmd and execute in worker
	this.callbacks.push( callback ); 
	this.worker.postMessage( {cmd: cmd, parse: false} );	
}
Lalolab.prototype.parse = function ( cmd , callback ) {
	// prepare callback, parse cmd and execute in worker
	this.callbacks.push( callback ); 
	this.worker.postMessage( {cmd: cmd, parse: false} );	 
}
Lalolab.prototype.load = function ( data , varname, callback ) {
	// load data in varname
	this.callbacks.push(  callback  ) ;	
	if ( typeof(data) == "string" ){
		this.worker.postMessage( {"cmd" : varname + "= load_data (\"" + data + "\")", parse: false} );
	}
	else {
		this.worker.postMessage( {"cmd" : "load_mat", data: data, varname: varname, parse: false} );
	}			
}
Lalolab.prototype.import = function ( script, callback ) {
	// load a script in lalolib language
	this.do('importLaloScript("' + script + '")', callback);	
}
function importLaloScript ( script ) {
	// load a script in lalolib language in the current Lab worker
	var xhr = new XMLHttpRequest();
	xhr.open('GET', script, false);
	xhr.send();
	var cmd = xhr.responseText;
 	return lalo(cmd); 
}
Lalolab.prototype.importjs = function ( script, callback ) {
	// load a script in javascript
	this.exec("importScripts('" + script + "');", callback); 
}
Lalolab.prototype.getObject = function ( varname, callback ) {
	this.exec("getObjectWithoutFunc(" + varname +")", function (res) {callback(renewObject(res));} );
}

function getObjectWithoutFunc( obj ) {
	// Functions and Objects with function members cannot be sent 
	// from one worker to another...
	
	if ( typeof(obj) != "object" ) 
		return obj;
	else {
		var res = {};

		for (var p in obj ) {
			switch( type(obj[p]) ) {
			case "vector": 
				res[p] = {type: "vector", data: [].slice.call(obj[p])};
				break;
			case "matrix":
				res[p] = obj[p];
				res[p].val = [].slice.call(obj[p].val);
				break;
			case "spvector":
				res[p] = obj[p];
				res[p].val = [].slice.call(obj[p].val);
				res[p].ind = [].slice.call(obj[p].ind);
				break;
			case "spmatrix":
				res[p] = obj[p];
				res[p].val = [].slice.call(obj[p].val);
				res[p].cols = [].slice.call(obj[p].cols);
				res[p].rows = [].slice.call(obj[p].rows);
				break;
			case "undefined":
				res[p] = obj[p];
				break;
			case "function":
				break;
			case "Array":
				res[p] = getObjectWithoutFunc( obj[p] );
				res[p].type = "Array";
				res[p].length = obj[p].length;
				break;
			default:
				res[p] = getObjectWithoutFunc( obj[p] );
				break;			
			}	
		}
		return res;
	}
}
function renewObject( obj ) {
	// Recreate full object with member functions 
	// from an object created by getObjectWithoutFunc()

	var to = type(obj);
	switch( to ) {
		case "number":
		case "boolean":
		case "string":
		case "undefined":
			return obj;
			break;
		case "vector":
			return new Float64Array(obj.data);
			break;
		case "matrix":
			return new Matrix(obj.m, obj.n, obj.val);
			break;
		case "spvector":
			return new spVector(obj.length,obj.val,obj.ind);
			break;
		case "spmatrix":
			return new spMatrix(obj.m, obj.n, obj.val, obj.cols, obj.rows);
			break;
		case "object":
			// Object without type property and thus without Class		
			var newobj = {}; 
			for ( var p in obj ) 
				newobj[p] = renewObject(obj[p]);
			return newobj;
			break;
		case "Array":
			var newobj = new Array(obj.length);
			for ( var p in obj ) 
				newobj[p] = renewObject(obj[p]);
			return newobj;
		default:
			// Structured Object like Classifier etc... 
			// type = Class:subclass
			var typearray = obj.type.split(":");
			var Class = eval(typearray[0]);
			if ( typearray.length == 1 ) 
				var newobj = new Class(); 
			else 
				var newobj = new Class(typearray[1]);
			for ( var p in obj ) 
				newobj[p] = renewObject(obj[p]);
				
			// deal with particular cases: 
			// Rebuild kernelFunc 
			if (typearray[1] == "SVM" || typearray[1] == "SVR" ) {				
				newobj["kernelFunc"] = kernelFunction(newobj["kernel"], newobj["kernelpar"], type(newobj["SV"]) == "spmatrix"?"spvector":"vector");
			}
			if (typearray[1] == "KernelRidgeRegression" ) {
				newobj["kernelFunc"] = kernelFunction(newobj["kernel"], newobj["kernelpar"], type(newobj["X"]) == "spmatrix"?"spvector":"vector");
			}
			
			return newobj;
			break;
	}
}

function load_data ( datastring ) {

	// convert a string into a matrix data 
	var i;
	var cmd = "mat( [ "; 
	var row;
	var rows = datastring.split("\n");
	var ri ;
	for ( i=0; i< rows.length - 1; i++) {
		ri = removeFirstSpaces(rows[i]);
		if ( ri != "" ) {
			row = ri.replace(/,/g," ").replace(/ +/g,",");
			cmd += "new Float64Array([" + row + "]) ,";
		}
	}
	ri = removeFirstSpaces(rows[rows.length-1]);
	if ( ri != "" ) {
		row = ri.replace(/,/g," ").replace(/ +/g,",");
		cmd += "new Float64Array([" + row + "]) ] , true) ";
	}
	else {
		cmd = cmd.substr(0,cmd.length-1); // remove last comma
		cmd += "] , true) ";
	}
		
	return eval(cmd);
	
}

function removeFirstSpaces( str ) {
	//remove spaces at begining of string
	var i = 0;
	while ( i < str.length && str[i] == " " )
		i++;
	if ( i<str.length ) {
		// first non-space char at i
		return str.slice(i);	
	}
	else 
		return "";
}

//// progress /////////////////////
function notifyProgress( ratio ) {
	postMessage( { "progress" : ratio } );
	console.log("progress: " + ratio);
}



