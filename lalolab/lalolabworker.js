/*
	Main Worker for LAlOLab
*/

//////////////////////////
//// Cross-browser compatibility
///////////////////////////

if(typeof(self.console) != "object" ) {
	console = {log: function ( ) { } };	
}
	

if( typeof(Math.sign) == "undefined" ) {
	// for IE, Safari
	Math.sign = function ( x ) { return ( x>=0 ? (x==0 ? 0 : 1) : -1 ) ;}
}

//////////////////////////////
//// Load LALOLib
/////////////////////////////
importScripts('laloliblab.js');

//importScripts('ASyncClassifier.js');


///////////////////////////////
/// Worker interface
///////////////////////////////

var LALOLABLastResult ; 

onmessage = function ( WorkerEvent ) {
		
	var WorkerCommand = WorkerEvent.data.cmd;
	
	if ( WorkerCommand == "LOAD_WORKSPACE" ) {
		// Set Workspace:
		var info = setWorkspace ( WorkerEvent.data.data );
		postMessage( { "cmd" : "Loading workspace..." , "output" : info } );
	}
	else if ( WorkerCommand.indexOf( "LOAD_DATA_FILE" ) >= 0 ) {		
		var WorkerOutput = load_data_to_data( WorkerEvent.data.data );		
		var WorkerOutputSize = size(WorkerOutput);	
		postMessage( { "cmd" : WorkerCommand , "output" : WorkerOutput, "size" : WorkerOutputSize } );
	}
	else if ( WorkerCommand.indexOf( "NEW_MATRIX" ) == 0 ) {		
		data = new Matrix(WorkerEvent.data.m,WorkerEvent.data.n,WorkerEvent.data.values);
		var WorkerOutput = data;
		var WorkerOutputSize = size(WorkerOutput);	
		postMessage( { "cmd" : "data=" , "output" : WorkerOutput, "size" : WorkerOutputSize } );
	}
	else {
		// Parse Commands
		var WorkerCommandList = WorkerCommand.split("\n");
		var k;
		var cmd = "";
		for (k = 0; k<WorkerCommandList.length; k++) {
			if( WorkerCommandList[k].length > 0 ) {
			  	if ( WorkerCommandList[k].indexOf("{") >= 0 || WorkerCommandList[k].indexOf("}") >= 0) {
			  		// this line includes braces => plain javascript: do not parse it!
			  		cmd += WorkerCommandList[k];
			  		if ( removeSpaces( WorkerCommandList[k] ).indexOf("}") > 0 ) {
		  				// braces closed on same line, probably an object parameter => we can end the line 
			  			cmd += " ;\n"; 
				  	}				  	
				  	else { 
				  		// either only a closing braces or a braces left open, just add a newline
				  		cmd += "\n";
				  	}
			  	}
			  	else {
			  		// standard lalolab line
			  		cmd += parseCommand(WorkerCommandList[k]) + " ;\n"; 
			  	}
			}
		}

		// Execute command
		var WorkerOutput = self.eval(cmd); 		
		var WorkerOutputSize = size(WorkerOutput);

		LALOLABLastResult = WorkerOutput;
		
		if ( WorkerOutput != "LALOLABPLOT") {
			// return results to be printed
			if ( WorkerEvent.data.hidecmd ) 
				postMessage( { "cmd" : "", "output" : WorkerOutput, "size" : WorkerOutputSize } );					
			else
				postMessage( { "cmd" : WorkerCommand, "output" : WorkerOutput, "size" : WorkerOutputSize } );	
		}
	}
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
		if ( assignmentStr.indexOf("set(") < 0 && typeof(self[removeSpaces(computeStr)]) != "undefined" ) { //self.hasOwnProperty( removeSpaces(computeStr) ) ) {
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

///////////////////////////////
/// Additional operators and useful stuff
///////////////////////////////

// JScode
function JScode( args ) {
	// args = { script: "...", html: true/false, ... }
	var WorkerCommand = args.script;
	var WorkerCommandList = WorkerCommand.split("\n");

	var k;
	var cmd = "";
	
	if ( args.html ) {
		// HTML Preambule
		cmd += htmlstr('<html>\n<head>\n'); 
		cmd += htmlstr('<meta charset="UTF-8">\n');		
		// cmd += htmlstr('<script type="application/x-javascript" src="glpk.js"> </script> \n'); //now in lalolib.js
		// cmd += htmlstr('<script type="application/x-javascript" src="lalolib.js"> </script> \n');
		cmd += htmlstr('<script src="http://mlweb.loria.fr/ml.js"> </script> \n');
		cmd += htmlstr('<script>\nfunction myLALOLabScript() {\n\n');

	}
	
	// Parse script
	for (k = 0; k<WorkerCommandList.length; k++) {
		if( WorkerCommandList[k].length > 0 ) {
		  	if ( WorkerCommandList[k].indexOf("{") >= 0 || WorkerCommandList[k].indexOf("}") >= 0) {
		  		// this line includes braces => plain javascript: do not parse it!
		  		cmd += WorkerCommandList[k] ;
		  		if ( removeSpaces( WorkerCommandList[k]).indexOf("}") > 0 ) {
		  			// braces closed on same line, probably an object parameter => we can end the line 
			  		cmd += " ;<br>"; 
			  	}				  	
			  	else { 
			  		// either only a closing braces or a braces left open, just add a newline
			  		cmd += "<br>";
			  	}
		  	}
		  	else {
		  		// standard lalolab line
		  		cmd += parseCommand(WorkerCommandList[k]) + " ;<br>"; 
		  	}
		}
	}
	
	// Correct print() -> laloprint()
	cmd = cmd.replace(/print\(/g,"laloprint(");
	
	if ( args.html ) {
		cmd += htmlstr('\n}\n</script>\n</head>\n');
		cmd += htmlstr('\n<body onload="myLALOLabScript();">\n');
		cmd += htmlstr("<div style='font-size: 80%;' title='LALOLib output' id='LALOLibOutput'></div>\n");
		cmd += htmlstr('</body>\n</html>\n');
	}
	
	return cmd; 
}
function htmlstr( str ) {
	return str.split("&").join("&amp;").split("<").join("&lt;").split(">").join("&gt;").split("\n").join("<br>");
}

//// Error handling

function error( msg ) {
	throw new Error ( msg ) ;	
//	postMessage( {"error": msg} );
}

/////// plot

function plot(multiargs) {
	// plot(x,y,"style", x2,y2,"style",y3,"style",... )

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
	var legend;
	var i;
	var n;
	var c = 0; // index of current curve
	while ( p < arguments.length)  {
	
		if ( type( arguments[p] ) == "vector" ) {

			if ( p + 1 < arguments.length && type ( arguments[p+1] ) == "vector" ) {
				// classic (x,y) arguments
				x = dense(arguments[p]);
				y = dense(arguments[p+1]);
				
				p++;
			}
			else {
				// only y provided => x = 0:n
				y = dense(arguments[p]);
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

	postMessage( { "cmd" : "plot() opened in new window", "output" :  {"data" : data, "minX" : minX, "maxX" : maxX, "minY" : minY, "maxY": maxY, "styles" : styles, "legend": legends }} );			

	return "LALOLABPLOT";
	
}

/////// color plot

function colorplot(multiargs) {
	// colorplot(x,y,z) or colorplot(X) or colorplot(..., "cmapname" )

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

	var title = undefined;
	
	var t0 =  type( arguments[0] );
	if ( t0 == "matrix" && arguments[0].n == 3 ) {
		x = getCols(arguments[0], [0]);
		y = getCols(arguments[0], [1]);
		z = getCols(arguments[0], [2]);
		
		if (arguments.length == 2 && typeof(arguments[1]) == "string")
			title = arguments[1];
	}
	else if ( t0 == "matrix" && arguments[0].n == 2 && type(arguments[1]) == "vector" ) {
		x = getCols(arguments[0], [0]);
		y = getCols(arguments[0], [1]);
		z = arguments[1];
		if (arguments.length == 3 && typeof(arguments[2]) == "string")
			title = arguments[2];
	}
	else if (t0 == "vector" && type(arguments[1]) == "vector" && type(arguments[2]) == "vector") {
		x = arguments[0];
		y = arguments[1];
		z = arguments[2];
		
		if (arguments.length == 4 && typeof(arguments[3]) == "string")
			title = arguments[3];
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

	postMessage( { "cmd" : "colorplot() opened in new window", "output" :  {"x" : x, "y": y, "z": z, "minX" : minX, "maxX" : maxX, "minY" : minY, "maxY": maxY,  "minZ" : minZ, "maxZ" : maxZ , "title": title}} );			

	return "LALOLABPLOT";
	
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
	var legend;
	var i;
	var n;
	var c = 0; // index of current curve
	while ( p < arguments.length)  {
	
		if ( type( arguments[p] ) == "vector" ) {

			if ( p + 2 < arguments.length && type ( arguments[p+1] ) == "vector" && type ( arguments[p+2] ) == "vector" ) {
				// classic (x,y,z) arguments
				x = dense(arguments[p]);
				y = dense(arguments[p+1]);
				z = dense(arguments[p+2]);				
				
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
			
	postMessage( { "cmd" : "plot3() opened in new window", "output" :  {"data" : data, "styles" : styles, "legend": legends }} );			

	return "LALOLABPLOT";
	
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
		var Xi = getrowref(X, i);
		for ( j=0; j < n; j++) {	// could do for j in X[i] if colormap for 0 is white...
			//color = dense( mul( ( get(X, i, j) - minX) / scale, ones(3) ) );
			color =   mul( ( Xi[j] - minX) / scale, ones(3) ) ;
			data[k] = [i/m, j/n, color];
			k++;
		}
	}
	style  = [m,n,minX,maxX];

	postMessage( { "cmd" : "image() opened in new window", "output" :  {"data" : data, "style" : style, "title": title }} );			

	return "LALOLABPLOT";
	
}
function colormap(x, y, f, title) {
	
	var style;
	var m = x.length;	
	var n = y.length;
	
	var i;
	var j;
	var k = 0;
	var data = new Array();
	var F = zeros(m,n);
	for ( i=0; i < m; i++) {
		for ( j=0; j < n; j++) {	
			F.val[i*n+j] = f([x[i],y[j]]); 			
		}	
	}
	var minf = min(F); 
	var maxf = max(F); 
	var scale = maxf - minf;
	 
	for ( i=0; i < m; i++) {
		for ( j=0; j < n; j++) {	
			var v = F.val[i*n+j];
			if ( isZero(v) )
				color = [255,255,255,0];
			else if ( v > 0 ) 
				color = [255,0,0, 2*v/scale] ; 			
			else
				color = [0,0,255, -2*v/scale] ; 

			data[k] = [i/m, j/n, color];
			k++;
		}
	}
	style  = [m,n,minf,maxf];

	postMessage( { "cmd" : "colormap() opened in new window", "output" :  {"data" : data, "style" : style, "title": title} } );			

	return "LALOLABPLOT";
	
}


function tex( x ) {
	switch ( type( x ) ) {
	case "number":
		return "$" + x + "$";
		break;
	case "vector":
		var res = "$$<br> \\begin{bmatrix}";
		var i;
		for ( i=0; i< x.length-1; i++)
			res += x[i] + " \\\\";
		res += x[i] + " \\end{bmatrix}<br> $$";
		return res;
		break;
	case "matrix":
		var res = "$$<br> \\begin{bmatrix}";
		var i;
		var j;
		for (i=0; i< x.length-1; i++) {			
			for( j=0;j<x.n-1; j++)
				res += x[i][j] + " & ";
			res += x[i][j] + " \\\\";
		}
		for( j=0;j<x.n-1; j++)
			res += x[i][j] + " & ";
		res += x[i][j] + " \\end{bmatrix}<br>$$";
		return res;
		break;
	default: 
		return "undefined";
	}
}

function who() {
	// List all variables
	var res = "Infinity (CONSTANT)<br>";
	var throwaway = ["MathFunctions" , "mf" , "self" , "console" , "location" , "onerror" , "onoffline" , "ononline" , "navigator" , "onclose" , "performance", "crypto", "indexedDB", "PERSISTENT", "TEMPORARY", "caches"]; 
	for ( var i in self ) {
		if ( typeof( self[i] ) != "function" && i.indexOf("GLP") < 0 && i.indexOf("LPX") < 0 && i.indexOf("LPF_") < 0 && i.indexOf("webkit") < 0) {
			if ( MathFunctions.indexOf(i) >= 0 ) 
				res += i + " (CONSTANT)<br> ";
			else if ( throwaway.indexOf(i) < 0 ) 
				res += i + " (VARIABLE)<br> ";
		}
	}
	return res;
}

function help() {
	// List all functions
	var res = "<strong>Available functions:</strong><br><br> ";
	var throwaway = ["postMessage" , "onmessage" , "close", "importScripts" , "dump" , "btoa" , "atob" , "setTimeout" , "clearTimeout" , "setInterval" , "clearInterval", "addEventListener", "removeEventListener" , "dispatchEvent" ]; 
	for ( var i in self ) {
		if ( typeof( self[i] ) == "function") {
			if ( MathFunctions.indexOf(i) >= 0 ) 
				res += i + " <br> ";
			else if ( throwaway.indexOf(i) < 0 ) 
				res += i + " <br> ";
		}
	}
	return res;
}


function print( variable , variablename) {
	if ( arguments.length == 1)
		var cmdstr = "print()";
	else
		var cmdstr = variablename + "=";
	if ( typeof(variable) == "object" && typeof(variable.info) == "function")
		postMessage( { "cmd" : cmdstr , "output" : variable.info() } );	
	else {
		postMessage( { "cmd" : cmdstr , "output" : variable, "size" : size(variable) } );	
		LALOLABLastResult = variable;
	}
	return "";
}


//// progress /////////////////////
function notifyProgress( ratio ) {
	postMessage( { "progress" : ratio } );
}

/////////////////////////////
//// Files 
//////////////////////////
function loadURLasync ( url ) {

	// load a remote file (from same web server only...)
	var xhr = new XMLHttpRequest();
	xhr.open('GET', url, true);
	xhr.responseType = 'blob';

	xhr.onload = function(e) {

	  var blob = new Blob([this.response]);
	  var reader = new FileReaderSync();
	  load_data_to_data(reader.readAsText(blob) );			  
	};

	xhr.send();
	
}
// for MNIST demo 
function loadMNIST ( url ) {
/*
m = loadMNIST("examples/")
knn = new Classifier(KNN, {K: 1})
knn.train(m.Xtrain, m.Ytrain)
knn.predict(m.Xtest[0:10,:] )


X = sparse(m.Xtest, true)
Y = m.Ytest
svm = new Classifier(SVM, {kernel: "rbf", kernelpar: 1000, normalization: "none"})
svm.train(X,Y)
*/
	// load a remote file (from same web server only...)
	var xhr = new XMLHttpRequest();
	xhr.open('GET', url + "mnist.train.images" , false); // false = synchronous
	xhr.responseType = 'blob';
	xhr.send(null);
	
	var blob = new Blob([xhr.response]);
  	var reader = new FileReaderSync();
  	var Xtrain = load_mnist_images(reader.readAsArrayBuffer(blob) , 60000);	
  	
  	xhr.open('GET', url + "mnist.test.images", false); // false = synchronous
	xhr.responseType = 'blob';
	xhr.send(null);
	
	blob = new Blob([xhr.response]);
  	reader = new FileReaderSync();
  	var Xtest = load_mnist_images(reader.readAsArrayBuffer(blob) , 10000);	
  	
  	xhr.open('GET', url + "mnist.train.labels", false); // false = synchronous
	xhr.responseType = 'blob';
	xhr.send(null);
	
	blob = new Blob([xhr.response]);
  	reader = new FileReaderSync();
  	var Ytrain = load_mnist_labels(reader.readAsArrayBuffer(blob) , 60000);	
  	
  	xhr.open('GET', url + "mnist.test.labels", false); // false = synchronous
	xhr.responseType = 'blob';
	xhr.send(null);
	
	blob = new Blob([xhr.response]);
  	reader = new FileReaderSync();
  	var Ytest = load_mnist_labels(reader.readAsArrayBuffer(blob) , 10000);	
  	
  	return {Xtrain: Xtrain, Xtest: Xtest, Ytrain: Ytrain, Ytest: Ytest};
}
function load_mnist_images( ab , N) {
	var bytes = new Uint8Array(ab.slice(16) ); 	// read as Ubytes and skip header
	return new Matrix(N, 28*28, bytes); //, true);
	/*
	var floats = new Float64Array(bytes);
	
	var i;
	var dim = 28*28;
	var mnist = new Array(N); 
	var p = 0; 
	for ( i=0; i < N; i++) {
		mnist[i] = floats.subarray( p, p + dim);
		p += dim;			
	}
	return mnist;
	*/
}
function load_mnist_labels( ab , N) {

	var bytes = new Uint8Array(ab.slice(8) ); 	// read as Ubytes and skip header
	return new Float64Array( bytes.subarray(0, N) );
}
function showmnist( x ) {
	
	var p = 0;
	var I = new Matrix(28,28,x,true);
	image(I);	
}

function loadURL ( url ) {

	// load a remote file (from same web server only...)
	var xhr = new XMLHttpRequest();
	xhr.open('GET', url, false); // false = synchronous
	xhr.responseType = 'blob';
	xhr.send(null);
	
	var blob = new Blob([xhr.response]);
  	var reader = new FileReaderSync();
  	return load_data(reader.readAsText(blob) );			
}


function removeFirstSpaces( str ) {
	//remove spaces at begining of string
	var i = 0;
	while ( i < str.length && str[i] == " " )
		i++;
	if ( i<str.length ) {
		// first tnon-space char at i
		return str.slice(i);	
	}
	else 
		return "";
}
/*
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
	
}*/

function load_data_to_data ( datastring ) {
	// convert a string into a matrix data 
	var i;
	var cmd = "data = [ "; 
	var row;
	var rows = datastring.split("\n");
	for ( i=0; i< rows.length - 1; i++) {
		if ( rows[i] != "" ) {
			row = rows[i].replace(/,/g," ").replace(/ +/g,",");
			cmd += "[" + row + "] ,";
		}
	}
	if ( rows[rows.length-1] != "" ) {
		row = rows[rows.length-1].replace(/,/g," ").replace(/ +/g,",");
		cmd += "[" + row + "] ] ";
	}
	else {
		cmd = cmd.substr(0,cmd.length-1); // remove last comma
		cmd += "]";
	}
	
	return eval(cmd);
	
}


function run( script ) {
	if ( script.substr(script.length-3) != ".js" ) 
		script += ".js";
	importScripts(script); 	
}
load = run;
/*
function getWorkspace() {
	var res = new Object();
	var throwaway = ["MathFunctions" , "mf" , "self" , "console" , "location" , "onerror" , "onoffline" , "ononline" , "navigator" , "onclose" , "performance", "crypto", "indexedDB", "PERSISTENT", "TEMPORARY", "EPS", "caches"]; 
	for ( var i in self ) {
		if ( typeof( self[i] ) != "function" && !(self[i] instanceof CacheStorage)) {
			if ( MathFunctions.indexOf(i) < 0 && throwaway.indexOf(i) < 0 && i.indexOf("GLP") < 0 && i.indexOf("LPX") < 0 && i.indexOf("LPF_") < 0 && i.indexOf("webkit") < 0 && i.indexOf("LALOLIB_ERROR") < 0 ) {
				res[i] = self[i] ;
			}
		}
	}

	return object2binary(res);
}

function save( varList ) {
	if (arguments.length == 0) {
		var blob = getWorkspace();
		
		if ( typeof(JSZip) == "undefined")
			importScripts("jszip.min.js");
			
		var zip = new JSZip();
		var reader = new FileReaderSync();
  		
		zip.file("workspace.lab", reader.readAsArrayBuffer(blob) );
		var zipblob = zip.generate({type:"blob", compression: "DEFLATE"});
		postMessage( {cmd: "SAVE", output: zipblob } );
			
//		postMessage( {cmd: "SAVE", output: blob } );
		return "Workspace saved.";
	}
	else {
		var list = {}; 
		for ( var i=0; i< arguments.length; i++) {
			list[arguments[i]] = eval(arguments[i]);
		}
		var blob = object2binary( list );
		postMessage( {cmd: "SAVE", output: blob } );
		return arguments.length + " variables saved.";	
	}
}

function setWorkspace( WorkspaceBlob ) {
	//binary2object( WorkspaceBlob ) ;
	
	if ( typeof(JSZip) == "undefined")
		importScripts("jszip.min.js");
	var zip = new JSZip(WorkspaceBlob);
	var ab = zip.file("workspace.lab").asArrayBuffer();
	var blob = new Blob([ab]);
	binary2object( blob ) ;	// XXX rewrite this function to work with ab directly....
	
	return "Workspace loaded.";
}


function object2binary ( x ) {
	var data = new Array(); 
	var p; 
	for ( p in x ) {
		switch ( type( x[p] ) ) {
			case "string":
				data.push( makeCorrectLength(p) + makeCorrectLength("string") ); 
				data.push( new Uint32Array([x[p].length]) );
				data.push ( x[p] );
				break;
			case "boolean":
				data.push( makeCorrectLength(p) + makeCorrectLength("boolean") );
				data.push( x[p] ? 1 : 0 );
				
				break;
			case "number":
				data.push( makeCorrectLength(p) + makeCorrectLength("number") );
				data.push( new Float64Array([x[p]]) );
				break;
			case "vector":
				if ( x[p].buffer ) {
					data.push ( makeCorrectLength(p) + makeCorrectLength("vector") );
					data.push( new Uint32Array([x[p].length]) );
					data.push( x[p] );
				}
				else {
					data.push ( makeCorrectLength(p) + makeCorrectLength("array") );
					data.push( new Uint32Array([x[p].length]) );
					data.push( new Float64Array(x[p]) ); // for Array-vectors (lists)
				}
				break;
			case "matrix":
				var m = x[p].length;
				var n = x[p].n; 
				var vec = x[p].val;
				data.push ( makeCorrectLength(p) + makeCorrectLength("matrix") );
				data.push( new Uint32Array([ x[p].length, x[p].n]) );
				data.push( vec );
				break;
			case "Array":				
				var objdata = object2binary( x[p] );
				data.push ( makeCorrectLength(p) + makeCorrectLength("Array") );
				data.push( new Uint32Array([objdata.size]) );
				data.push( objdata )
				break;
			case "function": 
				var funstr = x[p].toString();
				data.push ( makeCorrectLength(p) + makeCorrectLength("function") );
				data.push( new Uint32Array([funstr.length]) );
				data.push( funstr );
				break;
			case "undefined":
				data.push( makeCorrectLength(p) + makeCorrectLength("undefined") );
				break;				
			default:
				var objdata = object2binary( x[p] );			// recursive	
				data.push ( makeCorrectLength(p) + makeCorrectLength("object") );
				data.push( new Uint32Array([objdata.size]) );
				data.push( objdata ); 
				break;			
		}
	}

	return new Blob(data, {type: "octet/stream"}); // "application/octet-binary"}); 
}


function binary2object ( blob , returnvalue) {

	var correctLength = 64;

	var myblobreader = new FileReaderSync();

	var blobpos = 0;
	var varCounter = 0;
	
	var blobvalue = new Array();
	
	if ( returnvalue)
		var tmpobj = {};
	
	while ( blobpos < blob.size ) {
		var varName = removeSpaces(myblobreader.readAsBinaryString(blob.slice(blobpos, blobpos+correctLength)));
		blobpos += correctLength;
	
		var varType = removeSpaces(myblobreader.readAsBinaryString(blob.slice(blobpos, blobpos+correctLength)));
		blobpos += correctLength;
	console.log(varName, varType);
		switch ( varType ) {
			case "string":
				var s = new Uint32Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 4 ) ) );
				var len = s[0];
				blobpos += 4;
				blobvalue[varCounter] = myblobreader.readAsText(blob.slice( blobpos, blobpos + len ) );
				blobpos += len;			
				break;
			
			case "boolean":
				blobvalue[varCounter] = ( myblobreader.readAsBinaryString(blob.slice( blobpos, blobpos + 1 ) )== "1" ? true : false );
				blobpos++;
				break;
			
			case "number":
				blobvalue[varCounter] = (new Float64Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 8 ) ) ))[0];
				blobpos += 8;
				break;
			
			case "vector":
				var s = new Uint32Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 4 ) ) );
				var n = s[0];
				blobpos += 4;

				blobvalue[varCounter] = new Float64Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 8*n ) ) );
				blobpos += 8*n;			
				break;

			case "array":
				var s = new Uint32Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 4 ) ) );
				var n = s[0];
				blobpos += 4;
				var vec = new Float64Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 8*n ) ) );
				blobvalue[varCounter] = new Array(n);
				for ( var i=0; i < n; i++)
					blobvalue[varCounter][i] = vec[i];
				blobpos += 8*n;			
				break;
			
			case "matrix":
				var s = new Uint32Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 8) ) );
				var m = s[0];
				var n = s[1];
				blobpos += 8;

				blobvalue[varCounter] = new Matrix(m,n,myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 8*m*n ) )  );
				blobpos += 8*m*n;			
				
				break;
			case "Array":
			// arrays of anything (but 1D...)
				var s = new Uint32Array(blob.slice( blobpos, blobpos + 4 ) );
				var len = s[0];
				blobpos += 4;
				var tmparray = binary2object ( blob.slice(blobpos, blobpos + len) , true) ;
				blobvalue[varCounter] = new Array();
				for ( var t in tmparray)
					blobvalue[varCounter][t] = tmparray[t];
				blobpos += len
				break;		
			
			case "function": 
				var s = new Uint32Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 4 ) ) );
				var len = s[0];
				blobpos += 4;
				eval("blobvalue[varCounter] = " + myblobreader.readAsText(blob.slice( blobpos, blobpos + len ) ));
				//blobvalue[varCounter] = myblobreader.readAsText(blob.slice( blobpos, blobpos + len ) );
				blobpos += len;			
				break;
			
			case "undefined":
				blobvalue[varCounter] = undefined;
				break;
				
			default:
				var s = new Uint32Array(myblobreader.readAsArrayBuffer(blob.slice( blobpos, blobpos + 4 ) ) );
				var len = s[0];
				blobpos += 4;
				blobvalue[varCounter] = binary2object ( blob.slice(blobpos, blobpos + len) , true) ;
				blobpos += len
				break;				
		}	
		
		// create variable
		if ( returnvalue)
			tmpobj[varName] = blobvalue[varCounter]; 			
		else
			eval(  varName + " = blobvalue[" + varCounter + "];"); 
		varCounter++;
	}

	if ( returnvalue)
		return tmpobj; // for recursive use
}

function makeCorrectLength( str ) {
	var correctLength = 64;
	while ( str.length < correctLength )
		str+= " "; 
		
	return str;
}
*/
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
function getWorkspace() {
	var res = new Object();
	var throwaway = ["MathFunctions" , "mf" , "self" , "console" , "location" , "onerror" , "onoffline" , "ononline" , "navigator" , "onclose" , "performance", "crypto", "indexedDB", "PERSISTENT", "TEMPORARY", "EPS", "caches"]; 
	for ( var i in self ) {
		if ( typeof( self[i] ) != "function" && !(self[i] instanceof CacheStorage)) {
			if ( MathFunctions.indexOf(i) < 0 && throwaway.indexOf(i) < 0 && i.indexOf("GLP") < 0 && i.indexOf("LPX") < 0 && i.indexOf("LPF_") < 0 && i.indexOf("webkit") < 0 && i.indexOf("LALOLIB_ERROR") < 0 ) {
				res[i] = self[i] ;
			}
		}
	}

	return getObjectWithoutFunc(res);
}

function save( varList ) {
	if ( typeof(JSZip) == "undefined")
			importScripts("jszip.min.js");
			
	var zip = new JSZip();
	
	if (arguments.length == 0) {
		var str = JSON.stringify(getWorkspace());
		
		zip.file("workspace.lab", str );
		var zipblob = zip.generate({type:"blob", compression: "DEFLATE"});
		
		postMessage( {cmd: "SAVE", output: zipblob } );
			
		return "Workspace saved.";
	}
	else {
		var list = {}; 
		for ( var i=0; i< arguments.length; i++) 
			list[arguments[i]] = eval(arguments[i]);
		
		var str = JSON.stringify(getObjectWithoutFunc( list ));
		zip.file("workspace.lab", str );
		var zipblob = zip.generate({type:"blob", compression: "DEFLATE"});
		
		postMessage( {cmd: "SAVE", output: zipblob } );
		return arguments.length + " variables saved.";	
	}
}
function setWorkspace( WorkspaceBlob ) {
	if ( typeof(JSZip) == "undefined")
		importScripts("jszip.min.js");
	var zip = new JSZip(WorkspaceBlob);
	var obj = JSON.parse(zip.file("workspace.lab").asText());
	for ( var p in obj ) {
		self[p] = renewObject(obj[p]);
	}
	return "Workspace loaded.";
}
