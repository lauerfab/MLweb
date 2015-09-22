/*
	Library for plotting functions 
	
	You need to include 
	
		 <canvas id="plotcanvas" width="600" height="300" style="border: 1px solid black;">>   </canvas> 
		 
	
	Usage:
	
		setScalePlot ( minX, maxX, Nsamples, scaleY)	// scaleY is a factor of scaleX
		
		plot( f [, color_index ] ) 

	To clear the plot: 
		clearPlot();
*/


//////////////////////////////
// Cross-browsers compatibility:
/*
	Chrome : turn off hardware acceleration to get mousemove events!
*/

/* Array.fill : 
if (!Array.prototype.fill) {
  Array.prototype.fill = function(value) {
  	if (this == null) {
      throw new TypeError("this is null or not defined");
    }
    if ( typeof( value ) == "object") 
    	throw new TypeError("Array.fill:: the value is not valid => only simple values allowed");
    
    
    var O = Object(this);
    for ( var i= 0; i < this.length; i++) {
    	O[i] = eval(value);
    }
    return O;	
  }

}

*/

////////////////////////////
// class Diagram
//
// functions take lengths in % of width and height
/////////////////////////////
function Diagram(canvasId) {
	if(typeof(canvasId) === 'undefined' ) 
		canvasId = "diagram";

	this.canvasId = canvasId; 
		
	this.shapes = new Array();
	this.selectedShape = -1; // for mousemove
	this.selectedShapes = new Array();	// for user
	
	this.mousexprev = -1;
	this.mouseyprev = -1;

	////// Cross browser support ////
	var ctx = document.getElementById(this.canvasId).getContext("2d");
	if ( !ctx.setLineDash ) {
		ctx.setLineDash = function () {};
	}
}

Diagram.prototype.rect = function (x,y,w, h, color, txt, txtcolor, opacity ) {
	if(typeof(opacity) === 'undefined')
		var opacity = 0.6;
	if(typeof(txtcolor) === 'undefined')
		var txtcolor = 0;
	if(typeof(txt) === 'undefined')
		var txt = "";
	if(typeof(color) === 'undefined')
		var color = 1;
	
	
	this.shapes.push( ["rect",  x, y, w, h, color, txt, txtcolor, opacity  ] ) ;
	
	this.redraw();
}
Diagram.prototype.circle = function (x,y,w, h, color, txt, txtcolor, opacity ) {
	if(typeof(opacity) === 'undefined')
		var opacity = 0.6;
	if(typeof(txtcolor) === 'undefined')
		var txtcolor = 0;
	if(typeof(txt) === 'undefined')
		var txt = "";
	if(typeof(color) === 'undefined')
		var color = 1;
	
	
	this.shapes.push( ["circle",  x, y, w, h, color, txt, txtcolor, opacity  ] ) ;
	
	this.redraw();
}
Diagram.prototype.image = function (x,y,w, h, imagename , txt, txtcolor, opacity) {
	if(typeof(opacity) === 'undefined')
		var opacity = 0.6;
	if(typeof(txtcolor) === 'undefined')
		var txtcolor = 0;
	if(typeof(txt) === 'undefined')
		var txt = "";
	
	var t = this;
	var imageIndex = this.shapes.length;
	var image = new Image() ;
	image.src = imagename;	
	image.onload = function() {		
		t.shapes[imageIndex][9] = true;
		t.redraw(); 
	}
 
	this.shapes.push( ["image", x,y,w,h,image, txt, txtcolor, opacity, false ] ); 
}

Diagram.prototype.redraw = function () {
	var canvas = document.getElementById(this.canvasId);
	var ctx = canvas.getContext("2d");
	ctx.clearRect(0,0,canvas.width, canvas.height);
	
	var n;
	var shape;
	var x;
	var y;
	var w;
	var h;
	var color;
	var txt;
	var txtcolor;
	var opacity;
	var res;
	
	// Draw shapes
	for ( n = 0; n < this.shapes.length; n++) {
		shape = this.shapes[n][0];
		x = this.shapes[n][1];
		y = this.shapes[n][2];
		w = this.shapes[n][3];
		h = this.shapes[n][4];
		color = this.shapes[n][5];
		txt = this.shapes[n][6];
		txtcolor = this.shapes[n][7];
		opacity = this.shapes[n][8];
		
		if ( shape == "rect" ) {

			setcolortransparent(ctx, color, opacity);
		
			
			var cornerSize = 15;
			ctx.beginPath();
			ctx.moveTo ( x * canvas.width , y * canvas.height + cornerSize);
			ctx.quadraticCurveTo( x * canvas.width, y * canvas.height, x * canvas.width + cornerSize, y * canvas.height );
			// quadraticCurve = bezier curve ( control poitn, destination)
			ctx.lineTo ( (x+w) * canvas.width - cornerSize, y * canvas.height);
			ctx.quadraticCurveTo( (x+w) * canvas.width , y * canvas.height, (x+w) * canvas.width, y * canvas.height + cornerSize);
			ctx.lineTo ( (x+w) * canvas.width , (y+h) * canvas.height - cornerSize);
			ctx.quadraticCurveTo( (x+w) * canvas.width, (y+h) * canvas.height, (x+w) * canvas.width - cornerSize, (y+h) * canvas.height );
			ctx.lineTo ( x * canvas.width + cornerSize , (y+h) * canvas.height);
			ctx.quadraticCurveTo( x * canvas.width, (y+h) * canvas.height, x * canvas.width , (y+h) * canvas.height - cornerSize );

			ctx.closePath();
			ctx.fill();
			
			//ctx.fillRect( x * canvas.width, y * canvas.height, w * canvas.width, h * canvas.height ) ;
	
			
			// deal with selection
			if ( n == this.selectedShape  || this.selectedShapes.indexOf( n ) >= 0 ) {
				setcolortransparent(ctx, 5, 0.3);
				ctx.fillRect( (x-0.005) * canvas.width, (y-0.005) * canvas.height, (w+0.01) * canvas.width, (h+0.01) * canvas.height ) ;
			}
	
		}
		else if ( shape == "circle" ) {
			setcolortransparent(ctx, color, opacity);
		
			ctx.beginPath();
			ctx.moveTo ( (x+w/2) * canvas.width , y * canvas.height);
			ctx.quadraticCurveTo( (x+w) * canvas.width, y * canvas.height, (x+w) * canvas.width, (y+h/2) * canvas.height );
			ctx.quadraticCurveTo( (x+w) * canvas.width, (y+h) * canvas.height, (x+w/2) * canvas.width, (y+h) * canvas.height );
			ctx.quadraticCurveTo( x * canvas.width, (y+h) * canvas.height, x * canvas.width, (y+h/2) * canvas.height );
			ctx.quadraticCurveTo( x * canvas.width, y * canvas.height, (x+w/2) * canvas.width, y * canvas.height );
			
			ctx.fill();
			
			// deal with selection
			if ( n == this.selectedShape  || this.selectedShapes.indexOf( n ) >= 0 ) {
				setcolortransparent(ctx, 5, 0.3);
				ctx.fillRect( (x-0.005) * canvas.width, (y-0.005) * canvas.height, (w+0.01) * canvas.width, (h+0.01) * canvas.height ) ;
			}
			
		}
				else if ( shape == "point" ) {
			setcolortransparent(ctx, color, opacity);
		
			ctx.beginPath();
			ctx.arc(x * canvas.width , y * canvas.height , w * canvas.width, 0, 2 * Math.PI , true);
			ctx.closePath();
				
			ctx.fill();
			
			// deal with selection
			if ( n == this.selectedShape  || this.selectedShapes.indexOf( n ) >= 0 ) {
				setcolortransparent(ctx, 5, 0.3);
				ctx.fillRect( (x-0.005) * canvas.width, (y-0.005) * canvas.height, (w+0.01) * canvas.width, (h+0.01) * canvas.height ) ;
			}
			
		}
		else if ( shape == "label" ) {
			setcolortransparent(ctx, color, opacity);
			var lbl = document.getElementById(this.shapes[n][9]);
			lbl.style.left = x * canvas.width;
			lbl.style.top = y * canvas.height;
			lbl.style.visibility = "visible"; 
			
		}
		else if ( shape == "arrow" ) {
			setcolortransparent(ctx, color, opacity);
					
			var arrowSize = 15;
			
			ctx.save();
			ctx.translate(x * canvas.width , y * canvas.height);
			ctx.rotate(Math.PI * (this.shapes[n][9] / 180) );
			
			ctx.beginPath();
			ctx.moveTo ( 0,0);
			ctx.lineTo ( (w) * canvas.width,0);
			ctx.lineTo ( (w) * canvas.width, 0 - arrowSize*0.3);
			ctx.lineTo ( (w) * canvas.width + arrowSize, ( h/2) * canvas.height);
			ctx.lineTo ( (w) * canvas.width, (h) * canvas.height + arrowSize*0.3);			
			ctx.lineTo ( (w) * canvas.width , (h) * canvas.height);
			ctx.lineTo ( 0 , (h) * canvas.height);
			
			ctx.closePath();
			ctx.fill();
			
			ctx.restore();
			
		}
		else if ( shape == "image" ) {
			if ( this.shapes[n][9] ) {
				// iamge is ready
				ctx.drawImage(this.shapes[n][5], x*canvas.width, y*canvas.height, w * canvas.width, h * canvas.height);			
				// deal with selection
				if ( n == this.selectedShape || this.selectedShapes.indexOf( n ) >= 0 ) {
					setcolortransparent(ctx, 3, 0.3);
					ctx.fillRect( (x-0.005) * canvas.width, (y-0.005) * canvas.height, (w+0.01) * canvas.width, (h+0.01) * canvas.height ) ;
				}
			}
		}
		 
		if( txt != "" ) { 
			var words = txt.split("*");
			ctx.textAlign = "center";	// center of text appear at x position
			var txtsize = Math.floor(50 * w) ;
			ctx.font = txtsize + "pt sans-serif";
			setcolor(ctx, txtcolor);
		
			if ( words.length == 1 ) {
				ctx.fillText( txt, (x + w/2) * canvas.width , (y + h/2) * canvas.height ) ;
			}
			else { 
				for (var i = 0; i< words.length; i++) {
					ctx.fillText( words[i], (x + w/2) * canvas.width , (y + h/2 ) * canvas.height - (words.length/2 - i - 0.5)* (1.5 * txtsize)) ;
				}
			}
		}
		
	}
	
}

Diagram.prototype.mouseselect = function (event) {
	var canvas= document.getElementById(this.canvasId);
	var rect = canvas.getBoundingClientRect();
	var x = event.clientX - rect.left;	// mouse coordinates relative to plot
	var y = event.clientY - rect.top;
	
	if ( Math.abs(x - this.mousexprev) >= 1 || Math.abs(y - this.mouseyprev) >= 1 ) {
		this.mousexprev = x;
		this.mouseyprev = y;
		
		// Find shape... starting from last one added which is on top of others...
		var i = this.shapes.length - 1;
		while ( i >= 0 && this.isInShape(x,y,this.shapes[i] ) == false )
			i--;
	
		if ( i >= 0 ) {
			if ( i != this.selectedShape ) {	
				// new hit on shape i 
				this.selectedShape = i;
				this.redraw();
		
				this.onSelect();
			}
		}	
		else if ( this.selectedShape >= 0 ) {
			this.onDeselect();
			this.selectedShape = -1;
			this.redraw();
		}
	}
}

Diagram.prototype.isInShape = function (x, y, shape) {
	var canvas = document.getElementById(this.canvasId);
	if(shape[0] == "rect") {
		if ( x > shape[1] * canvas.width && x < ( shape[1] + shape[3] ) * canvas.width && y > shape[2] * canvas.height && y < (shape[2]+shape[4]) * canvas.height)
			return true;
		else 
			return false;
	}
	else if ( shape[0] == "circle" ) {
		if ( x > shape[1] * canvas.width && x < ( shape[1] + shape[3] ) * canvas.width && y > shape[2] * canvas.height && y < (shape[2]+shape[4]) * canvas.height)
			return true;
		else 
			return false;
	}
	else if ( shape[0] == "arrow" ) {
		return false;
	}
	else 
		return false;
}

Diagram.prototype.onSelect = function () {
	// empty selection event handler
}

Diagram.prototype.onDeselect = function () {
	// empty selection event handler
}

Diagram.prototype.select = function ( n ) {
	if ( typeof(n) == "number" ) {
		if ( this.selectedShapes.indexOf( n ) < 0 )  {
			this.selectedShapes.push ( n );	
			this.redraw();
		}
	}
	else {
		for ( var i=0; i < n.length; i++ ) {
			if ( this.selectedShapes.indexOf( n[i] ) < 0 )  {
				this.selectedShapes.push ( n[i] );					
			}
		}
		this.redraw();
	}
}
Diagram.prototype.deselect = function ( n ) {
	if ( typeof(n) == "number" ) {
		var idx = this.selectedShapes.indexOf( n );
		if ( idx >= 0 ) {
			this.selectedShapes.splice ( idx , 1 );	
			this.redraw();
		}
	}
	else {
		var idx;
		for ( var i=0; i < n.length; i++ ) {
			idx = this.selectedShapes.indexOf( n[i] );
			if ( idx >= 0 ) {
				this.selectedShapes.splice ( idx , 1 );	
			}
		}
		this.redraw();
	}
}
Diagram.prototype.selectall = function ( n ) {
	for( var i = 0; i < this.shapes.length; i++)
		this.selectedShapes.push( i);
	this.redraw();
}
Diagram.prototype.deselectall = function ( ) {
	while (this.selectedShapes.length > 0)
		this.selectedShapes.pop();
	this.redraw();
	
}

////////////////////////////
// Define Object class "Plot" to be assigned to a canvas
/////////////////////////////
function Plot(canvasId) {
	if(typeof(canvasId) === 'undefined' ) 
		canvasId = "plotcanvas";

	this.canvasId = canvasId; 
		
	this.minX = 0;
	this.maxX = 10;
	this.Nsamples = 1000;
	this.scaleX = 1;
	this.scaleY = 1;
	this.minY = 0;
	this.maxY = 1.5;	
	 
	this.fcts = new Array();
	this.lines= new Array();
	this.areas= new Array();
	this.points= new Array(); 	
	this.paths= new Array();

	this.legend = "topright";

	var canvas = document.getElementById(this.canvasId);
	this.buffer = document.createElement('canvas');
	this.buffer.width  = canvas.width;
	this.buffer.height = canvas.height;
	
	this.viewX = 0;
	this.viewY = 0;
	
	////// Cross browser support ////
	//var ctx = document.getElementById(this.canvasId).getContext("2d");
	var ctx = this.buffer.getContext("2d");
	if ( !ctx.setLineDash ) {
		ctx.setLineDash = function () {};
	}
}


Plot.prototype.addPoint = function(x,y,color_idx,radius, opacity) {
	if(typeof(color_idx) === 'undefined')
		color_idx = 0;
	if(typeof(radius) === 'undefined')
		radius = 5;
	if(typeof(opacity) === 'undefined')
		opacity = 1.1;
		
	this.points.push([x,y,color_idx,radius,opacity] );
}

Plot.prototype.plotAxis = function() {
	//var canvas = document.getElementById(this.canvasId);
	var canvas = this.buffer; 
  if (canvas.getContext) {
	var ctx = canvas.getContext("2d");
	ctx.fillStyle="white";
	ctx.fillRect (0,0 , canvas.width, canvas.height);
	ctx.strokeStyle = "black";			
	
	if (this.minY < 0 && this.maxY > 0) {
		// X-axis		
		var y0 = canvas.height - (-this.minY * this.scaleY);
		ctx.beginPath();
		ctx.moveTo(0, y0);
		ctx.lineTo(canvas.width, y0 );
		ctx.closePath();
		ctx.stroke();
		
		// ticks		 
		var tickspace = Math.ceil( (this.maxX - this.minX) / 10);
		for (var x = -tickspace; x>this.minX; x -= tickspace ) {
			var xx = (x - this.minX) * this.scaleX ;
			ctx.beginPath();
			ctx.moveTo(xx,y0 - 5 );
			ctx.lineTo(xx, y0 + 5 );
			ctx.stroke();		
		}
		for (var x = tickspace; x < this.maxX ; x+=tickspace ) {		
			var xx = (x - this.minX) * this.scaleX ;
			ctx.beginPath();
			ctx.moveTo(xx,y0 - 5 );
			ctx.lineTo(xx, y0 + 5 );
			ctx.stroke();		
		}
	}
	
	if (this.minX < 0 && this.maxX > 0) {
		// Y-axis
		var x0 = -this.minX * this.scaleX;
		ctx.beginPath();
		ctx.moveTo(x0 ,0);
		ctx.lineTo(x0 ,canvas.height);
		ctx.closePath();
		ctx.stroke();
		
		// ticks		 
		var tickspace = Math.ceil( (this.maxY - this.minY) / 10);
		for (var y = -tickspace; y>this.minY; y -= tickspace ) {
			var yy = (y - this.minY) * this.scaleY ;
			ctx.beginPath();
			ctx.moveTo(x0 -5 ,canvas.height-yy );
			ctx.lineTo(x0 + 5, canvas.height-yy );
			ctx.stroke();		
		}
		for (var y = tickspace; y<this.maxY; y += tickspace ) {
			var yy = (y - this.minY) * this.scaleY ;
			ctx.beginPath();
			ctx.moveTo(x0 -5 , canvas.height-yy );
			ctx.lineTo(x0 + 5, canvas.height- yy );
			ctx.stroke();	
		}		
	}
  }
}


Plot.prototype.replot = function (  ) {
	
	var x1;
	var x2;
	var y1;
	var y2;
	var opacity;
	var radius;
	var x;
	var y;
	
	var f;
	var legend;
	var color_idx;
	var dashed;
	var fillareaTo;
	var nlegend = 0;
	var res;

	var canvas = this.buffer;  
//	var canvas=document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		this.plotAxis();
		
		// use shadow but not on axis
		ctx.shadowColor = '#999';
      	ctx.shadowBlur = 3;
      	ctx.shadowOffsetX = 3;
      	ctx.shadowOffsetY = 3;
      
		
		const minX = this.minX;
		const minY = this.minY;		
		const scaleX = this.scaleX;
		const scaleY = this.scaleY;
		const height = canvas.height;
		
		var xplot = function (x) {
			return (x-minX ) * scaleX ;
		}
		var yplot = function (y) {
			return height - (y-minY)*scaleY ;
		}


	
		// Plot areas
		for (var n=0; n < this.areas.length; n++)	{
			res = this.areas[n];
			x1 = res[0];
			y1 = res[1];
			x2 = res[2];
			y2 = res[3];
			color_idx = res[4];
			opacity = res[5];
		
			if(color_idx == -1) {
				color_idx = n+1;
			}
			setcolortransparent(ctx, color_idx, opacity);
			var rectwidth = Math.abs(x2-x1);
			var rectheight = Math.abs(y2 -y1);
			var rectx = Math.min(x1,x2);
			var recty = Math.max(y1,y2);
			ctx.fillRect(( rectx-this.minX ) * this.scaleX , canvas.height - ( recty - this.minY) * this.scaleY , rectwidth * this.scaleX ,  rectheight * this.scaleY );

		}

		// Plot lines
		ctx.lineWidth="3";
		var cp = Infinity;
		for (var n=0; n < this.lines.length; n++)	{
			res = this.lines[n];
			
			if ( ( res[0] >= this.minX && res[0] <= this.maxX && res[1] >= this.minY && res[1] <= this.maxY ) //start in plot
				|| (( res[2] >= this.minX && res[2] <= this.maxX && res[3] >= this.minY && res[3] <= this.maxY ))  // end in plot
				|| ( res[0] < this.minX && res[2] > this.maxX && ((res[1] >= this.minY && res[1] <= this.maxY) || (res[3] >= this.minY && res[3] <= this.maxY))  )	// overflow on x axis but y inside plot
				|| ( res[2] < this.minX && 0 > this.maxX && ((res[1] >= this.minY && res[1] <= this.maxY) || (res[3] >= this.minY && res[3] <= this.maxY))  )	
				|| ( res[1] < this.minY && res[3] > this.maxY && ((res[0] >= this.minX && res[0] <= this.maxY) || (res[2] >= this.minX && res[2] <= this.maxX))  )// y-axis	
				|| ( res[3] < this.minY && res[1] > this.maxY && ((res[0] >= this.minX && res[0] <= this.maxX) || (res[2] >= this.minX && res[2] <= this.maxX))  )			
				) {
			
				x1 = xplot(res[0]);
				y1 = yplot(res[1]);
				x2 = xplot(res[2]);
				y2 = yplot(res[3]);

				if ( Math.abs(x2-x1)>1 || Math.abs(y2-y1) > 1 )  {
					color_idx = res[4];
					dashed = res[5];
	
					if(color_idx == -1) {
						color_idx = n+1;
					}
					
					if ( color_idx != cp )
						setcolor(ctx, color_idx);
		
					if (dashed) {
						ctx.setLineDash([5]);
						ctx.lineWidth="1";
					}
					
					ctx.beginPath();		
					ctx.moveTo(x1 , y1);
					ctx.lineTo(x2 ,y2);
					ctx.stroke();
					
					if (dashed) {
						ctx.setLineDash([1, 0]);
						ctx.lineWidth="3";
					}
					
					cp = color_idx;
				}
			}
		}	
		ctx.lineWidth="1";
		
		// Plot points 
		var xp = Infinity;
		var yp = Infinity;
		var cp = Infinity;
		var op = -1;
		for (var n=0; n < this.points.length; n++)	{
			res = this.points[n];
			
			if ( res[0] >= this.minX && res[0] <= this.maxX && res[1] >= this.minY && res[1] <= this.maxY) {
				
				x = xplot(res[0]);
				y = yplot(res[1]);
				if ( Math.abs(x-xp)>1 || Math.abs(y-yp) > 1  ) {
					color_idx = res[2];
					radius = res[3];
					opacity = res[4];
	
					if ( op != opacity || cp != color_idx) {
						if ( opacity > 1.0 ) 
							setcolor(ctx, color_idx);
						else 	
							setcolortransparent(ctx, color_idx, opacity);
					}
					
					ctx.beginPath();
					ctx.arc( x , y , radius, 0, 2 * Math.PI , true);
					ctx.closePath();
					ctx.fill();
				
					xp = x;
					yp = y;
					cp = color_idx;
					op = opacity;
				}
			}
		}
	
		// Plot paths (sets of point-lines with all the same style, e.g.,  for lalolab functions)
		for (var n=0; n < this.paths.length; n++) {
			res = this.paths[n];
			color_idx = res[1];
			radius = res[2];
			opacity = res[3];
			dashed = res[4];
			var marker = (opacity > 0 );
			if ( opacity > 1.0 ) 
				setcolor(ctx, color_idx);
			else 	
				setcolortransparent(ctx, color_idx, opacity);
		
			if (dashed) {
				ctx.setLineDash([5]);
				ctx.lineWidth="1";
			}
			else{
				ctx.lineWidth="3";
			}
			ctx.beginPath();

			x = xplot(res[0][0][0]);
			y = yplot(res[0][0][1]);

			ctx.arc( x , y , radius, 0, 2 * Math.PI , true);	
			ctx.moveTo(x,y);				

			for ( var i=1; i < res[0].length; i++) {
				x = xplot(res[0][i][0]);
				y = yplot(res[0][i][1]);
		
				if ( x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height ) {
					if( marker )
						ctx.arc( x , y , radius, 0, 2 * Math.PI , true);	
					ctx.lineTo(x,y);				
				}
			}
			//ctx.closePath();
			ctx.stroke();
			//ctx.fill();
			
			ctx.setLineDash([1, 0]);
			ctx.lineWidth="1";
		}
		
			// Plot functions
		for(var n=0; n < this.fcts.length; n++)	{

			res = this.fcts[n];
			f = res[0];
			legend = res[1];
			color_idx = res[2];
			dashed = res[3];
			fillareaTo = res[4];
		
	
			if(color_idx == -1) {
				color_idx = n+1;
			}
		
			setcolor(ctx, color_idx);

			if (dashed) {
				ctx.setLineDash([5]);
				ctx.lineWidth="1";
			}
			else{
				ctx.lineWidth="3";
			}
	

		
			if ( fillareaTo !== false ) {
				ctx.beginPath();
				ctx.moveTo(canvas.width, canvas.height - (fillareaTo  - this.minY)* this.scaleY);
				ctx.lineTo(0, canvas.height - (fillareaTo  - this.minY)* this.scaleY );
				//ctx.moveTo(0,canvas.height/2);
			}
			else {
				ctx.moveTo(0,canvas.height/2);		
				ctx.beginPath();
			}

			for(var x=this.minX; x < this.maxX; x += (this.maxX-this.minX) / this.Nsamples ) {
				var y = f(x);
				var yp = canvas.height - ( y - this.minY) * this.scaleY ;
				if (yp >= 0 && yp <= canvas.height) 
					ctx.lineTo(xplot(x) , yp );
				else
					ctx.moveTo(xplot(x) , yp);
	
			}
			ctx.stroke();
			if ( fillareaTo !== false ) {
				ctx.closePath();
				setcolortransparent(ctx, color_idx, 0.5); 
				ctx.fill();
			}
			ctx.setLineDash([1, 0]);
			ctx.lineWidth="1";
		
			// Add legend: 
			if ( this.legend != "" && legend != "") {
				setcolor(ctx, color_idx); 
				if ( this.legend == "topright") 
					ctx.strokeText(legend, canvas.width - 100, 20*(nlegend+1));
				else if ( this.legend == "topleft") 
					ctx.strokeText(legend, 10, 20*(nlegend+1));
				else if ( this.legend == "bottomright") 
					ctx.strokeText(legend, canvas.width - 100, canvas.height - 20*(nlegend+1));
				else if ( this.legend == "bottomleft") 
					ctx.strokeText(legend,10, canvas.height - 20*(nlegend+1));
			
				nlegend++;
			}

		}
	}
	
	// Copy buffer to viewport
	var viewcanvas = document.getElementById(this.canvasId);
	var ctx = viewcanvas.getContext("2d");
	ctx.drawImage(this.buffer, this.viewX, this.viewY, viewcanvas.width,viewcanvas.height,0,0, viewcanvas.width,viewcanvas.height);
}


Plot.prototype.plot = function ( f, legend, color_idx, dashed , fillareaTo ) {
	if (typeof(fillareaTo) === 'undefined')
		fillareaTo = false;
 
	if (typeof(dashed) === 'undefined')
		dashed = false; 
	if (typeof(color_idx) === 'undefined')
		color_idx = -1; 
	if (typeof(legend) === 'undefined') {
		if (dashed)
			legend = "";
		else
			legend = f.name; 
	}
	this.fcts.push([f, legend, color_idx,dashed, fillareaTo]);
	this.replot();
}


Plot.prototype.plot_line = function ( x1,y1,x2,y2, color_idx, dashed  ) {
	if (typeof(dashed) === 'undefined')
		dashed = false; 
	if (typeof(color_idx) === 'undefined')
		color_idx = -1; 
	this.lines.push([x1,y1,x2,y2, color_idx,dashed]);
	//this.replot();
}
Plot.prototype.plot_area = function ( x1,y1,x2,y2, color_idx, opacity  ) {
	if (typeof(opacity) === 'undefined')
		opacity = 1.0; 
	if (typeof(color_idx) === 'undefined')
		color_idx = -1; 
	this.areas.push([x1,y1,x2,y2, color_idx,opacity]);
	this.replot();
}
Plot.prototype.plot_path = function ( x,y, color_idx, radius, opacity, dashed  ) {
	if (typeof(dashed) === 'undefined')
		var dashed = false; 
	if (typeof(color_idx) === 'undefined')
		var color_idx = -1; 
	if (typeof(opacity) === 'undefined')
		var opacity = 1;
	if (typeof(radius) === 'undefined')
		var radius = 5;
	this.paths.push([x,y, color_idx,radius, opacity, dashed]);
	//this.replot();
}

Plot.prototype.clear = function () {
 var canvas = document.getElementById(this.canvasId);
  if (canvas.getContext) {
	var ctx = canvas.getContext("2d");
	
	this.plotAxis();
	
	// Empty list of functions to plot:
	while(this.fcts.length > 0) {
    	this.fcts.pop();
	}
	
	while(this.lines.length > 0) {
    	this.lines.pop();
	}
	
	while(this.areas.length > 0) {
    	this.areas.pop();
	}
	while(this.points.length > 0) {
    	this.points.pop();
	}
  }
}

Plot.prototype.setScalePlot = function  ( minX, maxX, Nsamples, scaleY) {
	this.minX = minX;
	this.maxX = maxX;
	this.Nsamples = Nsamples;
	
	var canvas = document.getElementById(this.canvasId);
	this.scaleX = canvas.width / (maxX - minX) ; 
	this.scaleY = this.scaleX * scaleY;
		
	this.maxY = (canvas.height/2) / this.scaleY ;
	this.minY = -this.maxY;// centered view 
	
	//this.clear();
	
	this.originalminX = this.minX;
	this.originalmaxX = this.maxX;
	this.originalminY = this.minY;
	this.originalmaxY = this.maxY;	
}

Plot.prototype.view = function  ( minX, maxX, minY, maxY) {
	this.minX = minX;
	this.maxX = maxX;
	this.minY = minY;
	this.maxY = maxY;
	
	var canvas = this.buffer;
	this.scaleX = canvas.width / (maxX - minX) ; 
	this.scaleY = canvas.height / (maxY - minY) ;
	this.replot(); 	
}

Plot.prototype.translate = function  ( dx, dy ) {
	var canvas = document.getElementById(this.canvasId);
	var newX = this.viewX - dx;
	var newY = this.viewY - dy;
	if ( newX >= 0 && newX < this.buffer.width - canvas.width && newY >= 0 && newY < this.buffer.height - canvas.height ) {
	
		this.viewX = newX;
		this.viewY = newY;
	
		var ctx = canvas.getContext("2d");
		ctx.clearRect (0, 0 , canvas.width, canvas.height);
		ctx.drawImage(this.buffer, this.viewX, this.viewY, canvas.width,canvas.height,0,0, canvas.width,canvas.height);
	}
}
Plot.prototype.zoom = function  ( zx, zy, x, y) {
	var viewcanvas = document.getElementById(this.canvasId);
	var canvas = this.buffer;
	
	if ( zy > 0 )
		canvas.height *= zy; 
	else
		canvas.height = viewcanvas.height; 
	if ( zx > 0 )
		canvas.width *= zx;
	else
		canvas.width = viewcanvas.width; 
		
	// do not zoom out further than original 
	if ( canvas.width < viewcanvas.width )
		canvas.width = viewcanvas.width; 
	if( canvas.height < viewcanvas.height )
		canvas.height = viewcanvas.height;

	// do not zoo in too much
	if ( canvas.width > 10000)
		canvas.width = 10000; 
	if( canvas.height > 10000 )
		canvas.height > 10000;
	
	var sx = this.scaleX;
	var sy = this.scaleY;
	this.scaleX = canvas.width / (this.maxX - this.minX) ; 
	this.scaleY = canvas.height / (this.maxY - this.minY) ;

	// zoom center is (x,y)
	if ( arguments.length < 4 ) {
		var x = viewcanvas.width/2;
		var y = viewcanvas.height/2;// by default viewport center is fixed during zoom
	}
	
	this.viewX = ((this.viewX + x) * this.scaleX / sx) - x;
	this.viewY = ((this.viewY + y) * this.scaleY / sy) - y;	
	if ( this.viewX < 0 )
		this.viewX = 0;
	if (this.viewY < 0 )
		this.viewY = 0; 
	if ( this.viewX > canvas.width - viewcanvas.width ) 
		this.viewX =  canvas.width - viewcanvas.width ;
	if ( this.viewY > canvas.height - viewcanvas.height ) 
		this.viewY =  canvas.height - viewcanvas.height ;

	if( sx != this.scaleX || sy != this.scaleY )
		this.replot(); 
}
Plot.prototype.resetzoom = function  ( ) {
	var viewcanvas = document.getElementById(this.canvasId);
	var canvas = this.buffer;
	this.viewX = 0;
	this.viewY = 0;
	canvas.height = viewcanvas.height; 
	canvas.width = viewcanvas.width; 
	this.scaleX = viewcanvas.width / (this.maxX - this.minX) ; 
	this.scaleY = viewcanvas.height / (this.maxY - this.minY) ;
	this.replot(); 	
}

Plot.prototype.pick_point = function(e) {
	if(e.button == 0) {
		e.preventDefault();	
		var canvas = document.getElementById(this.canvasId);
	
		var rect = canvas.getBoundingClientRect();

		var xmouse = e.clientX - rect.left;	// mouse coordinates relative to plot
		var ymouse = e.clientY - rect.top;

		var x = xmouse / this.scaleX + this.minX;
		var y = (canvas.height  - ymouse ) / this.scaleY + this.minY;
		
		return [x,y];
	}
	else 
		return false; // not correct button
}


Plot.prototype.proximityX = function (x, x0, epsilon) {
	if (typeof(epsilon) === 'undefined')
		epsilon = (this.maxX - this.minX) / 20;
		
	return ( Math.abs(x - x0) < epsilon ) ;
}


Plot.prototype.plotmathjax = function(stringindex, x, y) {
			
	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");

		var label = document.getElementById("jaxstring"+stringindex);
		label.style.top = canvas.height/2 - ( y * this.scaleY ) + canvas.offsetTop;
		label.style.left = (x - this.minX) * this.scaleX + canvas.offsetLeft;	
		label.style.visibility = "visible"; 
	}
}
	 
Plot.prototype.jpeg = function() {
	var canvas = document.getElementById(this.canvasId);
	
	var image = canvas.toDataURL("image/jpeg");
	
	document.location.href = image.replace("image/jpeg", "image/octet-stream");
}


Plot.prototype.zoomoriginal = function () {
	this.view(this.originalminX,this.originalmaxX,this.originalminY,this.originalmaxY);	
}

Plot.prototype.mousestartmove = function ( e ) {
	var canvas = document.getElementById(this.canvasId);
	if ( e.button == 0 ) {
		this.MOVING = true;
		var rect = canvas.getBoundingClientRect();
		this.xprev = e.clientX - rect.left;	// mouse coordinates relative to plot
		this.yprev = e.clientY - rect.top;
	}
	else {
		this.MOVING = false;
	}
}
Plot.prototype.mousestopmove = function ( e ) {
	this.MOVING = false;
}
Plot.prototype.mouseposition = function ( e ) {

	var canvas = document.getElementById(this.canvasId);
	var rect = canvas.getBoundingClientRect();

	var xmouse = e.clientX - rect.left;	
	var ymouse = e.clientY - rect.top;

	if ( this.MOVING ) {
		var dx = this.xprev - xmouse ;
		var dy = ymouse - this.yprev;
		if ( Math.abs( dx ) > 1 || Math.abs( dy ) > 1 ) {			
			//this.view(this.minX+dx/this.scaleX,this.maxX+dx/this.scaleX, this.minY+dy/this.scaleY, this.maxY+dy/this.scaleY);
			this.translate(dx, dy);
		}
		this.xprev = xmouse;
		this.yprev = ymouse;		
	}
	else {		
		var x = xmouse / this.scaleX + this.minX;
		var y = (canvas.height  - ymouse ) / this.scaleY + this.minY;	
		return "x = " + x.toFixed(3) + ", y = " + y.toFixed(3);	
	}
}



////////////////////////////
// Define Object class "ColorPlot" for (x,y) plots with z giving the point color
/////////////////////////////
function ColorPlot(canvasId) {
	if(typeof(canvasId) === 'undefined' ) 
		canvasId = "plotcanvas";

	this.canvasId = canvasId; 
		
	this.minX = 0;
	this.maxX = 10;
	this.scaleX = 1;
	this.scaleY = 1;
	this.minY = 0;
	this.maxY = 1.5;	
	this.minZ = 0;
	this.maxZ = 1;
		 
	this.x = new Array();
	this.y= new Array();
	this.z= new Array();
	
	this.cmap = this.colormap();
	
	var canvas = document.getElementById(this.canvasId);
	this.buffer = document.createElement('canvas');
	this.buffer.width  = canvas.width;
	this.buffer.height = canvas.height;
	
	this.viewX = 0;
	this.viewY = 0;

}

ColorPlot.prototype.colormap = function (cmapname) {
	switch(cmapname) {
	
	default:
    var cmap = [
		[0, 0, 143],
		[0, 0, 159],
		[0, 0, 175],
		[0, 0, 191],
		[0, 0, 207],
		[0, 0, 223],
		[0, 0, 239],
		[0, 0, 255],
		[0, 15, 255],
		[0, 31, 255],
		[0, 47, 255],
		[0, 63, 255],
		[0, 79, 255],
		[0, 95, 255],
		[0, 111, 255],
		[0, 127, 255],
		[0, 143, 255],
		[0, 159, 255],
		[0, 175, 255],
		[0, 191, 255],
		[0, 207, 255],
		[0, 223, 255],
		[0, 239, 255],
		[0, 255, 255],
		[15, 255, 239],
		[31, 255, 223],
		[47, 255, 207],
		[63, 255, 191],
		[79, 255, 175],
		[95, 255, 159],
		[111, 255, 143],
		[127, 255, 127],
		[143, 255, 111],
		[159, 255, 95],
		[175, 255, 79],
		[191, 255, 63],
		[207, 255, 47],
		[223, 255, 31],
		[239, 255, 15],
		[255, 255, 0],
		[255, 239, 0],
		[255, 223, 0],
		[255, 207, 0],
		[255, 191, 0],
		[255, 175, 0],
		[255, 159, 0],
		[255, 143, 0],
		[255, 127, 0],
		[255, 111, 0],
		[255, 95, 0],
		[255, 79, 0],
		[255, 63, 0],
		[255, 47, 0],
		[255, 31, 0],
		[255, 15, 0],
		[255, 0, 0],
		[239, 0, 0],
		[223, 0, 0],
		[207, 0, 0],
		[191, 0, 0],
		[175, 0, 0],
		[159, 0, 0],
		[143, 0, 0],
		[127, 0, 0]];
	break;
	}
	return cmap;
}
ColorPlot.prototype.addPoint = function(x,y,z) {
	this.x.push(x);
	this.y.push(y);
	this.z.push(z);
}

ColorPlot.prototype.plotAxis = function() {
	var canvas = this.buffer;	
  if (canvas.getContext) {
	var ctx = canvas.getContext("2d");
	ctx.fillStyle="white";
	ctx.fillRect (0,0 , canvas.width, canvas.height);
	ctx.strokeStyle = "black";			
	
	if (this.minY < 0 && this.maxY > 0) {
		// X-axis		
		var y0 = canvas.height - (-this.minY * this.scaleY);
		ctx.beginPath();
		ctx.moveTo(0, y0);
		ctx.lineTo(canvas.width, y0 );
		ctx.closePath();
		ctx.stroke();
		
		// ticks		
		var tickspace = Math.ceil( (this.maxX - this.minX) / 10);
		for (var x = -tickspace; x>this.minX; x -= tickspace ) {
			var xx = (x - this.minX) * this.scaleX ;
			ctx.beginPath();
			ctx.moveTo(xx,y0 - 5 );
			ctx.lineTo(xx, y0 + 5 );
			ctx.stroke();		
		}
		for (var x = tickspace; x < this.maxX ; x+=tickspace ) {		
			var xx = (x - this.minX) * this.scaleX ;
			ctx.beginPath();
			ctx.moveTo(xx,y0 - 5 );
			ctx.lineTo(xx, y0 + 5 );
			ctx.stroke();		
		}
	}
	
	if (this.minX < 0 && this.maxX > 0) {
		// Y-axis
		var x0 = -this.minX * this.scaleX;
		ctx.beginPath();
		ctx.moveTo(x0 ,0);
		ctx.lineTo(x0 ,canvas.height);
		ctx.closePath();
		ctx.stroke();
		
		// ticks
		for (var y = Math.ceil(this.minY); y < this.maxY; y++ ) {
			var yy = canvas.height - (y -this.minY) * this.scaleY;
			ctx.beginPath();
			ctx.moveTo(x0-5,yy);
			ctx.lineTo(x0+5,yy);
			ctx.stroke();		
		}
	}
  }
}

ColorPlot.prototype.replot = function (  ) {
	var x,y,z;
	var canvas=this.buffer;
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");

		this.plotAxis();
	  
	  	// use shadow but not on axis
		ctx.shadowColor = '#999';
      	ctx.shadowBlur = 3;
      	ctx.shadowOffsetX = 3;
      	ctx.shadowOffsetY = 3;
      
		// Plot points 
		var xp = Infinity;
		var yp = Infinity;
		var zp = Infinity;		
		for (var i=0; i < this.x.length; i++)	{
	
			if ( this.x[i] >= this.minX && this.x[i] <= this.maxX && this.y[i] >= this.minY && this.y[i] <= this.maxY) {
				
				x = (this.x[i]-this.minX ) * this.scaleX ;
				y =  canvas.height - (this.y[i] - this.minY) * this.scaleY ;
				z = Math.floor( (this.z[i] - this.minZ) * this.scaleZ);
				if ( z >= this.cmap.length )
					z = this.cmap.length-1;
				if ( z < 0)
					z = 0;
				if ( Math.abs(x-xp)>1 || Math.abs(y-yp) > 1 || z != zp ) {

					if ( z != zp )
						ctx.fillStyle = "rgb(" + this.cmap[z][0] + "," + this.cmap[z][1] + "," + this.cmap[z][2]+ ")";			

					ctx.beginPath();
					ctx.arc( x , y , 5, 0, 2 * Math.PI , true);
					ctx.closePath();
					ctx.fill();		
			
					zp = z;		
					xp = x;
					yp = y;
				}
			}
		}
	
	}
	
	// Copy buffer to viewport
	var viewcanvas = document.getElementById(this.canvasId);
	var ctx = viewcanvas.getContext("2d");
	ctx.drawImage(this.buffer, this.viewX, this.viewY, viewcanvas.width,viewcanvas.height,0,0, viewcanvas.width,viewcanvas.height);
}

ColorPlot.prototype.clear = function () {
	this.plotAxis();
	this.x = new Array();
	this.y = new Array();
	this.z = new Array();
}

ColorPlot.prototype.setScale = function  ( minX, maxX, minY, maxY, minZ, maxZ) {
	this.minX = minX;
	this.maxX = maxX;
	this.minY = minY;
	this.maxY = maxY;
	this.minZ = minZ;
	this.maxZ = maxZ;
	
	var canvas = document.getElementById(this.canvasId);
	this.scaleX = canvas.width / (maxX - minX) ; 
	this.scaleY = canvas.height / (maxY - minY);
	this.scaleZ = this.cmap.length / (maxZ - minZ) ;
	
	//this.clear();
	
	this.originalminX = this.minX;
	this.originalmaxX = this.maxX;
	this.originalminY = this.minY;
	this.originalmaxY = this.maxY;	
}

ColorPlot.prototype.view = function  ( minX, maxX, minY, maxY) {
	this.minX = minX;
	this.maxX = maxX;
	this.minY = minY;
	this.maxY = maxY;
	
	var canvas = this.buffer;
	this.scaleX = canvas.width / (maxX - minX) ; 
	this.scaleY = canvas.height / (maxY - minY) ;
	this.replot(); 	
}

ColorPlot.prototype.translate = function  ( dx, dy ) {
	var canvas = document.getElementById(this.canvasId);
	var newX = this.viewX - dx;
	var newY = this.viewY - dy;
	if ( newX >= 0 && newX < this.buffer.width - canvas.width && newY >= 0 && newY < this.buffer.height - canvas.height ) {
	
		this.viewX = newX;
		this.viewY = newY;
	
		var ctx = canvas.getContext("2d");
		ctx.clearRect (0, 0 , canvas.width, canvas.height);
		ctx.drawImage(this.buffer, this.viewX, this.viewY, canvas.width,canvas.height,0,0, canvas.width,canvas.height);
	}
}
ColorPlot.prototype.zoom = function  ( zx, zy, x, y) {
	var viewcanvas = document.getElementById(this.canvasId);
	var canvas = this.buffer;
	
	if ( zy > 0 )
		canvas.height *= zy; 
	else
		canvas.height = viewcanvas.height; 
	if ( zx > 0 )
		canvas.width *= zx;
	else
		canvas.width = viewcanvas.width; 
		
	// do not zoom out further than original 
	if ( canvas.width < viewcanvas.width )
		canvas.width = viewcanvas.width; 
	if( canvas.height < viewcanvas.height )
		canvas.height = viewcanvas.height;

	// do not zoo in too much
	if ( canvas.width > 10000)
		canvas.width = 10000; 
	if( canvas.height > 10000 )
		canvas.height > 10000;
	
	var sx = this.scaleX;
	var sy = this.scaleY;
	this.scaleX = canvas.width / (this.maxX - this.minX) ; 
	this.scaleY = canvas.height / (this.maxY - this.minY) ;

	// zoom center is (x,y)
	if ( arguments.length < 4 ) {
		var x = viewcanvas.width/2;
		var y = viewcanvas.height/2;// by default viewport center is fixed during zoom
	}
	
	this.viewX = ((this.viewX + x) * this.scaleX / sx) - x;
	this.viewY = ((this.viewY + y) * this.scaleY / sy) - y;	
	if ( this.viewX < 0 )
		this.viewX = 0;
	if (this.viewY < 0 )
		this.viewY = 0; 
	if ( this.viewX > canvas.width - viewcanvas.width ) 
		this.viewX =  canvas.width - viewcanvas.width ;
	if ( this.viewY > canvas.height - viewcanvas.height ) 
		this.viewY =  canvas.height - viewcanvas.height ;

	if( sx != this.scaleX || sy != this.scaleY )
		this.replot(); 
}
ColorPlot.prototype.resetzoom = function  ( ) {
	var viewcanvas = document.getElementById(this.canvasId);
	var canvas = this.buffer;
	this.viewX = 0;
	this.viewY = 0;
	canvas.height = viewcanvas.height; 
	canvas.width = viewcanvas.width; 
	this.scaleX = viewcanvas.width / (this.maxX - this.minX) ; 
	this.scaleY = viewcanvas.height / (this.maxY - this.minY) ;
	this.replot(); 	
}

ColorPlot.prototype.jpeg = function() {
	var canvas = document.getElementById(this.canvasId);
	
	var image = canvas.toDataURL("image/jpeg");
	
	document.location.href = image.replace("image/jpeg", "image/octet-stream");
}


ColorPlot.prototype.zoomoriginal = function () {
	this.view(this.originalminX,this.originalmaxX,this.originalminY,this.originalmaxY);	
}

ColorPlot.prototype.mousestartmove = function ( e ) {
	var canvas = document.getElementById(this.canvasId);
	if ( e.button == 0 ) {
		this.MOVING = true;
		var rect = canvas.getBoundingClientRect();
		this.xprev = e.clientX - rect.left;	// mouse coordinates relative to plot
		this.yprev = e.clientY - rect.top;
	}
	else {
		this.MOVING = false;
	}
}
ColorPlot.prototype.mousestopmove = function ( e ) {
	this.MOVING = false;
}
ColorPlot.prototype.mouseposition = function ( e ) {
	var canvas = document.getElementById(this.canvasId);
	var rect = canvas.getBoundingClientRect();

	var xmouse = e.clientX - rect.left;	
	var ymouse = e.clientY - rect.top;

	if ( this.MOVING ) {
		var dx = this.xprev - xmouse ;
		var dy = ymouse - this.yprev;
		if ( Math.abs( dx ) > 1 || Math.abs( dy ) > 1 ) {			
			this.translate(dx,dy);
		}
		this.xprev = xmouse;
		this.yprev = ymouse;		
	}
	else {		
		var x = xmouse / this.scaleX + this.minX;
		var y = (canvas.height  - ymouse ) / this.scaleY + this.minY;	
		return "x = " + x.toFixed(3) + ", y = " + y.toFixed(3);	
	}
}


/////////////////////////////////
// Define Object class "Plot2D"
function Plot2D(canvasId, tableId) {
	
	if(typeof(canvasId) === 'undefined' ) 
		this.canvasId = "plotcanvas2D";
	else
		this.canvasId = canvasId;
		
	if(typeof(tableId) === 'undefined' ) 
		this.tableId = "";	// No data table by default
	else
		this.tableId = tableId;

	this.minX1 = -10;
	this.maxX1 = 10;
	this.minX2 = -10;
	this.maxX2 = 10;
	this.scaleX1 ;
	this.scaleX2 ;
	this.NsamplesX1 = 500;
	this.NsamplesX2 = 500;

	// Training set 2D
	this.Xapp = new Array();
	this.Yapp = new Array();
	this.m = 0;
	
	////// Cross browser support ////
	var ctx = document.getElementById(this.canvasId).getContext("2d");
	if ( !ctx.setLineDash ) {
		ctx.setLineDash = function () {};
	}

}
	 


Plot2D.prototype.clear = function () {
	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		/* put this into setscale2D : */
		this.scaleX1 = canvas.width / (this.maxX1 - this.minX1); 
		this.scaleX2 = canvas.height / (this.maxX2 - this.minX2);
	
		this.NsamplesX1 = canvas.width / 4;
		this.NsamplesX2 = canvas.height / 4;		
	
		/////
		
		ctx.fillStyle = "white";
		ctx.fillRect (0,0 , canvas.width, canvas.height);

		ctx.strokeStyle = "black";	
		ctx.lineWidth = "1";		
	
		if (this.minX2 < 0 && this.maxX2 > 0) {
	
			// X1-axis
			ctx.beginPath();
			ctx.moveTo(0,canvas.height + this.minX2  * this.scaleX2);
			ctx.lineTo(canvas.width,canvas.height + this.minX2  * this.scaleX2);
			ctx.closePath();
			ctx.stroke();
		}
	
		if (this.minX1 < 0 && this.maxX1 > 0) {
			// X2-axis
			ctx.beginPath();
			ctx.moveTo(( -this.minX1 ) * this.scaleX1 ,0);
			ctx.lineTo(( -this.minX1 ) * this.scaleX1 ,canvas.height);
			ctx.closePath();
			ctx.stroke();
		
		}
	
	}
	
	//this.clearData();
}

Plot2D.prototype.clearData = function () {
	if( this.tableId  != "" )
		document.getElementById(this.tableId).innerHTML = "<tr> <td> x1 </td><td> x2 </td><td> y </td></tr> ";
		
	while(this.Yapp.length > 0) {
		this.Yapp.pop();
		this.Xapp.pop();
	}
	this.m = 0;
}

Plot2D.prototype.levelcurve = function (f, level ) {

	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		var started = false; 

		ctx.fillStyle = "rgb(0,0,200)";
		ctx.strokeStyle = "rgb(0,0,200)";
		//ctx.lineWidth="3";
		//ctx.beginPath();
		
		var Y = new Array();
		var i = 0;
		var j = 0;
		
		// Compute function values
		for(var x1=this.minX1; x1 < this.maxX1; x1 += (this.maxX1-this.minX1) / this.NsamplesX1 ) {
			Y[i] = new Array(); 
			for(var x2=this.minX2; x2 < this.maxX2; x2 += (this.maxX2-this.minX2) / this.NsamplesX2 ) {
				var x = [x1, x2];
				Y[i][j] =  f(x) ;
				j++;
			}
			i++;
		}			

		// Draw level curve
		var i = 0;
		var j = 0;
		for(var x1=this.minX1; x1 < this.maxX1; x1 += (this.maxX1-this.minX1) / this.NsamplesX1 ) {
			for(var x2=this.minX2; x2 < this.maxX2; x2 += (this.maxX2-this.minX2) / this.NsamplesX2 ) {
		
				if ( ( j > 0 && Y[i][j] >= level && Y[i][j-1] <= level ) 
					|| ( j > 0 && Y[i][j] <= level && Y[i][j-1] >= level )  
					|| ( i > 0 && Y[i][j] <= level && Y[i-1][j] >= level )  
					|| ( i > 0 && Y[i][j] >= level && Y[i-1][j] <= level )  )	{
				
					/*
					if ( !started ){						
						 ctx.moveTo(( x1-this.minX1 ) * this.scaleX1, canvas.height/2 - ( x2 * this.scaleX2 ));
						 started = true;
					}
					else
						ctx.lineTo(( x1-this.minX1 ) * this.scaleX1 , canvas.height/2 - ( x2 * this.scaleX2 ));
					*/
					ctx.fillRect (( x1-this.minX1 ) * this.scaleX1 - 2, canvas.height - ( ( x2 - this.minX2) * this.scaleX2 ) - 2, 4, 4);
					
				}
				j++;
			}
			
			i++;
		}
//		ctx.closePath();
		//ctx.stroke();
	
	}
}


Plot2D.prototype.colormap = function(f) {

	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		var started = false; 

		var maxf = -Infinity;
		var minf = +Infinity;
		var Y = new Array();
		var i = 0;
		var j = 0;
		
		// Compute function values
		for(var x1=this.minX1; x1 < this.maxX1; x1 += (this.maxX1-this.minX1) / this.NsamplesX1 ) {
			Y[i] = new Array(); 
			for(var x2=this.minX2; x2 < this.maxX2; x2 += (this.maxX2-this.minX2) / this.NsamplesX2 ) {
				var x = [x1, x2];
				Y[i][j] =  f(x) ;
				if(Y[i][j] > maxf ) {
					maxf = Y[i][j];
				}
				if(Y[i][j] < minf ) {
					minf = Y[i][j];
				}
				j++;
			}
			i++;
		}			
		
		
		var colorScale = 255 / (maxf - minf); 

		// Draw colormap
		var i = 0;
		var j = 0;
		for(var x1=this.minX1; x1 < this.maxX1; x1 += (this.maxX1-this.minX1) / this.NsamplesX1 ) {
			for(var x2=this.minX2; x2 < this.maxX2; x2 += (this.maxX2-this.minX2) / this.NsamplesX2 ) {
				if (Math.abs(Y[i][j] ) < 0.00001  ) {
					ctx.fillStyle = "black";
				}
				
				else if (Y[i][j] < 0 ) {
					ctx.fillStyle = "rgba(0,0," + (255 - Math.floor((Y[i][j] - minf) * colorScale )) + ", 0.9)";					
				}
				else {
					//ctx.fillStyle = "rgba(" + Math.floor((Y[i][j] - minf) * colorScale ) + ",0,255,0.5)";
					ctx.fillStyle = "rgba(" + Math.floor((Y[i][j] - minf) * colorScale ) + ",0,0, 0.9)";
				}
				ctx.fillRect (( x1-this.minX1 ) * this.scaleX1 - 2, canvas.height - ( ( x2 - this.minX2) * this.scaleX2 )- 2, 4, 4);		
				//margin
				if (Math.abs(Y[i][j] ) < 1 ) {
					ctx.fillStyle = "rgba(200,200,200,0.5)";
					ctx.fillRect (( x1-this.minX1 ) * this.scaleX1 - 2, canvas.height - ( ( x2 - this.minX2) * this.scaleX2 )- 2, 4, 4);	
				}			
				
				j++;
			}			
			i++;
		}
	
	}
}

Plot2D.prototype.point = function (x1, x2, color_idx, opacity,  radius ) {

	if (typeof(opacity) === 'undefined')
		opacity = 1.1; 
	if (typeof(radius) === 'undefined')
		radius = 5; 
	

	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		if (opacity < 1.0 ) 
			setcolortransparent(ctx, color_idx, opacity);
		else
			setcolor(ctx, color_idx);
		
		ctx.beginPath();
		ctx.arc( ( x1-this.minX1 ) * this.scaleX1 , canvas.height - ( x2 - this.minX2) * this.scaleX2, radius, 0, 2 * Math.PI , true);
		// arc( x, y, radius, agnlestart, angleend, sens)
	
		ctx.closePath();
		ctx.fill();
	}
}


Plot2D.prototype.pointmouse = function  (event ) {

	if(event.button == 0) {
		event.preventDefault();	
	
		var canvas = document.getElementById(this.canvasId);
		if (canvas.getContext) {
			var ctx = canvas.getContext("2d");
			var rect = canvas.getBoundingClientRect();

			var x = event.clientX - rect.left;	// mouse coordinates relative to plot
			var y = event.clientY - rect.top;
			var color_idx = parseInt(document.getElementById("selectcolor").value);
			
			// Add to training set
			var etiquette = color_idx; 
			var x1 = x / this.scaleX1 + this.minX1;
			var x2 =  (canvas.height  - y) / this.scaleX2 + this.minX2;
			// plot point		

			this.point(x1,x2,color_idx );	
		
			this.Xapp[this.m] = new Array(); 
			this.Xapp[this.m][0] = x1;
			this.Xapp[this.m][1] = x2; 
			this.Yapp[this.m] = etiquette; 
			this.m++;
		
		
			if ( this.tableId != "" ) {
				// add to table of points
				var t = document.getElementById(this.tableId); 
				t.innerHTML += "<tr><td>"+ x1.toFixed(2) + "</td><td>" + x2.toFixed(2) + "</td><td>" + etiquette + "</td></tr>";
			}
		}
	}
}

Plot2D.prototype.plot_data = function () {
	if (this.m != this.Yapp.length )
		this.m = this.Yapp.length;
		
   	for(var i=0;i < this.m ;i++) {
		this.point (this.Xapp[i][0], this.Xapp[i][1], this.Yapp[i] );
	}
}

Plot2D.prototype.plot_vector = function(start_x1,start_x2, end_x1,end_x2, vectorname, veccolor) {
	if(typeof(veccolor) === 'undefined') {
		veccolor = 0;
	}
	
	start_x1 = (start_x1 - this.minX1) * this.scaleX1;
	end_x1 = (end_x1 - this.minX1) * this.scaleX1;	
	start_x2 = (start_x2 - this.minX2) * this.scaleX2;
	end_x2 = (end_x2 - this.minX2) * this.scaleX2;	
	
	var theta1 = Math.atan((end_x2 - start_x2)/(end_x1 - start_x1)); // angle entre vecteur et axe X1
	var theta2 = Math.atan((end_x1 - start_x1) /(end_x2 - start_x2)); // angle entre vecteur et axe X2	
	
	var arrowsize = 10;
	var arrow1_x1 = end_x1 ; 
	var arrow1_x2 = end_x2 ; 
	
	var arrow2_x1 = end_x1 ;
	var arrow2_x2 = end_x2 ;
	
	if ( end_x2 >= start_x2) {
		arrow1_x1 -= arrowsize*Math.sin(theta2 - Math.PI/12);
		arrow1_x2 -= arrowsize*Math.cos(theta2 - Math.PI/12);
	}
	else {
		arrow1_x1 += arrowsize*Math.sin(theta2 - Math.PI/12);
		arrow1_x2 += arrowsize*Math.cos(theta2 - Math.PI/12);
	}		
	if ( end_x1 >= start_x1 ) {
		arrow2_x1 -= arrowsize*Math.cos(theta1 - Math.PI/12);	
		arrow2_x2 -= arrowsize*Math.sin(theta1 - Math.PI/12);			
	}
	else {
		arrow2_x1 += arrowsize*Math.cos(theta1 - Math.PI/12);	
		arrow2_x2 += arrowsize*Math.sin(theta1 - Math.PI/12);		
	}
	
	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		ctx.lineWidth="1";
		setcolor(ctx,veccolor);
		
		ctx.beginPath();
		ctx.moveTo(start_x1,canvas.height - start_x2);
		ctx.lineTo(end_x1,canvas.height - end_x2);
		ctx.lineTo(arrow1_x1,canvas.height - arrow1_x2);
		ctx.lineTo(arrow2_x1,canvas.height - arrow2_x2);
		ctx.lineTo(end_x1,canvas.height - end_x2);
		ctx.stroke();
		ctx.fill();

		if(typeof(vectorname) !== 'undefined') {
			ctx.lineWidth="1";
			var dx =5;
			if ( end_x1 < start_x1)
				dx = -15;

			ctx.strokeText(vectorname, end_x1 + dx,canvas.height - end_x2);
		}
	}
}


Plot2D.prototype.plot_line = function(start_x1,start_x2, end_x1,end_x2, linename, linecolor, dashed, linewidth) {
	if(typeof(linecolor) === 'undefined') {
		linecolor = 0;
	}
	if(typeof(dashed) === 'undefined') {
		dashed = false;
	}
	if(typeof(linewidth) === 'undefined') {
		linewidth = 1;
	}
	
	start_x1 = (start_x1 - this.minX1) * this.scaleX1;
	end_x1 = (end_x1 - this.minX1) * this.scaleX1;	
	start_x2 = (start_x2 - this.minX2) * this.scaleX2;
	end_x2 = (end_x2 - this.minX2) * this.scaleX2;	
	
	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		ctx.lineWidth=""+linewidth;
		setcolor(ctx,linecolor);
		if (dashed) {
			ctx.setLineDash([5]);
			//ctx.lineWidth="1";
		}
		ctx.beginPath();
		ctx.moveTo(start_x1,canvas.height - start_x2);
		ctx.lineTo(end_x1,canvas.height - end_x2);
		ctx.stroke();
		ctx.setLineDash([1, 0]);
		
		if(typeof(linename) !== 'undefined') {
			if ( linename != "" ) {
				ctx.lineWidth="1";
				ctx.strokeText(linename, (end_x1 + start_x1)/2 - 10 ,canvas.height - 10 - (end_x2+start_x2)/2);
			}
		}
	}
}
Plot2D.prototype.plot_classifier = function (w, b, coloridx, disappear) {
	if (typeof(disappear) === 'undefined')
		var disappear = false; 
	if (typeof(coloridx) === 'undefined')
		var coloridx = 0; 

	var x1 = this.minX1;
	var x2 = this.maxX1;
	var y1;
	var y2;
	
	if (w[1] != 0) {
		y1 = (-b - w[0]*x1) / w[1];
		y2 = (-b - w[0]*x2) / w[1];
	}
	else {
		x1 = -b / w[0];
		x2 = -b / w[0];
		y1 = this.minX2;
		y2 = this.maxX2;
	}
	
	var canvas = document.getElementById(plot.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
	
		ctx.lineWidth="3";
		if ( disappear ) 
			ctx.strokeStyle = "grey";
		else
			setcolor(ctx, coloridx);
			
		ctx.setLineDash([1, 0]);
		ctx.beginPath();		
		ctx.moveTo(( x1-this.minX1 ) * this.scaleX1 , canvas.height/2 - ( y1 * this.scaleX2 ));
		ctx.lineTo(( x2-this.minX1 ) * this.scaleX1 , canvas.height/2 - ( y2 * this.scaleX2 ));
		ctx.stroke();


	}

	
}

Plot2D.prototype.coord2datavector = function(x1,x2) {
	var canvas = document.getElementById(this.canvasId); 
	var x = [0,0] ;
	x[0] = ( x1 / this.scaleX1 ) + this.minX1 ;
	x[1] = (-( x2-canvas.height) / this.scaleX2 ) + this.minX2;
	return x;
}
Plot2D.prototype.plotmathjax = function(stringindex, x, y) {
			
	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");

		var label = document.getElementById("jaxstring"+stringindex);
		label.style.top = canvas.height - ( y - this.minX2) * this.scaleX2  + canvas.offsetTop;
		label.style.left = (x - this.minX1) * this.scaleX1 + canvas.offsetLeft;	
		label.style.visibility = "visible"; 
	}
}
Plot2D.prototype.clearmathjax = function(stringindex) {
	var label = document.getElementById("jaxstring"+stringindex);
	label.style.visibility = "hidden"; 
	
}
	 
Plot2D.prototype.text = function (x1,x2,txt) {
	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		ctx.lineWidth="0.5"; 
		ctx.strokeStyle = "black";
		ctx.strokeText(txt,  ( x1-this.minX1 ) * this.scaleX1 , canvas.height - ( x2 - this.minX2) * this.scaleX2);
	}
}

///////////////////////////////
/// Plot 3D ///////////////////
///////////////////////////////
function Plot3D(canvasId) {
	
	if(typeof(canvasId) === 'undefined' ) 
		this.canvasId = "plotcanvas3D";
	else
		this.canvasId = canvasId;

	var canvas =  document.getElementById(this.canvasId);

	this.minX1 = -10;
	this.maxX1 = 10;
	this.minX2 = -10;
	this.maxX2 = 10;
	this.minX3 = -10;
	this.maxX3 = 10;
	this.scaleX1 ;
	this.scaleX2 ;
	this.scaleX3 ;
	this.NsamplesX1 = 50;
	this.NsamplesX2 = 50;
	this.NsamplesX3 = 50;	

	this.axisNameX1 = "x1";
	this.axisNameX2 = "x2";
	this.axisNameX3 = "x3";

	// Training set 3D
	this.X = new Array();
	this.Y = new Array();
	
	// other stuff to plot
	this.lines = new Array();
	this.planes = new Array();
	this.spheres = new Array();
	
	// 2D Graphics
	this.view2D = new Array();
	this.viewminX1 = -10;
	this.viewmaxX1 = 10;
	this.viewminX2 = -10;
	this.viewmaxX2 = 10;
	this.viewscaleX1 = canvas.width / (this.viewmaxX1 - this.viewminX1);
	this.viewscaleX2 = canvas.width / (this.viewmaxX2 - this.viewminX2);
	
	this.angleX = 0.0;//- Math.PI/6; // rotations around axis
	this.angleY = 0.0;
	this.angleZ = 0.0;// - Math.PI/8;

	this.cameraDistance = 20;

	// Mouse animation
	this.ROTATING = false;
	this.mouseX = 0;
	this.mouseY = 0;

	// automatic animation
	this.animation = null; 
	this.animateAuto = 0; 	// if 0: do not relaunch animation after mouse released 
							// if > 0: samplingrate of animation
	
	////// Cross browser support ////
	var ctx = document.getElementById(this.canvasId).getContext("2d");
	if ( !ctx.setLineDash ) {
		ctx.setLineDash = function () {};
	}
	
	if(window.addEventListener)
        canvas.addEventListener('DOMMouseScroll', this.mousezoom, false);//firefox
 
    //for IE/OPERA etc
    canvas.onmousewheel = this.mousezoom;

}


Plot3D.prototype.test = function() {
	this.X.push([5,0,0]);
	this.X.push([0,5,0]);
	this.X.push([0,0,5]);
	this.Y.push(1);
	this.Y.push(2);
	this.Y.push(3);
	this.X.push([2,0,0]);
	this.X.push([0,-6,0]);
	this.X.push([0,0,2]);
	this.Y.push(1);
	this.Y.push(2);
	this.Y.push(3);
	
	this.X.push([5,5,5]);
	this.Y.push(3);
	
	
	this.sphere([5,-5, 1], 50, "", 1) ;
	this.sphere([-5,5, -3], 30, "", 3) ;
	
	this.replot();
	this.animateAuto = 100;
	this.animate(50);
}
Plot3D.prototype.computeRanges = function () {

	var i;
	for (i=0;i<this.Y.length ; i++) {
		var norm = Math.sqrt( this.X[i][0]*this.X[i][0] + this.X[i][1]*this.X[i][1] + this.X[i][2]*this.X[i][2] ) 
		if ( norm > this.maxX2 ) {
			this.maxX2 = norm;
			this.minX2 = -norm;
		}
	}	
}	

Plot3D.prototype.replot = function() {
	// Compute 2D coordinates from 3D ones
	
	
	var x1;
	var x2;
	var distance;
	var opacity;
	var radius;
	var res;
	
	var i;	
	var maxDistance = this.cameraDistance + this.maxX2 - this.minX2; 

	this.clear();
	
	// plotorigin
	this.point2D(0, 0, 0 , 1.0, 3 ) ; 
	
	// plot axis
	this.line([ -1, 0, 0], [10, 0,0] , this.axisNameX1);
	this.line([ 0, -1, 0], [0, 10 ,0] ,this.axisNameX2);
	this.line([ 0, 0, -1], [0, 0,10] , this.axisNameX3);
	
	// plot points
	for (i=0;i<this.Y.length ; i++) {
	
		res = this.project(this.X[i] );
		x1 = res[0];
		x2 = res[1];
		distance = res[2];
	
		
		if ( distance < maxDistance ) 
			opacity = ( distance / maxDistance ) ;
		else
			opacity = 1.0;
			
		radius = Math.floor(2 + (1 - opacity) * 10);
				
		this.point2D(x1, x2, this.Y[i] , opacity, radius ) ; 
	}
	
	// plot lines
	for (i=0;i<this.lines.length; i++) {
		this.line(this.lines[i][0],this.lines[i][1],this.lines[i][2],this.lines[i][3]);
	}
	
	// plot planes
	for (i=0;i<this.planes.length; i++) {
		this.drawplane(this.planes[i][0],this.planes[i][1],this.planes[i][2],this.planes[i][3]);
	}
	
	// plot spheres 
	//  plot the most distant ones first !!! 
	var distances = new Array();
	for (i=0;i<this.spheres.length; i++) {	
		var res = this.project( this.spheres[i][0] ); 
		distances[i] = res[2];
	}
	for (var n=0;n<this.spheres.length; n++) {
		var idx = 0;
		for ( i=1; i< this.spheres.length; i++) {
			if ( distances[i] > distances[idx] )
				idx = i;
		}
		this.drawsphere( this.spheres[idx][0], this.spheres[idx][1], this.spheres[idx][2], this.spheres[idx][3] );
		distances[idx] = -1;			
	}
}
Plot3D.prototype.clear = function(  ) {
	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		ctx.clearRect(0,0,canvas.width,canvas.height);
	}
}
Plot3D.prototype.clear_data = function(  ) {
	while (this.Y.length > 0) {
		this.Y.pop();
		this.X.pop();
	}
}
Plot3D.prototype.clear_planes = function(  ) {
	while(this.planes.length > 0) {
		this.planes.pop();
	}
}

Plot3D.prototype.rotateX = function( deltaangle ) {
	this.angleX += deltaangle; 
	this.replot();
}

Plot3D.prototype.rotateY = function( deltaangle ) {
	this.angleY += deltaangle; 
	this.replot();
}
Plot3D.prototype.rotateZ = function( deltaangle , do_replot) {
	if ( typeof(do_replot) == "undefined" )
		var do_replot = true;
		
	this.angleZ += deltaangle; 
	if ( do_replot )
		this.replot();
}

Plot3D.prototype.mouserotation = function(e) {

	if ( this.ROTATING ) {
		e.preventDefault();
		var dx = e.clientX - this.mouseX;	
		var dy = e.clientY - this.mouseY;	
		this.mouseX = e.clientX;
		this.mouseY = e.clientY;
		
		if ( Math.abs(dx) > 0.2 ) 
			this.rotateZ(dx / 20, !(Math.abs(dy) > 0.2) );
		if ( Math.abs(dy) > 0.2 ) 
			this.rotateX(dy / 20);
	}
}
Plot3D.prototype.mousedown = function(e) {
	e.preventDefault();
	this.ROTATING = true;
	this.mouseX = e.clientX;
	this.mouseY = e.clientY;
	
	this.animateStop();
}

Plot3D.prototype.mouseup = function(e) {
	e.preventDefault();
	this.ROTATING = false;
	if ( this.animateAuto > 0 ) {
		this.animate( this.animateAuto );
	}
}
Plot3D.prototype.mousezoom = function(e) {
	// !!! use plot3 instead of this due to event handler...
	
	var delta = 0;
 
    if (!e) 
    	e = window.event;
 	
 	e.preventDefault();
	
    // normalize the delta
    if (e.wheelDelta) {
         // IE and Opera
        delta = e.wheelDelta / 30;
    } 
    else if (e.detail) { 
        delta = -e.detail ;
    }
 
	plot3.cameraDistance -= delta ;
	
	if ( plot3.cameraDistance < 5 ) {
		plot3.cameraDistance = 5;
	}
	else if ( plot3.cameraDistance > 100 ) 
		plot3.cameraDistance = 100;
	
	plot3.replot();
}

Plot3D.prototype.project = function ( x3D ) {
	/*
		x3D : points in World coordinate system
		Camera / view coordinate system initialized like World system
		Camera is fixed to (0,cameraDistance,0) in camera system
		
		1. rotate World in camera system
		2. project camera system to 2D XZ plane since camera on Y-axis
		3. distance to camera = cameraDistance + Y 
		
	
	*/
	
	// 1. rotation
	var tmpX = new Array(3); 
	// rotation around X-axis:
	tmpX[0] = x3D[0]; // does not change X-coordinate
	tmpX[1] = Math.cos(this.angleX) * x3D[1] - Math.sin(this.angleX) * x3D[2];
	tmpX[2] = Math.sin(this.angleX) * x3D[1] + Math.cos(this.angleX) * x3D[2];	
	
	// rotation around Y-axis:
	var tmpY = new Array(3); 
	tmpY[0] = Math.cos(this.angleY) * tmpX[0] - Math.sin(this.angleY) * tmpX[2];
	tmpY[1] = tmpX[1];
	tmpY[2] = Math.sin(this.angleY) * tmpX[0] + Math.cos(this.angleY) * tmpX[2];	

	// rotation around Z-axis:
	var tmpZ = new Array(3); 
	tmpZ[0] = Math.cos(this.angleZ) * tmpY[0] - Math.sin(this.angleZ) * tmpY[1];
	tmpZ[1] = Math.sin(this.angleZ) * tmpY[0] + Math.cos(this.angleZ) * tmpY[1];	
	tmpZ[2] = tmpY[2];
	
	// Scaling
	var scale = ( this.cameraDistance/20 ) ;
	tmpZ[0] /= scale;
	tmpZ[1] /= scale;
	tmpZ[2] /= scale;		
	
	// Project to 2D plane 	
	var x1 = tmpZ[0];
	var x2 = tmpZ[2]; 
	var distance = this.cameraDistance + tmpZ[1];

	return [x1,x2, distance];
}

Plot3D.prototype.line = function( start, end, linename, linecolor, dashed, linewidth ) {
	var start_x1;
	var start_x2;

	var res = this.project(start);
	start_x1 = res[0];
	start_x2 = res[1];
	
	var end_x1;
	var end_x2;

	res = this.project(end);
	end_x1 = res[0];
	end_x2 = res[1];
	
	this.line2D(start_x1, start_x2, end_x1, end_x2, linename, linecolor, dashed, linewidth);
}
Plot3D.prototype.plot_line = function( start, end, linename, color ) {
	if (typeof(color) === 'undefined')
		var color = 0; 

	this.lines.push([start, end, linename, color]);
	this.line( start, end, linename, color );
}
Plot3D.prototype.plane = function( start, end, polyname, color ) {
	if (typeof(color) === 'undefined')
		var color = 3; 
	if (typeof(polyname) === 'undefined')
		var polyname = "";
	
	this.planes.push([start, end, polyname, color]);
	this.drawplane( start, end, polyname, color );
}
Plot3D.prototype.drawplane = function( start, end, polyname, color ) {
	var res;
	var corner1 = new Array(3);// 2 other corners
	var corner2 = new Array(3);
	corner1[0] = start[0];
	corner1[1] = end[1];
	corner1[2] = start[2];
	corner2[0] = end[0];
	corner2[1] = start[1];
	corner2[2] = end[2];
	
	res = this.project(start);		
	var start_x1 = res[0];
	var start_x2 = res[1];
	
	res = this.project(end);		
	var end_x1 = res[0];
	var end_x2 = res[1];
	
	res = this.project(corner1);		
	var corner1_x1 = res[0];
	var corner1_x2 = res[1];
 	res = this.project(corner2);		
	var corner2_x1 = res[0];
	var corner2_x2 = res[1];
 			
	this.polygone2D( [ start_x1, corner1_x1, end_x1, corner2_x1], [ start_x2, corner1_x2, end_x2, corner2_x2], polyname, color);
}

Plot3D.prototype.sphere = function( center, radius, spherename, color ) {
	if (typeof(color) === 'undefined')
		var color = 1; 
	if (typeof(spherename) === 'undefined')
		var spherename = "";
	this.spheres.push([center, radius, spherename, color ]);
	this.drawsphere( center, radius, spherename, color );
}
Plot3D.prototype.drawsphere = function( center, radius, spherename, color ) {
	var res;
	res = this.project(center);		
	var x1 = res[0];
	var x2 = res[1];
	var distance = res[2];
	
	if ( distance >= 0 ) {
		var opacity = 1.0;
		var maxDistance = this.cameraDistance + this.maxX2 - this.minX2; 
	
		if ( distance < maxDistance ) 
			opacity = 0.5 * ( distance / maxDistance ) ;

		var radius2D = Math.floor(radius * ( 0 +3* (1 - opacity)*(1 - opacity) ) );

		this.disk2D( x1, x2, radius2D, spherename, color, opacity);
	
	}
}

Plot3D.prototype.point2D = function (x1, x2, color_idx, opacity,  radius ) {

	if ( x1 >= this.viewminX1 && x1 <= this.viewmaxX1 && x2 >= this.viewminX2 && x2 <= this.viewmaxX2 ) {

		if (typeof(opacity) === 'undefined')
			var opacity = 1.1; 
		if (typeof(radius) === 'undefined')
			var radius = 5; 
	

		var canvas = document.getElementById(this.canvasId);
		if (canvas.getContext) {
			var ctx = canvas.getContext("2d");
		
			if (opacity < 1.0 ) 
				setcolortransparent(ctx, color_idx, opacity);
			else
				setcolor(ctx, color_idx);
		
			ctx.beginPath();
			ctx.arc( ( x1-this.viewminX1 ) * this.viewscaleX1 , canvas.height - ( x2 - this.viewminX2) * this.viewscaleX2, radius, 0, 2 * Math.PI , true);
			// arc( x, y, radius, agnlestart, angleend, sens)
	
			ctx.closePath();
			ctx.fill();
		}
	}
}
Plot3D.prototype.line2D = function(start_x1,start_x2, end_x1,end_x2, linename, linecolor, dashed, linewidth) {
	if(typeof(linecolor) === 'undefined') {
		linecolor = 0;
	}
	if(typeof(dashed) === 'undefined') {
		dashed = false;
	}
	if(typeof(linewidth) === 'undefined') {
		linewidth = 1;
	}
	
	start_x1 = (start_x1 - this.viewminX1) * this.viewscaleX1;
	end_x1 = (end_x1 - this.viewminX1) * this.viewscaleX1;	
	start_x2 = (start_x2 - this.viewminX2) * this.viewscaleX2;
	end_x2 = (end_x2 - this.viewminX2) * this.viewscaleX2;	
	
	
	var canvas = document.getElementById(this.canvasId);

	if ( start_x1 < 0 ) 
			start_x1 = 0;
	if ( start_x1 >= canvas.width ) 
			start_x1 = canvas.width-1;
	if ( start_x2 <= 0 ) 
			start_x2 = 1;
	if ( start_x2 > canvas.height ) 
			start_x2 = canvas.height;
	if ( end_x1 < 0 ) 
			end_x1 = 0;
	if ( end_x1 >= canvas.width ) 
			end_x1 = canvas.width-1;
	if ( end_x2 <= 0 ) 
			end_x2 = 1;
	if ( end_x2 > canvas.height ) 
			start_x2 = canvas.height;

	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		ctx.lineWidth=""+linewidth;
		setcolor(ctx,linecolor);
		if (dashed) {
			ctx.setLineDash([5]);
			//ctx.lineWidth="1";
		}
		ctx.beginPath();
		ctx.moveTo(start_x1,canvas.height - start_x2);
		ctx.lineTo(end_x1,canvas.height - end_x2);
		ctx.stroke();
		if (dashed) {
			ctx.setLineDash([1, 0]);
		}
		
		if(typeof(linename) !== 'undefined') {
			if (linename != "") {
				var x = -10 + (end_x1 + start_x1)/2 ;
				var y = canvas.height + 10 - (end_x2 + start_x2)/2 ; 
					
				if (linename.indexOf("jaxstring") == 0 ) {
					// put mathjaxstring as line name
					var label = document.getElementById(linename);
					label.style.fontSize = "70%";
					label.style.top = y + canvas.offsetTop;
					label.style.left = x + canvas.offsetLeft;	
					label.style.visibility = "visible"; 
				}
				else {
					ctx.lineWidth="1";
					ctx.strokeText(linename, x, y );
				}
			}
		}
	}
}

Plot3D.prototype.polygone2D = function(x1,x2, polyname, color) {
	/*
		x1,x2 : arrayx of X1,X2 coordinates of all points
	*/

	if(typeof(color) === 'undefined') {
		color = 3;
	}
	
	var i;
	// loop over all points:
	
	for (i=0;i<x1.length;i++) {
		x1[i] = (x1[i] - this.viewminX1) * this.viewscaleX1;	
		x2[i] = (x2[i] - this.viewminX2) * this.viewscaleX2;
	}
	
	var canvas = document.getElementById(this.canvasId);
	if (canvas.getContext) {
		var ctx = canvas.getContext("2d");
		
		
		setcolortransparent(ctx,color, 0.5);
		
		ctx.beginPath();
		ctx.moveTo(x1[0],canvas.height - x2[0]);
		for (i=0;i<x1.length;i++) {
			ctx.lineTo( x1[i],canvas.height - x2[i]);
		}
		ctx.fill();
		
		if(typeof(polyname) !== 'undefined') {
			if (polyname != "") {
				var x = -10 + x1[0];
				var y = canvas.height + 10 - x2[0];
					
				if (polyname.indexOf("jaxstring") == 0 ) {
					// put mathjaxstring as line name
					var label = document.getElementById(polyname);
					label.style.fontSize = "70%";
					label.style.top = y + canvas.offsetTop;
					label.style.left = x + canvas.offsetLeft;	
					label.style.visibility = "visible"; 
				}
				else {
					ctx.lineWidth="1";
					ctx.strokeText(polyname, x, y );
				}
			}
		}
	}
}

Plot3D.prototype.disk2D = function (x1, x2, radius, spherename, color, opacity ) {
	if (typeof(opacity) === 'undefined')
		var opacity = 1.1; 
	if (typeof(radius) === 'undefined')
		var radius = 5; 
	
	if ( x1 + radius >= this.viewminX1 && x1 - radius <= this.viewmaxX1 && x2 + radius >= this.viewminX2 && x2 - radius <= this.viewmaxX2 ) {
		
		var canvas = document.getElementById(this.canvasId);
		if (canvas.getContext) {
			var ctx = canvas.getContext("2d");
			var x1view =  ( x1-this.viewminX1 ) * this.viewscaleX1 ;
			var x2view =  canvas.height - ( x2 - this.viewminX2) * this.viewscaleX2;
		
			if (opacity < 1.0 ) 
				setcolortransparentgradient(ctx, color, opacity, Math.sqrt(x1view*x1view+x2view*x2view) + radius);
			else
				setcolorgradient(ctx, color,  Math.sqrt(x1view*x1view+x2view*x2view) + radius);
		
			ctx.beginPath();
			ctx.arc( x1view, x2view, radius, 0, 2 * Math.PI , true);
			ctx.closePath();
			ctx.fill();
			
			if(typeof(spherename) !== 'undefined') {
				if (spherename != "") {
					var x = -10 + x1view;
					var y = 10 + x2view;
					
					if (spherename.indexOf("jaxstring") == 0 ) {
						// put mathjaxstring as line name
						var label = document.getElementById(spherename);
						label.style.fontSize = "70%";
						label.style.top =  x2view  + canvas.offsetTop;
						label.style.left = x1view  + canvas.offsetLeft;	
						label.style.visibility = "visible"; 
					}
					else {
						var words = spherename.split("*");
						ctx.textAlign = "center";	// center of text appear at x position
						var txtsize = Math.floor(0.2 * radius) ;
						var tmpfont = ctx.font;
						ctx.font = txtsize + "pt sans-serif";
						ctx.fillStyle = "black";
		
						if ( words.length == 1 ) {
							ctx.fillText( spherename, x1view  , x2view  ) ;
						}
						else {
							for (var i = 0; i< words.length; i++) {
								ctx.fillText( words[i], x1view  ,x2view - (words.length/2 - i - 0.5)* (1.5 * txtsize)) ;
							}
						}
						ctx.font = tmpfont;
					}
				}
			}
		}
	}
}
Plot3D.prototype.animate = function(samplingRate) {
	if ( typeof(samplingRate) === 'undefined' ) 
		var samplingRate = this.animateAuto ;
		
		
	this.animateStop(); // make sure a single animation runs
		
	var p3 = this;
	this.animation = setInterval( function () {
			p3.rotateZ(0.01);	// cannot use "this" here => plot3
		}, samplingRate
	);
	
}
Plot3D.prototype.animateStop = function() {
	if ( this.animation != null ) {
		clearInterval( this.animation );
		this.animation = null;
	}
	
}
Plot3D.prototype.isInSphere = function (x, y, z, sphere) {
	var dx = (x - sphere[0][0]);
	var dy = (y - sphere[0][1]);
	var dz = (z - sphere[0][2]);		
	var norm2 = dx*dx+dy*dy+dz*dz;
	
	if ( norm2 <= sphere[1]*sphere[1] )
		return true;
	else 
		return false;

}

////////////////////////////////////////
// General canvas tools 	 
function setcolor(ctx, color_idx) {
	if( color_idx > 10 ) {
		setcolortransparent(ctx, color_idx - 10, 0.5);
		return;
	}
	
	switch(color_idx) {
	case -1: 
		ctx.fillStyle = "white";			
		ctx.strokeStyle = "white";
		break;

	case 0: 
		ctx.fillStyle = "rgb(0,0,0)";
		ctx.strokeStyle = "rgb(0,0,0)";
		break;
	case 1: 
		ctx.fillStyle = "rgb(0,0,200)";	
		ctx.strokeStyle = "rgb(0,0,200)";					
		break;
	case 2: 
		ctx.fillStyle = "rgb(200,0,0)";			
		ctx.strokeStyle = "rgb(200,0,0)";			
		break;
	case 3: 
		ctx.fillStyle = "rgb(0,200,0)";
		ctx.strokeStyle = "rgb(0,200,0)";						
		break;
	case 4: 
		ctx.fillStyle = "rgb(200,0,200)";			
		ctx.strokeStyle = "rgb(200,0,200)";
		break;
	case 5: 
		ctx.fillStyle = "rgb(255,255,0)";			
		ctx.strokeStyle = "rgb(255,255,0)";
		break;
	case 6: 
		ctx.fillStyle = "rgb(0,255,255)";			
		ctx.strokeStyle = "rgb(0,255,255)";
		break;
	case 7: 
		ctx.fillStyle = "rgb(102,51,0)";			
		ctx.strokeStyle = "rgb(102,51,0)";
		break;
	case 8: 
		ctx.fillStyle = "rgb(204,51,0)";			
		ctx.strokeStyle = "rgb(204,51,0)";
		break;
	case 9: 
		ctx.fillStyle = "rgb(255,102,204)";			
		ctx.strokeStyle = "rgb(255,102,204)";
		break;
	case 10: 
		ctx.fillStyle = "rgb(120,120,120)";			
		ctx.strokeStyle = "rgb(120,120,120)";
		break;
	
	default:
		ctx.fillStyle = "rgb(0,0,200)";			
		ctx.strokeStyle = "rgb(0,0,200)";			
		break;															
	}

}
	 
function setcolortransparent(ctx, color_idx, opacity) {
	switch(color_idx) {
	case 0: 
		ctx.fillStyle = "rgba(0,0,0," + opacity +" )";
		ctx.strokeStyle = "rgba(0,0,0," + opacity +" )";
		break;
	case 1: 
		ctx.fillStyle = "rgba(0,0,200," + opacity +" )";
		ctx.strokeStyle = "rgba(0,0,200," + opacity +" )";
		break;
	case 2: 
		ctx.fillStyle = "rgba(200,0,0," + opacity +" )";
		ctx.strokeStyle = "rgba(200,0,0," + opacity +" )";
		break;
	case 3: 
		ctx.fillStyle = "rgba(0,200,0," + opacity +" )";
		ctx.strokeStyle = "rgba(0,200,0," + opacity +" )";
		break;
	case 4: 
		ctx.fillStyle = "rgba(200,0,200," + opacity +" )";
		ctx.strokeStyle = "rgba(200,0,200," + opacity +" )";
		break;
	case 5: 
		ctx.fillStyle = "rgba(255,255,0," + opacity +" )";
		ctx.strokeStyle = "rgba(255,255,0," + opacity +" )";
		break;
	case 6: 
		ctx.fillStyle = "rgba(0,255,255," + opacity +" )";		
		ctx.strokeStyle = "rgba(0,255,255," + opacity +" )";
		break;
	case 7: 
		ctx.fillStyle = "rgba(102,51,0," + opacity +" )";			
		ctx.strokeStyle = "rgba(102,51,0," + opacity +" )";
		break;
	case 8: 
		ctx.fillStyle = "rgba(204,51,0," + opacity +" )";			
		ctx.strokeStyle = "rgba(204,51,0," + opacity +" )";
		break;
	case 9: 
		ctx.fillStyle = "rgab(255,102,204," + opacity +" )";		
		ctx.strokeStyle = "rgba(255,102,204," + opacity +" )";
		break;
	case 10: 
		ctx.fillStyle = "rgba(120,120,120," + opacity +" )";	
		ctx.strokeStyle = "rgba(120,120,120," + opacity +" )";
		break;
	default:
		ctx.fillStyle = "rgba(0,0,200," + opacity +" )";
		ctx.strokeStyle = "rgba(0,0,200," + opacity +" )";
		break;															
	}

}

function setcolorgradient(ctx, color_idx,size) {
	if ( typeof(size) === "undefined")
		var size = 400 * Math.sqrt(2);
		
	var gradient = ctx.createRadialGradient(0,0,size, 0 , 2*Math.PI, true);
	gradient.addColorStop(1,"white");
	
	switch(color_idx) {
	case 0: 
		gradient.addColorStop(0,"rgb(0,0,0 )");
		break;
	case 1: 
		gradient.addColorStop(0,"rgb(0,0,200 )");
		break;
	case 2: 
		gradient.addColorStop(0,"rgb(200,0,0 )");
		break;
	case 3: 
		gradient.addColorStop(0,"rgb(0,200,0 )");
		break;
	case 4: 
		gradient.addColorStop(0,"rgb(200,0,200 )");
		break;
	case 5: 
		gradient.addColorStop(0,"rgb(255,255,0 )");		
		break;
	default:
		gradient.addColorStop(0,"rgb(0,0,200 )");
		break;
													
	}
	ctx.fillStyle = gradient;
}
	 
function setcolortransparentgradient(ctx, color_idx, opacity,size) {
	if ( typeof(size) === "undefined")
		var size = 400 * Math.sqrt(2);
		
	var gradient = ctx.createRadialGradient(0,0,size, 0 , 2*Math.PI, true);
	gradient.addColorStop(1,"white");
	
	switch(color_idx) {
	case 0: 
		gradient.addColorStop(0.3,"rgba(0,0,0," + opacity +" )");		
		gradient.addColorStop(0,"rgb(0,0,0 )");
		break;
	case 1: 
		gradient.addColorStop(0.3,"rgba(0,0,200," + opacity +" )");
		gradient.addColorStop(0,"rgb(0,0,200 )");
		break;
	case 2: 
		gradient.addColorStop(0.3,"rgba(200,0,0," + opacity +" )");
		gradient.addColorStop(0,"rgb(200,0,0 )");
		break;
	case 3: 
		gradient.addColorStop(0.3,"rgba(0,200,0," + opacity +" )");
		gradient.addColorStop(0,"rgb(0,200,0 )");
		break;
	case 4: 
		gradient.addColorStop(0.3,"rgba(200,0,200," + opacity +" )");
		gradient.addColorStop(0,"rgb(200,0,200 )");
		break;
	case 5: 
		gradient.addColorStop(0.3,"rgba(255,255,0," + opacity +" )");
		gradient.addColorStop(0,"rgb(255,255,0 )");		
		break;
	default:
		gradient.addColorStop(0.3,"rgba(0,0,200," + opacity +" )");
		gradient.addColorStop(0,"rgb(0,0,200 )");
		break;
													
	}
	ctx.fillStyle = gradient;
}


