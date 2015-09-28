// Globals for HElP content
var htmlspace = "&nbsp;";
var htmltab = htmlspace+htmlspace+htmlspace+htmlspace; 
var htmlmatrix = "<a href='lalolib.html#matrix' target='_blank'>matrix</a>"; 
var htmlvector = "<a href='lalolib.html#vector' target='_blank'>vector</a>";
var HELPcontent = new Array();
var pathToBook = "../book/en/";

//////////////////////////
/// Help functions
//////////////////////////
var helplocked = false;

function help(cmd , forceshow ) {
	if ( !helplocked || forceshow ) {
		if ( cmd == "transpose")
			cmd = "' (transpose)";
		var cmdhelp = HELPcontent[cmd]; 
		if ( cmdhelp ) {
			var htmlhelp = "<h2>" + cmdhelp[1] + "</h2>";
			htmlhelp += "<h4>" + cmdhelp[2].split("\n").join("<br>") + "</h4>";
			htmlhelp += "<p class='helpcontent'>" + cmdhelp[3] + "</p>";
		
			if ( cmdhelp[5] ) {
				if ( typeof(cmdhelp[5]) == "string" ) {
					htmlhelp += "<p class='helpcontent'><a target='_blank' href='" + pathToBook + cmdhelp[5] + "'>More information on this topic.</a></p>";	
				}
				else {
					htmlhelp += "<p class='helpcontent'>For more information on this topic, see <a target='_blank' href='" + pathToBook  + cmdhelp[5][0] + "'>" + cmdhelp[5][0] + "</a>"; 
					for ( var l = 1 ; l < cmdhelp[5].length; l++) {
						htmlhelp += ", <a target='_blank' href='" + pathToBook  + cmdhelp[5][1] + "'>" + cmdhelp[5][1] + "</a>" ; 
					}
					htmlhelp += "</p>";
				}
			}
		
			if(  cmdhelp[4] ) {
				if ( typeof( cmdhelp[4] ) == "string" ) {
					// single example
					htmlhelp += "<strong>Example:</strong><pre>" +  cmdhelp[4] + "</pre>";
					ExampleClipboard[0] = cmdhelp[4];
						if (!self.hasOwnProperty("__INLALOLIBHTMLPAGE")) 
							htmlhelp += "<button id='copyexampletocmdblock' class='topbuttons' onclick='cmdblock.value=ExampleClipboard[0];'>Copy to multi-line input</button>";
				}
				else {
					// more examples
					for ( var l = 0; l < cmdhelp[4].length; l++) {
						htmlhelp += "<strong>Example " + (l+1) + ":</strong><pre>" +  cmdhelp[4][l] + "</pre>";
						ExampleClipboard[l] = cmdhelp[4][l];
						if (!self.hasOwnProperty("__INLALOLIBHTMLPAGE")) 
							htmlhelp += "<button class='topbuttons' onclick='cmdblock.value=ExampleClipboard[" + l + "];'>Copy to multi-line input</button><br><br>";
					}
				}
			}
				
			helpview.innerHTML = htmlhelp;
		
			if ( domathjax && typeof(MathJax) != "undefined" && MathJax.isReady) {
				helpview.focus(); 
				helpview.blur();
				MathJax.Hub.Queue(["Typeset",MathJax.Hub,helpview]);		
			}
		}
		
		if ( forceshow ) 
			helplocked = true;
	}
}
function populateHelp() {
	var c;
	var category;
	var sections = ["Welcome", "Basics"]; // default sections
	for ( c in HELPcontent ) {
		if (HELPcontent.hasOwnProperty(c) )  {
			
			if ( sections.indexOf(HELPcontent[c][0]) == -1) {
				// create new HELP section
				sections.push(HELPcontent[c][0]);
				var subname = HELPcontent[c][0].substr(1); 
				var prettyname = HELPcontent[c][0][0];
				for ( var i=0; i < subname.length; i++) {
					var ci = subname.charAt(i)
					if (ci == ci.toUpperCase() )
						prettyname += " " + subname.charAt(i).toLowerCase();
					else
						prettyname += subname.charAt(i);
				}

				helpcmds.innerHTML += '<p class="helpsectiontitle" onclick=\'toggleHelp(\"' + HELPcontent[c][0] + '\");\'>' + prettyname + ' <label id="helptogglebtn' + HELPcontent[c][0] + '" class="helptogglebtn">+</label></p><div id="helpcmds' + HELPcontent[c][0] + '" class="helpsection"></div>';

			}
			
			if (HELPcontent[c][0] != "Welcome" || !self.hasOwnProperty("__INLALOLIBHTMLPAGE") ) {
				category = document.getElementById("helpcmds" + HELPcontent[c][0]);
				
				if ( c.indexOf("transpose") >=0 )
					category.innerHTML += "<a class='helpcmd' onmouseenter='help(\"transpose\");'  onclick='help(\"transpose\", true);' >" + c + "</a><br>";
				else if ( c.indexOf("\\") >= 0 ) 
					category.innerHTML += "<a class='helpcmd' onmouseenter='help(\"solve\");' onclick='help(\"solve\" , true);'  >" + c + "</a><br>";					
				else									
					category.innerHTML += "<a class='helpcmd' onmouseenter='help(\"" + c + "\");' onclick='help(\"" + c + "\", true);' >" + c + "</a><br>";	
			}
			
		}
	}
	//category.innerHTML += "<br>";
	for (c in sections) {
		if ( sections[c] != "Welcome") {
			category = document.getElementById("helpcmds" + sections[c]);
			category.innerHTML += "<label class='helptogglebtn' onclick='toggleHelp(\"" + sections[c] + "\");'>--</label><br>";
		}
	}
	
	if (!self.hasOwnProperty("__INLALOLIBHTMLPAGE")) {
		helpcmds.innerHTML += "<p><a href='toolboxes.html' target='_blank'>Add a toolbox...</a></p>";
	
		help("LALOLab");
	}
}

function toggleHelp( section ) {
	var sec = document.getElementById("helpcmds" + section);
	var btn = document.getElementById("helptogglebtn" + section);	
	if ( sec.style.opacity == "1" ) {
		sec.style.opacity = "0" ;
		sec.style.visibility = "hidden" ;
		sec.style.maxHeight = "0px" ;
		btn.innerHTML = "+";
		if ( helpcmds.scrollTop > sec.offsetTop - 20)
			setTimeout(function(){helpcmds.scrollTop = sec.offsetTop - 20;}, 200);
	}
	else {
		sec.style.opacity = "1";
		sec.style.visibility = "visible" ;
		sec.style.maxHeight = sec.scrollHeight + "px" ;
		btn.innerHTML = "--";
	}		
}

