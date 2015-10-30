all: src/glpk.min.js jszip.min.js lalolib.js ml.js laloliblab.js lalohelp.js plotlib.js 

# Dependencies: GLPK.js and JSZIP.js
src/glpk.min.js: 
	@echo "Fetching glpk.js..."
	@rm -rf tmp
	@mkdir tmp
	@wget -O tmp/glpk.zip https://github.com/hgourvest/glpk.js/archive/master.zip
	@cd tmp ; unzip -o -qq -j glpk.zip 
	@cp tmp/glpk.min.js src/
	@rm -rf tmp

jszip.min.js:
	@echo "Fetching jszip.js..."
	@rm -rf tmp
	@mkdir tmp
	@wget -O tmp/jszip.zip https://github.com/Stuk/jszip/archive/master.zip
	@cd tmp ; unzip -o -qq -j jszip.zip 
	@cp tmp/jszip.min.js ./
	@rm -rf tmp
	
# We do not include toolboxes in lalolib.js: explicitly load them with <script...> in the html file
# This avoids loading a large file with toolboxes that are not used.
lalolib.js: src/lalolibbase.js src/linalg.js src/sparse.js src/laloglpk.js src/glpk.min.js src/stats.js src/laloplots.js 
	cat src/lalolibbase.js src/linalg.js src/sparse.js src/laloglpk.js src/glpk.min.js src/stats.js src/laloplots.js > lalolib.js 

# ML.js includes LALOLib plus Machine learning functions
ml.js: lalolib.js src/kernels.js src/Classifier.js src/Regression.js src/Clustering.js src/DimReduction.js src/Loaders.js src/MLutils.js
	cat lalolib.js src/kernels.js src/Classifier.js src/Regression.js src/Clustering.js src/DimReduction.js src/Loaders.js src/MLutils.js > ml.js
	
# version for LALOLab (different printing functions...)
# All toolboxes are included in LALOLab by default
laloliblab.js: src/linalg.js src/sparse.js src/laloglpk.js src/stats.js src/glpk.min.js src/kernels.js src/Classifier.js src/Regression.js src/Clustering.js src/DimReduction.js src/GraphicalModel.js src/Loaders.js src/MLutils.js toolboxes
	cat src/linalg.js src/sparse.js src/laloglpk.js src/glpk.min.js src/stats.js src/kernels.js src/Classifier.js src/Regression.js src/Clustering.js src/DimReduction.js src/GraphicalModel.js src/Loaders.js src/MLutils.js toolboxes/*.js > laloliblab.js

# Online Help 
lalohelp.js: src/lalohelpbase.js help/*
	cat src/lalohelpbase.js help/* > lalohelp.js

# LALOLAb plots
plotlib.js: src/plotlib.js
	cp src/plotlib.js ./

