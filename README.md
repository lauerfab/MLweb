# MLweb

MLweb is an open-source project that aims at bringing machine learning capabilities to web pages and web applications. See the [official website](http://mlweb.loria.fr/) for more information.

MLweb includes the following three components: 

- **ML.js**: a javascript library for machine learning
- **LALOLib**: a javascript library for scientific computing (linear algebra, statistics, optimization)
- **LALOLab**: an online Matlab-like development environment (try it at [http://mlweb.loria.fr/lalolab/](http://mlweb.loria.fr/lalolab/))

## Documentation

Documentation for LALOLib and ML.js is available [here](http://mlweb.loria.fr/lalolab/lalolib.html).

[LALOLab](http://mlweb.loria.fr/lalolab/) comes with an online help including the list of all functions and many examples. 

## Note to users

This repository is mostly intended for developers wishing to modify or extend these tools.
Ready-to-use versions of the tools are available online at:

- [http://mlweb.loria.fr/lalolib.js](http://mlweb.loria.fr/lalolib.js) for LALOLib
- [http://mlweb.loria.fr/ml.js](http://mlweb.loria.fr/ml.js) for ML.js 
- [http://mlweb.loria.fr/lalolab/](http://mlweb.loria.fr/lalolab/) for LALOLab

or as modules (see the [documentation](http://mlweb.loria.fr/lalolab/lalolib.html) for details) at:

- [http://mlweb.loria.fr/lalolib-module.js](http://mlweb.loria.fr/lalolib-module.js)
- [http://mlweb.loria.fr/mljs-module.js](http://mlweb.loria.fr/mljs-module.js)
- [http://mlweb.loria.fr/lalolib-noglpk-module.js](http://mlweb.loria.fr/lalolib-noglpk-module.js)
- [http://mlweb.loria.fr/mljs-noglpk-module.js](http://mlweb.loria.fr/mljs-noglpk-module.js)

## Functions provided by LALOLib

- **Linear algerbra:** basic vector and matrix operations, linear system solvers, matrix factorizations (QR, Cholesky), eigendecomposition, singular value decomposition, conjugate gradient sparse linear system solver, complex numbers/matrices, discrete Fourier transform... )
- **Statistics:** random numbers, sampling from and estimating standard distributions
- **Optimization:** steepest descent, BFGS, linear programming (thanks to [glpk.js](https://github.com/hgourvest/glpk.js)), quadratic programming

See [this benchmark](http://mlweb.loria.fr/benchmark/) for a comparison of LALOLib with other linear algebra javascript libraries.

## Machine learning capabilities provided by ML.js

#### Classification 

- K-nearest neighbors,
- Linear/quadratic discriminant analysis,
- Naive Bayes classifier,
- Logistic regression,
- Perceptron,
- Multi-layer perceptron, 
- Support vector machines, 
- Multi-class support vector machines, 
- Decision trees

#### Regression 

- Least squares, 
- Least absolute devations, 
- K-nearest neighbors, 
- Ridge regression, 
- LASSO, 
- LARS, 
- Orthogonal least squares, 
- Multi-layer perceptron, 
- Kernel ridge regression, 
- Support vector regression, 
- K-LinReg
		
#### Clustering 

- K-means, 
- Spectral clustering

#### Dimensionality reduction

- Principal component analysis, 
- Locally linear embedding, 
- Local tangent space alignment 


## Installation

Download the source files from [here](http://mlweb.loria.fr/lalolab/mlweb.zip) or by cloning this repository and run

```
cd lalolab
make
```

to build the libraries in the `lalolab/` folder:
```
lalolib.js and lalolibworker.js  --> for LALOLib
ml.js and mlworker.js            --> for ML.js
```

Then, you can launch *LALOLab* by opening `lalolab/index.html` in a browser, for instance with 

```
firefox index.html
```

**Note to Chrome users:** you need to use the `--allow-file-access-from-files` flag on Chrome command line. For Chromium under Linux, you can use the convenient script `lalolab/chromelab`.

