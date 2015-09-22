data = loadURL( "examples/iris.train" )
X = data[:, 0:4]
Y = data[:,4]
data = loadURL( "examples/iris.test" )
Xtest = data[:, 0:4]
Ytest = data[:,4]
svm = new Classifier(SVM, {kernel: "rbf", kernelpar: 1} )
trainingError = svm.train(X, Y)
plot( svm.predict( Xtest ) )
RecRate = svm.test( Xtest, Ytest )
