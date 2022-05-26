Working with Python and the Machine Learning and Data Science ecosystem is fun, but
when it comes to deployment, you may not want to have to use Python. The goal of this
repository is to address this need; you can experiment and train models in the rich
Python ecosystem, but deploy your models in other languages and platforms. 
**scikit-learn4x** is a free an open source library that allows you to deploy 
scikit-learn model in other programming languages. As such, the training codes are not
included in this repository (the fit methods), only the inference of models is 
supported (the predict, predict_proba and predict_log_proba).

This repository is the code for the Java Virtual Machine (**scikit-learn 4 JVM**). 
You can also use **scikit-learn for .NET** to deploy in C# and other .NET based 
languages.

### Important Links
scikit-learn4x: https://scikitlearn4x.ai
scikit-learn4x Python Library: https://github.com/scikitlearn4x/scikit-learn-4x-python-lib
scikit-learn 4 .NET Repository: https://github.com/scikitlearn4x/scikit-learn-4-net

## Example Usage
The process of using **scikit-learn4x** is easy, you train your model in Python and
save it into a file using the sklearn4x library. Please find the link to the library
at https://pypi.org/project/sklearn4x/. Then, you use the scikit-learn 4 JVM to load
the saved model in Java or other JVM based languages.

```
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn4x.sklearn4x import save_scikit_learn_model
import pandas as pd

ds = datasets.load_iris()
X = ds.data
y = ds.target

train_data = pd.DataFrame(data=X, index=None, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], dtype=X.dtype, copy=False)

classifier = GaussianNB()
classifier.fit(train_data, y)

predictions = classifier.predict(X)
prediction_probabilities = classifier.predict_proba(X)
prediction_log_probabilities = classifier.predict_log_proba(X)

test_data = {
    "training_data": X,
    "predictions": predictions,
    "prediction_probabilities": prediction_probabilities,
    "prediction_log_probabilities": prediction_log_probabilities,
}

print(f'First data point prediction: {predictions[0]}')
print(f'First data point probabilities: {prediction_probabilities[0, 0]:.3f}, {prediction_probabilities[0, 1]:.3f}')
print(f'First data point log probabilities: {prediction_log_probabilities[0, 0]:.3f}, {prediction_log_probabilities[0, 1]:.3f}')


save_scikit_learn_model(classifier, '/some/path/on/disk.skx', test_data)

# You should see the following outputs:
#
# First data point prediction: 0
# First data point probabilities: 1.000, 0.000
# First data point log probabilities: 0.000, -41.141
```

Now, in the Java side:

```
String path = "/same/path/on/disk.skx";
IScikitLearnPackage binaryPackage = ScikitLearnPackageFactory.loadFromFile(path);

// Check actual computed values
GaussianNaiveBayes classifier = (GaussianNaiveBayes)binaryPackage.getModel(0);

NumpyArray<Double> x = (NumpyArray<Double>)binaryPackage.getExtraValues().get("training_data");

NumpyArray<Long> predictions = classifier.predict(x);
NumpyArray<Double> probabilities = classifier.predictProbabilities(x);
NumpyArray<Double> logProbabilities = classifier.predictLogProbabilities(x);

System.out.println("First data point prediction: " + predictions.get(0));
System.out.println(String.format("First data point probabilities: %.3f, %.3f", probabilities.get(0, 0), probabilities.get(0, 1)));
System.out.println(String.format("First data point log probabilities: %.3f, %.3f", logProbabilities.get(0, 0), logProbabilities.get(0, 1)));

/*
    You should see the same outputs as Python's:

    First data point prediction: 0
    First data point probabilities: 1.000, 0.000
    First data point log probabilities: 0.000, -41.141
*/
```

## Support
This is an ongoing project at the moment and only a subset of the scikit-learn APIs
are ported to **scikit-learn for JVM**. The roadmap for the future is porting the 
components with the following order:

First, a baseline classifier is ported. In this case, the GaussianNB was selected due
to its simplicity to focus on implementing proper testing and CI/CD pipeline. This step
is already done.

Second, porting the utility classes in scikit-learn such including but not limited to
the LabelEncoder and Pipelines.

Third, porting the rich library of the classifiers with proper testing to JVM and 
other languages.

Forth, porting the rich library of the regressors with proper testing to JVM and
other languages.

### Classifiers
The following classifiers are supported:
* Naive Bayes
    - Gaussian Naive Bayes (GaussianNB)
### Regressors
The regressors are not supported yet.

## Installation

To be completed

## Development

To be completed

## Performance

To be completed

## Project Raison d'Être

To be completed

## Credits

To be completed

## Help and Support

Feel free to contact me with my email address:
ma (initials for Mohammad Ali), then underscore and then my last name (Yektaie). Finally,
add at outlook.com.

