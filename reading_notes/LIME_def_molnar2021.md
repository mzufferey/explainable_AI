## LIME

*(source: https://christophm.github.io/interpretable-ml-book/)*

**Surrogate models** = trained to approximate the predictions of the underlying black box model

**LIME** is an example of **local** surrogate models (= interpretable models that are used to  explain **individual** predictions of black box machine learning models)

Idea of LIME

- forget about the training data and imagine you only have the  black box model where you can input data points and get the predictions
- you can probe the box as often as you want; goal: understand why the machine learning model made a certain prediction
- LIME tests what happens to the predictions when you give variations of  your data into the machine learning model
- LIME **generates a new dataset consisting of perturbed samples and the  corresponding predictions** of the black box model
- on this new dataset LIME then trains **an interpretable model** (can be of any type, e.g. Lasso or a decision tree), which is  **weighted by the proximity of the sampled instances to the instance of  interest** 
- the learned model should be a good approximation of the machine learning model predictions locally, but it does not have to be a good global  approximation (this kind of accuracy is aka **local fidelity**)



Mathematically, **local surrogate models with interpretability constraint** can be expressed as follows:

$explanation(x) = {\underset{g∈G}{argmin}} \textrm{ } L(f, g, π_{x}) + Ω(g)$ 



- The **explanation model** for instance $x$ is the **model** **$g$** (e.g. linear  regression model) that minimizes **loss** **$L$** (e.g. mean squared error), which measures how close the explanation is to the prediction of the **original model $f$** (e.g. an xgboost model), while the **model complexity $Ω(g)$** is kept low (e.g. prefer fewer features). 

- **$G$** is the **family of possible explanations**, for example all possible linear regression models. 
- The **proximity measure $\pi_{x}$** defines how large the neighborhood around instance $x$ is that we  consider for the explanation. 



In practice, **LIME only optimizes the loss part**. 



**The user has to determine the complexity,** e.g. by selecting the maximum  number of features that the linear regression model may use.



The recipe for training local surrogate models:

- Select your **instance of interest** for which you want to have an explanation of its black box prediction.
- **Perturb your dataset** and get the black box predictions for these new points.
- **Weight** the new samples according to their proximity to the instance of interest.
- Train a **weighted, interpretable model** on the dataset with the variations.
- **Explain the prediction** by interpreting the local model.



Choice of the interpretable surrogate model

- can be e.g. linear regression
- K, the number of features, should be selected in advance
  - the lower K, the easier it is to interpret the model
  - a higher K potentially produces models with higher fidelity
- several methods for training models with exactly K features
  - Lasso is a good choice
    - a high regularization parameter λ  yields a model without any feature
    - by retraining the Lasso models with slowly decreasing λ, one after the other, the features get weight estimates that differ from zero
    - if there are K features in the model, you have reached the desired  number of features
  - other strategies are forward or backward selection of features
    - = either start with the full model (= containing all  features) or with a model with only the intercept and then test which  feature would bring the biggest improvement when added or removed, until a model with K features is reached



How to get the variations of the data ? Depends on the type of data:

- text and images: turn single words or super-pixels on or off
- tabular data: create new samples by perturbing each feature individually, drawing from a normal distribution with mean and  standard deviation taken from the feature

### LIME for Tabular Data

LIME samples are not taken around the instance of interest, but from the training data’s mass center, which is problematic. But it increases the probability that the result for some of the sample  points predictions differ from the data point of interest and that LIME  can learn at least some explanation.

Defining a **meaningful neighborhood** around a point is difficult. LIME currently uses an **exponential smoothing kernel** to define the  neighborhood. 

- **smoothing kernel** = a function that takes two data instances and  returns a proximity measure. 
- the **kernel width** determines how large the neighborhood is
  - small kernel width: an instance must be very close to  influence the local model
  - larger kernel width: instances  that are farther away also influence the model. 

LIME uses an exponential smoothing kernel (on the normalized data) and the kernel width is 0.75 times the square root of  the number of columns of the training data. 

The big problem is that we do not have a good way to find the best  kernel or width. And where does the 0.75 even come from? In certain scenarios, you can easily turn your explanation around by  changing the kernel width.

The example shows only one feature. It gets worse in high-dimensional feature spaces. It is also very unclear whether the distance measure should treat all features equally.

### LIME for Text

Variations of the data are generated differently: 

- new texts created by randomly removing words from the original text
- dataset represented with binary features for each word (1 = the corresponding word is included; 0 = it has been removed)



### LIME for Images



Perturbing individual pixels would not make sense since many more than one pixel contribute to one class. Also randomly changing individual pixels would probably not change the  predictions by much. Instead:

- Variations are created by segmenting the image  into “superpixels” and **turning superpixels off or on**. 
  - superpixels are interconnected pixels with similar colors 
  - turned off by replacing each pixel with a user-defined color such as  gray
  - the user can also specify a probability for turning off a superpixel in  each permutation.





### Advantages

Even if you **replace the underlying machine learning model**, you can still use the same local, interpretable model for explanation. Suppose the people looking at the explanations understand decision trees best. Because you use local surrogate models, you use decision trees as  explanations without actually having to use a decision tree to make the  predictions. For example, you can use a SVM. And if it turns out that an xgboost model works better, you can replace  the SVM and still use as decision tree to explain the predictions.

When using Lasso or short trees, the resulting **explanations are short (= selective) and possibly contrastive** (thus human-friendly). even though not sufficient for complete attributions.

LIME is one of the few methods that **works for tabular data, text and images**.

The **fidelity measure** (how well the interpretable  model approximates the black box predictions) gives us a good idea of  how reliable the interpretable model is in explaining the black box  predictions in the neighborhood of the data instance of interest.

Implemented in Python ([lime](https://github.com/marcotcr/lime) library) and R ([lime package](https://cran.r-project.org/web/packages/lime/index.html) and [iml package](https://cran.r-project.org/web/packages/iml/index.html)), **very easy to use**.

The explanations created with local surrogate models **can use other (interpretable) features than the original model was trained on.** Of course, these interpretable features must be derived from the data  instances. 

- A text classifier can rely on abstract word embeddings as features, but  the explanation can be based on the presence or absence of words in a  sentence. 
- A regression model can rely on a non-interpretable transformation of  some attributes, but the explanations can be created with the original  attributes. 
- For example, the regression model could be trained on components of a  principal component analysis (PCA) of answers to a survey, but LIME  might be trained on the original survey questions. 

Using **interpretable features** for LIME can be a big advantage over other methods, especially when the model was trained with non-interpretable  features.

### Disadvantages

The correct **definition of the neighborhood** is a very big, unsolved problem when using LIME with tabular data. For each application you have to try different kernel settings and see for yourself if the explanations make sense to find **good kernel widths**.

**Sampling** could be improved in the current implementation of LIME. Data points are sampled from a Gaussian distribution, ignoring the correlation between features. This can lead to unlikely data points which can then be used to learn local explanation models.

The **complexity** of the explanation model has to be defined in advance. This is just a small complaint, because in the end the user always has to define the compromise between fidelity and sparsity.

Another really big problem is the **instability of the explanations**. Has been shown that the explanations of two very close points varied greatly in a simulated setting. When repeating the sampling process, explanations that come out can be different. Difficult to trust the explanations.

LIME explanations can be manipulated to hide biases. The possibility of **manipulation** makes it more difficult to trust explanations generated with LIME.

------