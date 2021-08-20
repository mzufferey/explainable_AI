### “Why Should I Trust You?” Explaining the Predictions of Any Classifier - Ribeiro et al. 2016



**LIME** = an explanation technique that explains the predictions of any classifier in an interpretable and faithful manner, by learning an interpretable model locally around the prediction

two different (but related) definitions of trust: (1) trusting a prediction, i.e. whether a user trusts an individual prediction sufficiently to take some action based on it, and (2) trusting a model, i.e. whether the user trusts a model to behave in reasonable ways if deployed. Both are directly impacted by

 how much the human understands a model’s behaviour, as opposed to seeing it as a black box



“explaining a prediction”, we mean presenting textual or
visual artifacts that provide qualitative understanding of the relationship between the instance’s components (e.g. words in text, patches in an image) and the model’s prediction



Every machine learning application also requires a certain
measure of overall trust in the model. 

Looking at examples offers an alternative method to assess truth in the model, especially if the examples are explained. We thus propose explaining several representative individual predictions of a model as a way to provide a global understanding.



everal ways a model or its evaluation can go
wrong. Data leakage, for example, defined as the uninten- tional leakage of signal into the training (and validation) data that would not appear when deployed [14], potentially increases accuracy



individual prediction explanations can be used to select between models, in conjunction with accuracy. In this case, the algorithm with higher accuracy on the validation set is actually much worse, a fact that is easy to see when explanations are provided (again, due to human prior knowledge), but hard otherwise. Further, there is frequently a mismatch between the metrics that we can compute and optimize (e.g. accuracy) and the actual metrics of interest such as user engagement and retention. While we may not be able to measure such metrics, we have knowledge about how certain model behaviors can influence them. Therefore, a practitioner may wish to choose a less accurate model for content recommendation that does not place high importance in features related to “clickbait” articles (which may hurt user retention), even if exploiting such features increases the accuracy of the model in cross validation.



desired characteristics from explanation methods.



**interpretable**, i.e., provide qualitative understanding between the input variables and the response. 

explanations should be easy to understand, which is not necessarily true of the features used by the model, and thus the “input variables” in the explanations may need to be different than the features

interpretability must take into account the user’s limitations (e.g. linear model with thousands of significant features is not interpretable)



s **local fidelity**. Although it is
often impossible for an explanation to be completely faithful unless it is the complete description of the model itself, for an explanation to be meaningful it must at least be locally faithful, i.e. it must correspond to how the model behaves in the vicinity of the instance being predicted.

local fidelity does not imply global fidelity: features that are globally important may not be important in the local context, and vice versa. While global fidelity would imply local fidelity, identifying globally faithful explanations that are interpretable remains a challenge for complex models



e there are models that are inherently interpretable [6,
17, 26, 27], an explainer should be able to explain any model, and thus be **model-agnostic** (i.e. treat the original model as a black box). A



providing a providing a **global**
**perspective** is important to ascertain trust in the model. As mentioned before, accuracy may often not be a suitable metric to evaluate the model, and thus we want to explain the mode is important to ascertain trust in the model. As mentioned before, accuracy may often not be a suitable metric to evaluate the model, and thus we want to explain the mode



### Local interpretable model-agnostic explanations (LIME)

The overall goal of LIME is to identify an interpretable model over the interpretable representation that is locally faithful to the classifier.

#### Interpretable data representation

distinguish between features and interpretable data representations. As mentioned before, interpretable expla- nations need to use a representation that is understandable to humans, regardless of the actual features used by the model.



#### Fidelity-interpretability trade-off

- **explanation** = a model $g ∈ G$
  - $G$ = a class of potentially interpretable models

- $Ω(g)$ = a measure of **complexity** (as opposed to interpretability) of the explanation $g ∈ G$
- $π_{x}(z)$ =   **proximity** measure between an instance $z$ to $x$, so as to define locality around $x$. 
- $f$ = the **model** being explained be denoted (e.g. in classification, $f(x)$ = the probability that $x$ belongs to a certain class)
- $L(f, g, π_{x})$ =  measure of how unfaithful $g$ is in approximating $f$ in the locality defined by $π_{x}$ (**loss**)



**Ensure both interpretability and local fidelity**: minimize $L(f, g, π_{x}$) while having $Ω(g)$ low enough to be interpretable by humans. The explanation produced by LIME is obtained by the following:
$$
ξ(x) = {\underset{g∈G}{argmin}} \textrm{ } L(f, g, π_{x}) + Ω(g)
$$
This formulation can be used with different explanation families $G$, fidelity functions $L$, and complexity measures $Ω$

#### Sampling for local exploration

- $x$ = the original representation of an instance being explained
- $x'$ = its interpretable representation (e.g. a binary vector indicating presence or absence of a word in text classification) 

Aim: minimize the locality-aware loss $L(f, g, π_{x}$) without making any assumptions about $f$ since we want the explainer to be model-agnostic. To  larn the local behavior of $f$ as the interpretable inputs vary

- approximate $L(f, g, π_{x}$) by drawing samples, weighted by $π_{x}$.
- sample instances around $x'$ by drawing nonzero elements of $x'$ uniformly at random (where the number of such draws is also uniformly sampled)
- given a perturbed sample $z'$, recover the sample in the original representation $z$ ∈ 
- obtain $f(z)$, which is used as a label for the explanation model
- given this dataset $Z$ of perturbed samples with the associated labels, optimize the equation to get an explanation ξ(x).



Even though the original model may be too complex to explain globally, LIME presents an explanation that is locally faithful, where the locality is captured by $π_{x}$. 

The method is fairly robust to sampling noise since the samples are weighted by $π_{x}$



#### Sparse linear explanations

For text classification, we ensure that the explanation is interpretable by letting the interpretable representation be a bag of words, and by setting a limit $K$ on the number of words.

- constant value used for $K$ (exploration of different values for future work)



Use the same Ω for image classification, using “super-pixels” instead of words, such that the interpretable representation of an image is a binary vector where 1 indicates the original super-pixel and 0 indicates a grayed out super-pixel. 

- this choice of Ω makes directly solving of ξ(x) intractable, but approximated by
  - first selecting $K$ features with Lasso 
  - then learning the weights via least squares 



Any choice of interpretable representations and $G$ will have some inherent drawbacks

- while the underlying model can be treated as a black-box, certain interpretable representations will not be powerful enough to explain certain behaviors.
- the choice of $G$$ (sparse linear models) means that if the underlying model is highly non-linear even in the locality of the prediction, there may not be a faithful explanatio
  - we can estimate the faithfulness of the explanation on $Z$, and present this information to the user.
  - this estimate of faithfulness can also be used for selecting an appropriate family of explanations from a set of multiple interpretable model classes, thus adapting to the given dataset and the classifier (such exploration for future work)



#### Submodular pick for explaining models

Although an explanation of a single prediction provides some understanding into the reliability of the classifier to the user, it is not sufficient to evaluate and assess trust in the model as a whole. 

Give a global understanding of the model by **explaining a set of individual instances**. This approach is still model agnostic, and is complementary to computing summary statistics such as held-out accuracy. 

Even though explanations of multiple instances can be insightful, **these instances need to be selected judiciously**, since users may not have the time to examine a large number of explanations. 

- time/patience that humans have represented by a **budget $B$** that denotes the number of explanations they are willing to look at in order to understand a model
- given a set of instances $X$,  **pick step** = the task of selecting $B$ instances for the user to inspect
  - does not depend on the existence of explanations
    - one of the main purpose of tools like Modeltracker is to assist users in selecting instances themselves, and examining the raw data and predictions
    - since looking at raw data is not enough to understand predictions and get insights, **the pick step should take into account the explanations that accompany each prediction**
  - this method should pick a diverse, **representative set** of explanations to show the user – i.e. non-redundant explanations that represent how the model behaves globally



**Submodular pick algorithm**

- construct an explanation matrix $W$ that represents the local importance of the interpretable components for each instance.
- for each component (column), denote the global importance $I_{j}$ of that component in the explanation space.
  - we want $I$ such that features that explain many different instances have higher importance scores.
- while we want to pick instances that cover the important components, the set of explanations must not be redundant in the components they show the users, i.e. avoid selecting instances with similar explanations. 
  - formalize this non-redundant coverage intuition by defining **coverage** as the set function $c$ that, given $W$ and $I$, computes the total importance of the features that appear in at least one instance in a set $V$

The pick problem consists of finding the set $V$, $|V| ≤ B$ that achieves highest coverage.
$$
Pick(W,I)=\underset{V,|V|≤ B}{argmax} \textrm{ } c(V,W,I)
$$
Maximizing a weighted coverage function is NP-hard

- $c(V∪\{i\},W, I)−c(V,W, I)$ =  marginal coverage gain of adding an instance $i$ to a set $V$
- submodularity:  a greedy algorithm that iteratively adds the instance with the highest marginal coverage gain to the solution offers a constant-factor approximation guarantee of $1−1/e$ to the optimum



[...]

### Related work

Other tools do not address the problem of explaining individual predictions. 

The submodular pick procedure can be incorporated in other tools to aid users in navigating larger datasets.

Some recent work aims to **anticipate failures** in machine learning, specifically for vision tasks. Can lead to an increase in trust, by avoiding “silly mistakes”.

- either require additional specific annotations and feature engineering or do not provide insight into why a decision should not be trusted
- assume that the current evaluation metrics are reliable, which may not be the case if problems such as data leakage are present

Other recent work focuses on exposing users to **different kinds of mistakes** (our pick step). Often will not generalize for this task. 

- with LIME, even non-experts are able to identify these irregularities when explanations are present
- LIME can complement these existing systems, and allow users to assess trust even when a prediction seems “correct” but is made for the wrong reasons



In interpretable models, interpretability comes at the cost of flexibility, accuracy, or efficiency. E.g. 

- tool that focuses on an already interpretable model (Naive Bayes)

- tool constrained to specific neural network architectures or incapable of detecting “non object” parts of the images

- LIME focuses on general, model-agnostic explanations that can be applied to any classifier or regressor that is appropriate for the domain - even ones that are yet to be proposed

  

A common approach to model-agnostic explanation is learning a potentially interpretable model on the predictions of the original model. Having the explanation be a gradient vector captures a similar locality intuition to that of LIME. 

- interpreting the coefficients on the gradient is difficult, particularly for confident predictions (where gradient is near zero)

- these explanations approximate the original model globally, thus maintaining local fidelity becomes a significant challenge

- LIME solves the much more feasible task of finding a model that approximates the original model locally

  

None of existing approaches explicitly take cognitive limitations into account, and thus may produce non-interpretable explanations (e.g. gradients or linear models with thousands of non-zero weights); worse problem if the original features are nonsensical to humans (e.g. word embeddings)

- LIME incorporates interpretability both in the optimization and in our notion of interpretable representation, such that domain and task specific interpretability criteria can be accommodated



### Conclusion and future work

LIME = a modular and extensible approach to faithfully explain the predictions of any model in an interpretable manner

SP-LIME = a method to select representative and non-redundant predictions, providing a global view of the model to users