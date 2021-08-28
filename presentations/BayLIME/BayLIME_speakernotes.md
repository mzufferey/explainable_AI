### Explainable AI

explainable AI is a field of research that aims at improving the trust and transparency of AI

it seeks to provide good explanations. 

but what is an explanation ? 

- in a very general way it can be defined as the answer to a why-question (Miller 2017)
- in AI field, an explanation relates the feature values of an instance to its model prediction in a humanly understandable way (Molnar 2021)
  - more specifically, by “explaining a prediction”, we mean presenting textual or visual artifacts that provide qualitative understanding of the relationship between the instance’s components (e.g. words in text, patches in an image) and the model’s prediction
  - example: The linear regression model is already equipped with an explanation method (interpretation of the weights). (Molnar 2021)

If there is not straightforward definition of "goodness", 2 import evaluation criteria are

- interpretability 
  - which is the qualitative understanding between the input variables and the response
  - can also be defined as the degree to which a human can understand the cause of a decision (Molnar 2021)
- fidelity = how truthfully the explanation represents the unknown behaviour of the underlying AI model
  - we want the XAI method to explain the true cause of the underlying model's prediction
  - the explanation should predict the event as truthfully as possible; so if we say that a second balcony increases the price of a house, then  that also should apply to other houses (or at least to similar houses) (Molnar 2021)

“explaining a prediction”, we mean presenting textual or
visual artifacts that provide qualitative understanding of the relationship between the instance’s components (e.g. words in text, patches in an image) and the model’s prediction.

[Ribeiro et al. 2016a]

### Explainable AI



What is an explanation ?



## LIME: Local Interpretable Model-agnostic Explanations

LIME is a tool that was released in 2016 and is one of the state-of-the-art XAI method today.

The name stands for "Local Interpretable Model-agnostic Explanations" and focuses on the implementation of local surrogate models

- **surrogate** models are **interpretable** models trained to approximate the predictions of the underlying black box model

NB: examples of potentially interpretable models are linear models or decision trees

- **local** = means that they focus on training surrogate models to explain **individual** predictions
- **model-agnostic** = means that they can be used for any machine learning model
- LIME also has the desirable property to work for all tabular data, text and images
- LIME is a widely used XAI method today





### LIME: how does it proceed ?

As BayLIME is largely based on LIME, it is worth to go a bit deeper in how LIME is working.

In short, the aim of LIME is to identify an interpretable model over the interpretable representation that is locally faithful.

That's being said, how does it work ?

For any black-box AI model (**model-agnostic**), the procedure is the following:

1. select your instance of interest for which you want to have an explanation for the black box prediction (**local**)

* an instance can be an image or a row from a tabular dataset

2. perturb the dataset

- text and images: turn single words or super-pixels on or off
- tabular data: create new samples by perturbing each feature individually, drawing from a normal distribution with mean and  standard deviation taken from the feature

3. get new black box predictions for these new points

Such changes in the predictions, as we vary the features of the instance, can help us understand why the AI model made a certain prediction over the instance

4.  weight the perturbed samples according to their proximity to the instance of interest

the weights are determined by some kernel function

5. train a weighted, interpretable model (for example a linear regressor) on the dataset with the variations

new dataset consisting of the perturbed inputs and the cor- responding predictions made by the black-box AI model can be generated, upon which LIME trains a surrogate model that is interpretable to humans (e.g. linear regressors)

6. explain the prediction by interpreting the local model (e.g. the absolute values of the coefficients of the regressor as an indication of feature importance, based on which rankings can be done)

Then, the absolute values of the coefficient vector β
represent the importance of the m features, based on which
rankings can be done. By default, LIME 9 uses the Ridge
regressor with weighted samples; with the ridge regularization parameter r; when r=0, this is weighted OLS

(Molnar 2021, §5.8)



### LIME: how does it proceed ?

This was for the intuition...

If we translate it mathematically, this gives the following formula ($f$ is the model to be explained, $g$ is the explanation model) we see that it combines 

- a fidelity function measure of how unfaithful $g$ is in approximating $f$ in the locality defined by $Πx$ - so we want to minimize it 
- the fidelity function is combined with the complexity measure $Ω(g)$ which reflects how interpretable/uninterpretable is the model (e.g. could be for example the depth of a tree) - so we also want it as low as possible

the formula reflects the trade-off between fidelity and interpretability

with LIME, we will estimate the fidelity by generating perturbed samples around the instance, making predictions with the black box model and weighting them according to their proximity to the instance (Ribeiro et al. 2016b)



### LIME: how does it proceed ?

how LIME works can be represented with this toy example drawing

here a globally complex model is explained using a locally-faithful linear explanation

- the blue/pink background represents the black-box model complex decision function $f$
- the bright bold red cross is the instance being explained
- LIME samples instances, gets predictions using $f$, and weights them by the proximity to the instance being explained (reflected here by size)
- the dashed line is the explanation that is locally (but not globally) faithful

recall: the function $f$ remains unknown to LIME

(Ribeiro 2016b)

maybe more concretely, this is a kind of output that you can have for text data with LIME 

in this example we want to classify the small text on the right in 2 classes, either sincer or insecere

on the left we have the prediction probabilities and in the middle the weights; they can be interpreted as follows 

if we remove the word "blame" from the text, we expect the classifier to predict "insincere" with 0.16-0.07 = 0.09 probability



Intuitively, an explanation is a local linear approximation of the model's behaviour. 

!!! CHANGE WITH THE EXAMPLE FROM THE BLOG !!!



While the model may be very complex globally, it is easier to approximate it around the vicinity of a particular instance. While treating the model as a black box, we perturb the instance we want to explain and learn a sparse linear model around it, as an explanation. The figure below illustrates the intuition for this procedure. The model's decision function is represented by the blue/pink background, and is clearly nonlinear. The bright red cross is the instance being explained (let's call it X). We sample instances around X, and weight them according to their proximity to X (weight here is indicated by size). We then learn a linear model (dashed line) that approximates the model well in the vicinity of X, but not necessarily globally.





### **LIME weaknesses**

Despite being widely used, several weaknesses of LIME have been reported



First, it is known that in repeated runs different explanations can be generated for the same prediction; so it suffers from **inconsistency**: 

*  this instability arises from the randomness in generating perturbed samples that are used for the training of local surrogate models
* randomness can reduced by enlarging sample size, but might be impractical in real-word applications
  *  the time required to produce an explanation is
    dominated by the complexity of the black-box AI model
  * to improve LIME's efficiency, the sole option is to limit the number of queries made to the AI
    model (which is very costly when the AI model is
    deep)
* this inconsistency llimits its usefulness in critical applications (e.g. medical domain)

FIGURE 1

Here is an illustration of the inconsistency of LIME.

A CNN predicts the instance of Fig1 A as 

“Bernese mountain dog” (top-1 label). To explain this prediction, we vary the size of perturbed samples (denoted as n) and record the time consumption in Fig. 1 (B). We see that the computational time is linear with respect to n. If, say, an application requires LIME to respond in 20s, we have to limit n to around 100 (in our case). Then, we may easily get three inconsistent explanations, as shown in Fig. 1 (C)-(E), in three repeated runs of LIME.





### **LIME weaknesses**

Another drawback of LIME is its **unrobustness** to kernel settings

- kernel functions are used to calculate the distance to the instance of interest 
- this is needed in the 4-th step described earlier, to weight the samples
- LIME uses an exponential smoothing kernel to define this neighbourhood: the kernel width determines how large the neighbourhood 
  - smaller kernel width means that an instance needs to be closer to influence the local model
- in practice, there is no effective way exists to find the best kernel settings; the best strategy so far consists in trying different kernel settings and see if the explana- tions make sense, which inevitably is subject to errors/bias. 
- also, an explanation may be easily turned around by changing the kernel settings

(Molnar 2021)



### BayeLIME: LIME in a Bayesian way

Now that we got an overview of LIME and its weaknesses; we can move to the method described in the article I am presenting, namely BayeLIME, which specifically aims at adressing the 2 main drawbacks of LIME

BayeLIME is a Bayesian modification of LIME.

so in this bayesian linear regression framework

we are interested in the posterior distribution of $\mu _n$which is the posterior mean vector of $\beta$ the coefficient vector of the model

for an input of $n$ samples and $m$ features,  $\mu _n$ is defined as follows:
$$
µ_n = (λI_m + αX^TWX)^{−1}λI_mµ_0 +\\
(λI_m + αX^TWX)^{−1}αX^TWXβ_{MLE}
$$
where 

* $µ_0$ and $\mu_n$ are respectively the prior and posterior mean vector of $\beta$, which is the coefficient vector of $m$ features for the linear regression model
* $β_{MLE}$ is the Maximum Likelihood Estimates (MLE) for the linear regression model on the weighted samples.

- $α$ the precision parameter (reciprocal of the variance) representing noise in this linear assumption

  (Bishop: $\alpha \rightarrow 0$ gives infinitely broad prior; $S_0 = \alpha^{-1}I$ )

-  $\lambda$ = precision parameter that governs the Gaussian prior of the covariance matrix of $\beta$

* $W$ = diagonal matrix of the weights calculated by a kernel function according to the new samples’ proximity to the original instance 
* $I$ is of course the identity matrix

the formula looks rather complicated but the key point here is to observe that it is a **weighted sum** of the prior estimates $µ_0$ and the likelihood of the new samples $β_{MLE}$, in other words a **Bayesian combination of prior knowledge and the new observations**



NB: In *statistics*, *precision* is the reciprocal of the variance, and the *precision* matrix is the matrix inverse of the *covariance matrix*

*(NB: Under normal error assumption, as is typically assumed in linear regression, the MLE and the LSE are the same)*

*$β_{MLE} = (X^TWX)^{−1}X^TWy$ is the Maximum Likelihood Estimates (MLE) for the linear regression model on the weighted samples*

*i.e. the estimated parameter values are linear combinations of the observed values*

*(the OLS estimators in the ˆβ vector are a linear combination of existing random variables (X and y))*

*[NB: as Gaussian, MLE = OLS]*

*beta = cov(x,y) / var(x)*

*the first term relates to denominator, the second to numerator*

*the variance covariance matrix of the OLS estimator ^beta = σ2(X′X)−1*

*X′X is (n times) its sample variance, and (X′X)−1 its recirpocal. So the higher the variance = variability in the  regressor, the lower the variance of the coefficient estimator: the more variability we have in the explanatory variable, the more accurately we can estimate the unknown coefficient.*  

*f β^ is inversely proportional to the variance of x*

*Thus, the diagonal elements of **X**' **X** are sums of squares, and the off-diagonal elements are cross products. Note that the cross product matrix **X**' **X** is a [symmetric](https://stattrek.com/statistics/dictionary.aspx?definition=Symmetric_matrix) matrix.*

*https://stattrek.com/matrix-algebrasums-of-squares.aspx*

*(XTX)−1XTY∼N[β,(XTX)−1σ2] =>  (XTX)−1XT is just a complicated scaling matrix that transforms the distribution of Y*

*larger variability of the predictor variable will in general lead to  more precise estimation of its coefficient. This is the idea often  exploited in the design of experiments, where by choosing values for the (non-random) predictors, one tries to make the determinant of (X^TX) as large as possible, the determinant being a measure of variability.*

*In addition, as the number of measurements gets larger, the variance of  the estimated parameters will decrease. So, overall the absolute value  of the entries of X**T**X will be higher, as the number of columns of X**T is n and the number of rows of X is n, and each entry of X**T**X is a sum of n product pairs. The absolute value of the entries of the inverse (X**T**X)−1 will be lower.* 

*Since each weight is inversely proportional to the error variance, it reflects the information in that observation*

*sample VCV is (1/n)XTX*

*X**T**X is not invertible : 1) more parameters than samples; 2) not all columns independent*

### BayeLIME: LIME in a Bayesian way

how are the weights of this weighted sum defined ?

especially in the 1 sample 1 feature case, it becomes apparent that 

* if we look at the part with $\mu_0$ , it is proportional to  $\lambda I_m$ 
  * = the **"pseudo-count" of prior sample size** based on which we form the prior estimates $µ_{0}$  ??????
* as for the part with the ML estimates, it is scaled by the **"accurate-actual-count" of observation sample size**, i.e. the actual observation of the *n* new samples scaled by the precision *α*  and the weights $w$ of the new samples

?????

link between covariance and counts ???!!!

### BayeLIME: procedure recap

1. before the new experiment, form the **prior estimate** of $µ_{0}$ based on $λ$ data points 
2. in the experiments, collect **n** new samples and consider their precision ($α$) and weights ($w_{c}$) for forming a **ML estimate** $β_{MLE}$
3. **combine $µ_{0}$ and $β_{MLE}$** according to their proportions of the effective samples size ($λ$ and $αw_{c}n$, respectively)
4. calculate the **posterior precision** captured by all effective samples (i.e. $λ + αw_{c}n$); the posterior precision gives the confidence in the new posterior estimate



###  BayeLIME: options for the priors

regarding the kind of priors, BayLIME implements 3 options

1.  **non-informative** priors

* When no prior knowledge is available, we assume a zero mean vector for µ_0 and do Bayesian model selection for λ and α. 

* solutions for λ and α are implicit, since they are obtained by starting with initial values and then iterating over some interval equations until convergence.
  BayLIME ??????

*NB: An implicit solution is when you have f(x,y)=g(x,y) which means that y and x are mixed together. y is not expressed in terms of x only. You can  have x and y on both sides of the equal sign or you can have y on one  side and x,y on the other side. An example of implicit solution is  y=x(x+y)^2*

2.  **partial informative** priors

* in this case we assume a known complete prior distribution of the feature coefficients with mean vector µ0 and precision λ.

* the parameter α is still unknown, and fitted from data. 
* as before, we modify the Bayesian model selection algorithm by iterating α (but with fixed λ in this case) to maximise the log marginal likelihood until convergence



3.  **full informative** priors (ideal scenario)

* and the last possibility, the full informative priors, represents the ideal scenario, when full prior knowledge of all the µ0, λ and α parameters is available
* in this situation BayLIME may directly implement the closed-form results 



### BayeLIME: upsides

The combination between prior knowledge and new observations in this weighted sum allows BayLIME to address LIME's weaknesses as it

$\rightarrow$ **improves consistency** by averaging out randomness 

* βMLE is a function of the n randomly perturbed samples that causes inconsistency, while µ0 is independent from the cause

$\rightarrow$  **improves robustness** by averaging out effects from kernels

* βMLE is a function of W that depends on the choice of kernel settings, while µ0 is independent from kernel settings. 

$\rightarrow$  **improves explanation fidelity** by combining diverse information

* µ0 normally contains added diverse information to βMLE (black-box queries) that benefits the explanation fidelity



### Methods: BayeLIME validation

These are nice claims. This was evaluated with the 3 corresponding research questions

 RQ1. <b>consistency</b> improvement (vs. LIME)

 RQ2. <b>robustness</b> to kernel settings improvement (vs. LIME)

 RQ3. explanation <b>fidelity</b> improvement (vs. XAI methods)

by conducting various experiments on 3 datasets: 

1. Boston house-price dataset
2. Breast cancer Wisconsin dataset 
3. a set of CNNs pretrained on the ImageNet and GTSRB



### Methods: (in)consistency



**Kendall's W**

The Kendall's W was used for comparing BayLIME and LIME on their inconsistency

* this metric measures the agreement among raters (i.e. repeated explanations in our case)
* it ranges from 0 (no agreement) to 1 (complete agreement)

### Methods: (in)consistency

As the Kendall’s W only considers the discrete ranks of features, it cannot discriminate explanations with the same ranking of features but different importance vectors. ????

So to handle such situations, they introduce a new metric based on the index ofdispersion (IoD) of each feature in repeated runs. 

* a weighted sum of IoD of each feature’s ranks in repeated explanations, so that the IoD of a more important feature is weighted higher

NB: index of dispersion = variance-to-mean ratio



### Methods: (in)consistency

\* <u>procedure</u>:

​    \- select a set of BayLIME explainers with **different options and prior parameters**

​    \- for each, iterate the explanation of the given instance *k* times, and quantify the inconsistency 



### Methods: robustness

Let's move to the other criterion - the robustness to kernel settings

here the procedure is as follows

*  define a kernel width settings interval $[l_{lo}, l_{up}]$ 
* randomly sample from that interval 5000 ***\*pairs of kernel width parameters\****
* for each pair, calculate the "distance" of the 2 explanations 
  * the norm of the difference between the importance vectors for the 2 given kernel width
* obtain a sample set of ratios between the 2 distances of explanations and the kernel width pair
* its ***\*median value\**** provides insights on the general robustness

​      ($\rightarrow$  this definition of robustness weaker empirical global Lipschitz value)



\---

### Methods: explanation fidelity



It remains to assess the last criterion, the explanation fidelity 

We want the XAI method to explain the true cause of the underlying model’s prediction. So the actual causality is taken here as an indicator for the explanation fidelity in two ways

* with 2 causal metrics:

1. **deletion**: starting with a complete instance and then gradually removing top-important features the deletion metric measures the decrease in the probability of the pre- dicted label.
   * intuitively, the removal of the “cause” (important fea- tures) will force the underlying AI model to change its decision
   * A sharp drop and thus a low Area Under the probability Curve (AUC), as a function of the fraction of removed features, suggests a good explanation that captures the real causality. 
2. in complement, the **insertion** metric measures the increase in the probability as more and more important features are introduced. Here higher AUC indicates a better explanation.

* the other way to evaluate explanation fidelity is through **neural backdoors**

The major difficulty of evaluating fidelity is the lack of ground truth behaviour of the underlying AI model. As the most important features that cause deliberate misclassi- fication, backdoor triggers provide such ground truth and therefore should be highlighted by a good XAI method. ?????

[if a feature is important -> identified as a trigger -> should have high value in the importance vector of the XAI]

To measure the success in highlighting the trigger, they use the Intersection over Union (IoU) metric.

Simply stated, given a bounding box around the true trigger area and the area highlighted by the XAI method, we calculate the ratio between the overlapped area and the area of their union.

IoU is an established metric originally designed for object detection. It ranges from 0 to 1, where the higher value the better. 

The drawback of IoU is that it only considers the overlapping of the highlighted area and the ground truth. Thus, it ignores how geometrically closed they are when there is no overlapping. So the authors develop a complementary metric called Average Minimum Distance (AMD) which measures the distance between the highlighted area (i.e. the explanation) and the ground truth. The value is s normalised to [0, 1] to make independent of the image dimensions.

It put here it in brackets, because it is presented in the Appendix of the article.



### Methods: how to obtain prior knowledge ?

so far so good... BayLIME is designed for integrating prior knowledge; but how can we elicit prior knowledge ?

Naturally it is an application-specific question, and remains challenging for Bayesian models. 

They implement 3 ways of getting priors for their experiments, that can also give inspiration for other contexts.

1. The first one is using explanations of a set of similar instances (to the instance under explanation) to form the prior know- ledge.
   * but needs a definition of what is a “similar” instance
   * used as illustrative example; for the 2 first research questions (robustness and consistency)

2. The other way is with other XAI techniques

   - as a matter of fact, there are many XAI tools; explanations by some other diverse XAI explainers based on fundamentally different theories to LIME (e.g., gradient-based vs perturbation- based, global vs local) may provide useful prior knowledge
   - they implement it by using GradCAM  results as the priors

   This way of implementing prior was used for assessing explanation fidelity, so the 3d research question.

3. The last implemented way to obtain prior is with Validation and Verification (V&V) methods 

   - V&Vs methods directly analyse the behaviour of the underlying AI model
   - this may in- dicate the importance of certain features, yielding priors required by BayLIME
   - for example, when explaining a prediction made by an infected model, (imperfect) detection tools may provide prior knowledge on possible backdoor triggers.
   - this scenario is implemented using the results of  NeuralCleanse

   This way of implementing prior was also used for assessing explanation fidelity, so the 3d research question.



*Grad-CAM is a method which generates visual*
*explanations via gradient based localization. To do so, it*
*extracts the gradients from the last convolution layer of the*
*network. The intuition behind this method is that the layer prior to the classification retains the information of feature*
*relevance while maintaining spatial relations, and therefore it*
*can generate a heatmap (based on a weighted combination of*
*activation maps dependent on gradient score) which highlights*
*the features with a positive influence for the specific class*
*which is chosen as the prediction.*

*NeuralCleanse* 

*he first robust and generalizable detection and*
*mitigation system for DNN backdoor attacks. Our techniques*
*identify backdoors and reconstruct possible triggers.*

*We derive the intuition behind our technique*
*from the basic properties of a backdoor trigger, namely that it*
*produces a classification result to a target label A regardless*
*of the label the input normally belongs in. Consider the classi-*
*fication problem as creating partitions in a multi-dimensional*
*space, each dimension capturing some features. Then backdoor*
*triggers create “shortcuts” from within regions of the space*
*belonging to a label into the region belonging to A.*

*One approach is Neural Cleanse [35], in which the au-*
*thors propose to detect attacks by optimizing for minimal*
*triggers that fool the pre-trained model. The rationale here*
*is that the backdoor trigger is a consistent perturbation that*
*produces a classification result to a target class, T , for any*
*input image in source class S. Therefore, the authors seek a*
*minimal perturbation that causes the model to classify the*
*images in the source class as the target class. The opti-*
*mal perturbation then could be a potential backdoor trigger.*
*This promising approach is computationally demanding as*

*the attacked source class might not be a priori known, and*
*such minimal perturbations need to be calculated for poten-*
*tially all pairs of source and target classes. Besides, a strong*
*prior on the type of backdoor trigger is needed to be able to*
*discriminate a possibly benign minimal perturbation from*
*an actual backdoor trigger*

*Definition of backdoor: misclassify any sample with trigger into the target label,* 
*regardless of its original label*

### Results: consistency improvement

Let's look at the result that looks at compares consistency of BayLIME with the one of LIME.

These curves  show the Kendall’s W measurements as a function of the perturbed sample size n. Each panel is for different set of parameter settings.

In red, we have the curve for LIME. We observe very low consistency when n is relatively small (e.g., n < 200); an issue that was already reported in previous studies. 

Non-informative BayLIME is indistinguishable from LIME. This is not suprising as both of them only exploit information from the randomly generated samples.

The smaller the sample size is, the greater the randomness. This explains the increase in consistency when n increases.

BayLIME  with partial or full prior knowledge does not have this issue of the sample size.

To assess how different priors can affect the consistency, we need to look at the $\lambda/\alpha$ value. This can be viewed as a regularization term:

- a larger value (small $\alpha$) penalizes the training data -> the prior knowledge dominates the posterior ??????
- a small value (small $\lambda$)  means no penalty on the data -> the posterior is dominated by the new observation

So we look now at the yellow curves that can have variable alpha and lambda parameter settings.

In the A and C figures, this $\lambda/\alpha$ ratio is the same (=20). This means that how much the prior or the new data contributes to the posterior is fixed; so the ability of "averaging out" sampling noise is also fixed.

When the ratio increases (fig B and D, $\lambda/\alpha$  = 200), the prior dominates more and more the posterior, and so ability of averaging out sampling noise is even stronger. This explains the higher level of the yellow curves in B/D compared to A/C

As for BayLIME with partial informative priors (blue curves), we observe worse consistency with smaller λ (e.g. in C vs. D). Here also: smaller $\lambda$ means less contributions from the priors to the posteriors, meaning with less ability to average out the randomness in the samples.

when n → +∞, all plots will eventually converge (to the measurement based on MLE using infinitely large samples)

*dans l'équation: lambda multiplie le mu_0 et alpha multiplie le betaMLE (the new samples)*

*gaussian prior of beta governed by the precision parameter lambda*

*covariance of the corresponding posterior distribution over w*

*SN-1= αI + βΦ T Φ.*  [lambda*I + alpha...]

*Maximization of this posterior distribution with respect to w is therefore equiva-*
*lent to the minimization of the sum-of-squares error function with the addition of a*
*quadratic regularization term, corresponding to (3.27) with λ = α/β.*

[in the notation used in BayLIME: alpha/beta -> lambda/alpha]

## Results: robustness to kernel settings improvement



Here we have the boxplot of the metric introduced earlier that quantifies the robustness to kernel settings. 

High value indicates reduced robustness. 

Here again LIME and BayLIME with non-informative priors exhibit similar robustness, since there is no prior knowledge being considered. They exhibit the worst result.

As before, it is insightful to look how the $\lambda/\alpha$  ratio affects the robustness.

When the contribution from the priors (that is independent from kernel setting), dominates over the contribution from the new data (that is sensitive to kernel settings), which means large $\lambda/\alpha$ value, the robustness increases.

In the case of the full prior, we can see it by looking at the trend of the three last boxes that represent an increasing ratio (ratio=200, 400, 2000).

The same holds for the 3 boxes of partial informative priors, which show higher robustness when $\lambda$ increases.

## Results: explanation fidelity - XAI methods

Now we can move to the last research question. We assess here the explanation fidelity of BayLIME compared to another XAI method GradCAM

Here we have two set of examples for the AUC scores that are shown;

With 

- GradCAM on the top - which represent the prior used for BayLIME

- LIME in  the middle  - which only use the newly generated data

- and BayLIME at the bottom which, in a certain manner, combines the two

On the left we have the scores for the deletion

* BayLIME provides the smallest value; LIME the highest
* recall that a good explanation is represented by a sharp drop in the probability, so a low AUC -> so BayLIME performs better

On the right we have the scores for the insertion metric

* BayLIME provides the greatest value; LIME the smallest
* here is it the reverse: the probability should rapidly increase, hence higher AUC means better explanation -> here again performs better

*deletion decrease in the probability of the predicted label when starting with a complete instance and then gradually removing top-important features*; *good explanation = sharp drop (low AUC as a function of the fraction of removed features)*

*insertion: increase in the probability as more and more important features are introduced*; *good explanation = higher AUC*

*Figure 4: Two sets of examples (n = 300), and the average AUC (based on 1000 images per value of n) of the deletion (smaller is better) and insertion (bigger is better) metrics of BayLIME comparing with other XAI methods.*

## Results: explanation fidelity - XAI methods

Then we can look at how these average AUC scores vary along with $n$

We observe that

* BayLIME performs better than SHAP and LIME (smaller deletion AUCs, higher insertion AUCs)
* but there is a converging trend when n increases

This highlights the importance of the prior when the number of samples is small.

We also see that 

* GradCAM is not a function of $n$ and only performs  when $n$ is extremely small (even smaller than the number of features)

 We conclude that, compared to the other 3 XAI methods, BayLIME has better fidelity in the middle and most practical range of n

*SHAP (Shapley Additive Explanations) by Lundberg and Lee ([2016](https://dl.acm.org/doi/10.5555/3295222.3295230)) is a method to explain individual predictions, based on the game  theoretically optimal Shapley values. Shapley values are a widely used  approach from cooperative game theory that come with desirable  properties. The feature values of a data instance act as players in a  coalition. The Shapley value is the average marginal contribution of a  feature value across all possible coalitions[[1](https://christophm.github.io/interpretable-ml-book/shap.html#shap-feature-importance)].*

*https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670*

## Results: explanation fidelity - V&V methods

The V&V tool NeuralCleanse yields reversed triggers as the approximation of backdoor, which are far from perfect (as we can see on the figure (1st row, 2nd column), with IoU = 0);  despite being very closed to.

Even directly apply LIME on an attacked image may provide a better IoU than NeuralCleanse. 

Nevertheless, BayLIME - which considers both the reversed triggers and the surrogate model - performs the best. 

We can see it here on the 2 images and in the table which shows to average IoU of 500 random images.





 As shown in Table 2, statistics on averaging the IoU of 500 random images confirm the above observation (NeuralCleanse is independent from individual images).
Table

Figure 5: Two examples and their IoU measurements of BayLIME using V&V results as priors (reversed triggers by NeuralCleanse), compared to the results of NeuralCleanse and LIME applied individually. LIME and BayLIME use the top-few features (where each feature is a fixed size square superpixel, cf. Appendix A.2 for details) such that the total number of pixels is the same as the ground truth. NB: We omit the background image to highlight the triggers only.



### Conclusion

BayLIME:

* is the first to exploit prior knowledge for better consistency, robustness to kernel settings and explanation fidelity

* improves over LIME

* the prior knowledge is independent from the causes of inconsistency and unrobustness (thus benefits both properties)

* improve fidelity by including additional useful information 

* performs better than V&V and other XAI methods





$\Rightarrow$ a way to obtain better explanations of AI models



$\Rightarrow$ a (Bayesian) way to inject knowledge in AI model interpretation (but defining good priors remains challenging !)



### Appendix



a) GradCAM has several inherent problems [Chattopadhay et al., 2018], thus it generally performs worse compared to others with larger sample size (e.g., n > 100 in our cases). Especially for ResNet50, our experiments show that Grad- CAM performs noticeably worse, even when comparing to other explainers in the case when they are working with smaller sample size. We conjecture that this is due to the known major drawback of GradCAM – the up-sampling process to create the coarse heatmap results in artefacts and loss of signal. ResNet50 has a severer problem here since it requires a relatively more precise up-sampling process Xception.



IoU only con-siders the overlapping of the highlighted area and the ground truth, ignoring how geometrically closed they are when there is no overlapping. We believe the distance between the highlighted area and the ground truth may also provide some insights on the fidelity, thus we develop a complement- ary metric to the IoU, called Average Minimum Distance (AMD). AMD measures how close the explanation is to the ground truth:



w as shown in the AMD column of Table 4, BayLIME again shows better performance than LIME, although the prior by NeuralCleanse shows the best performance. This is unsurprising due to the unique character of NeuralCleanse that only a “minimised trigger” will be identified to be as closed to the ground truth as possible, but at the risk of completely missing it (i.e., without any overlapping as reflected by IoU).
Table 4: Statistics on IoU (higher is better) and AMD (smal- ler is better) based on 500 backdoor-attacked images. The priors (derived from reversed triggers) are shown in Fig



TABLE !!!



the inconsistency is a weighted sum of IoD of each feature’s ranks in repeated explanations, so that the IoD of a more important feature is weighted higher. While there could be a range of similar metrics that can be defined, we choose the most intuitive one to use in this paper. Since our main purpose is to complement Kendall’s W in some corner cases and double check our experimental conclusions, rather than to justify any particular metric.
An



Fig.s 9 and 10 show more experimental results of both Kend- all’s W and the inconsistency metric Eq. (15) based on more instances, e.g., images, confirming our conclusions on RQ1. As expected, the plots of metric Eq. (15) can be interpreted similarly as Kendall’s W. But Kendall’s W only considers the (discrete) ranks of each feature and treats all features



equally. In contrast, our metric Eq. (15) weights the IoD of the discrete ranks of each feature by its (continuous) import- ance. With the extra information considered (compared to Kendall’s W), our metric Eq. (15) thus complements Kend- all’s W in terms of: (i) discriminating explanations with the same ranks of features but the importance vectors are differ- ent; (ii) minimising the effect from the normal fluctuation of the ranks of less-important/irrelevant features.





The intuition behind LIME is as follows. For a given black-box AI model (model-agnostic), we may probe it as many times as possible by perturbing some fea- tures (e.g., hiding superpixels of an image) of the input instance of interest (locally) and see how the prediction changes. Such changes in the predictions, as we vary the features of the instance, can help us understand why the AI model made a certain prediction over the instance. Then a new dataset consisting of the perturbed inputs and the cor- responding predictions made by the black-box AI model can be generated, upon which LIME trains a surrogate model that is interpretable to humans (e.g., linear regressors, cf. [Molnar, 2020, Chpt. 4] for more). The training of the in- terpretable surrogate model is weighted by the proximity of the perturbed samples to the original instanc



1) Choose an instance x to interpret (e.g., an image or a row
from a tabular dataset) in which there are m features7 (e.g., the superpixels for images or columns for tabular data).
2) Do perturbation on x (e.g., switch on/off image super- n samples, i.e. X = (xij) ∈ Rn×m. pixels) and generate a new input set X = {x1, ...,xn} with
3) Probe the black-box AI model with X and record the predictions as a column vector y = [y1, ..., yn]T.
4) Weight the n perturbed samples in X according to their proximity to the original instance x. Say the weights calcu- lated by some kernel function (by default, an exponential kernel defined on some kernel width) are {w1, ..., wn}, and denote the new weighted dataset8 as (X?, y?).
5) On the dataset (X?, y?), train a linear regressor y? =X?β + ?
(16) where β and ? are the coefficients and Gaussian white noise.
6) Then, the absolute values of the coefficient vector β represent the importance of the m features, based on which rankings can be done. By default, LIME 9 uses the Ridge regressor with weighted samples:
β = (XTWX + rI)−1XTWy (17)
where r is the Ridge regularisation parameter, and W = diag(w1 , ..., wn) is a diagonal matrix with diagonal ele- ments equal to those wi’s. It becomes the weighted ordin- ary least squares estimate β = (XTWX)−1XTWy when r = 0.
7