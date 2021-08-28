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

6. explain the prediction by interpreting the local model (e.g. the coefficients of the regressor as an indication of feature importance)

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



### BayeLIME

Now that we got an overview of LIME and its weaknesses; we can move to the method described in the article I am presenting, namely BayeLIME, which specifically aims at adressing the 2 main drawbacks of LIME

BayeLIME is a Bayesian modification of LIME.

In very few words, BayeLIME is a “Bayesian principled weighted sum” of the prior knowledge and the estimates based on new samples.

The rationale is that the combination between prior knowledge and the estimates based on new samples will allow to 

* improve consistency by averaging out randomness 
* improve robustness by averaging out effects from kernels
* improve explanation fidelity by combining diverse information



As for LIME, the local surrogate model is trained on weighted samples perturbed around the instance of interest, with weights rep-resenting their respective proximity to the instance.



the weights are proportional to
the “pseudo-count” of prior sample size based on which we form our prior estimates µ_{0}µ 
0
	

the “accurate-actual-count” of observation sample size, i.e. the actual observation of the n new samples scaled by the precision α











index ofdispersion (IoD) of each feature in repeated runs. The new metric weights the IoD of the rank of each feature by its importance,



this global Lipschitz value quantifies the robust- ness of an explainer to the change of kernel width. The computation of L can be seen as an optimisation problem

which, unfortunately, is very challenging to solve analytic- ally or to estimate numerically

To bypass the difficulty and still provide insights on the robustness defined earlier, we instead introduce a weaker empirical notion of the robustness to kernel settings:
Definition



Assume L1 and L2 are both random variables ofkernel width settings within [llo, lup]. Then, R is the me- dian2 (denoted as M(·)) ofthe ratio between the perturbed distances ofhis and the pair (L1, L2):
[eq]
which represents the average robustness to the kernel set-tings when explaining the instance i.

hi as the importance vector ofm features taking l as the kernel width setting. [function that takes l as parameter







use the metric Intersection over Union (IoU) to measure the success in highlighting the trigger. Given a bounding box around the true trigger area BT and the highlighted area by XAI meth- ods B?
T, the IoU is their overlapped area divided by the area
of their union, i.e., (BT ∩B?
T)/(BT ∪B?
T). IoU is an estab-
lished metric originally designed for object detection that ranges from 0 to 1, a higher IoU is better. It only considers the overlapping of the highlighted area and the ground truth, ignoring how geometrically closed they are when there is no overlapping.

. To complement this, we introduce a secondary metric





 α is the precision parameter (reciprocal of the variance) representing noise in this linear assumption.









Figure 2: Kendall’s W in k = 200 repeated explanations by LIME/BayLIME on random Boston house-price instances. Each set shows an illustrative combination of α and λ. Same patterns are observed on images, cf. Appendix B for more.
only



The red curves in Fig. 2 present the Kendall’s W measurements as a function of the perturbed sample size n. Although it increases quickly, we observe very low consistency when n is relatively small (e.g., n < 200). These results support our earlier conjecture on the inconsistency issue of LIME, especially when n has to be limited by an upper-bound due to efficiency considera- tions. Non-informative BayLIME is indistinguishable from LIME, since both of them are only exploiting the inform- ation from the n samples that generated randomly – the presence of randomness means that the results of sampling cannot be duplicated if the process were repeated. Naturally, the more sparse the samples are, the greater randomness presents in the dataset. Thus, LIME and non-informative BayLIME show monotonic trends as n increases.

Both the plots of BayLIME with full informative priors (yellow curves) in Fig. 2 (A) and (C) have a regularization



factor λ/α = 20, and are basically identical. This is because, once λ/α = 20 is fixed, the “proportion” of contributions to the posteriors by the priors and the new data is fixed. In other words, given n samples, the ability of “averaging out” sampling noise by the priors is fixed. When λ/α increases to 200, as shown by the yellow curves in Fig. 2 (B) and (D), such ability of averaging out sampling noise is even stronger, which explains why Kendall’s W measurements in this case are higher than the case of λ/α = 20. For BayLIME with partial informative priors (blue curves in Fig. 2), we observe that smaller λ results in worse consistency, e.g., Fig. 2 (C) vs (D). Again, Remark 1 applies here – smaller λ implies less contributions from the priors to the posteriors, meaning with less ability to average out the randomness in the samples.
Starting from a non-zero small number, as n increases, we can see BayLIME with partial/full informative priors may exhibit an uni-modal pattern, e.g., the Fig. 2 (C)5 with a minimum point. This represents a tension between the per- fectly consistent prior knowledge (does not change at all in repeated explanations) and quite consistent MLE based on large samples. There must be a “balance-point” in-between that compromises both ends of the tension, yielding a min- imised consistency. Finally, when n → +∞, it is trivial to see (e.g., by taking the limit of Eq. (12) as a function of n), all plots will eventually converge (to the measurement based on MLE using infinitely large samples).
RQ2.



Fig. 3 are box-and-whisker charts providing insights on the general robustness of eight AI explainers to kernel width settings, in which the median values defined by Eq. (2) are marked, as usual, by horizontal bars inside the boxes.



Figure 3: The general (un-)robustness of eight AI explainers to kernel settings (box-and-whisker plots without outliers).
Again, LIME and BayLIME with non-informative priors exhibit similar robustness, since there is no prior knowledge being considered, rather the data solely determines the ex- planations of both. In stark contrast, when either partial or full prior knowledge is taken into account, we observe an obvious improvement on the robustness to kernel settings. 

The regularisation factor λ/α and Remark 1 are still handy here in the discussions on how varying the λ and α affects the robustness – it all boils down to how much contribution from the priors (that is independent from kernel setting), compared with the contribution from the new data (that is sensitive to kernel settings), to the posteriors.



Figure 4: Two sets of examples (n = 300), and the average AUC (based on 1000 images per value of n) of the deletion (smaller is better) and insertion (bigger is better) metrics of BayLIME comparing with other XAI methods.



Fig. 4 first shows two sets of such examples via the AUC scores of the dele- tion and insertion metrics, respectively. Then statistics on the average scores (varying n) are shown in the last row of Fig. 4. We observe: (i) BayLIME performs better than SHAP and LIME, while there is a converging trend when n increases. This aligns with the second point in Remark 1. (ii) GradCAM is not a function of n (thus showing hori- zontal lines) and only performs better in the corner cases when n is extremely small (even smaller than the number of features). We conclude that, compared to the other 3 XAI methods, BayLIME has better fidelity in the middle and most practical range of n (e.g., 100∼400 in our case). For



Figure 5: Two examples and their IoU measurements of BayLIME using V&V results as priors (reversed triggers by NeuralCleanse), compared to the results of NeuralCleanse and LIME applied individually. LIME and BayLIME use the top-few features (where each feature is a fixed size square superpixel, cf. Appendix A.2 for details) such that the total number of pixels is the same as the ground truth. NB: We omit the background image to highlight the triggers only.

E.g., in the BadNet example of Fig. 5 (1st row), the reversed trigger completely missed the ground truth trigger (thus IoU = 0), despite being very closed to. Even directly apply LIME on an attacked image may provide a better IoU than NeuralCleanse. Nevertheless, BayLIME performs the best after considering both the reversed triggers and a surrogate model as LIME (trained on the same number of n samples). As shown in Table 2, statistics on averaging the IoU of 500 random images confirm the above observation (NeuralCleanse is independent from individual images).







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