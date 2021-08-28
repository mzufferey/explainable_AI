---
marp: true
theme: gaia
color: #000
colorSecondary: #333
backgroundColor: #fff
paginate: true
size: 4:3
---
<style>
section {
  font-size: 25px;
}
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>
<!-- _paginate: false -->
​
# BayLIME: Bayesian Local Interpretable Model-Agnostic Explanations


(Zhao et al. 2021)


---

## Explainable AI (XAI)


* research field aiming at improving trust and transparency of AI
* seeks to provide **good explanations** <span style="font-size:20px;">(Ribeiro et al. 2016a,b)</span>
  * *explanation*
    * the answer to a why-question <span style="font-size:20px">(Miller 2017)</span>
    * relates the feature values of an instance to its model prediction in a humanly understandable way <span style="font-size:20px">(Molnar 2021)</span>
  * *goodness*
    * **interpretability** = qualitative understanding between the input variables and the response
    * **fidelity** = how truthfully the explanation represents the unknown behaviour underlying AI decision 
    <span style="font-size:20px">(\+ possibly several other criteria (see e.g. Molnar 2021))</span>


---

## LIME: Local Interpretable Model-agnostic Explanations


* implementation of local surrogate models  <span style="font-size:20px;">(Ribeiro et al. 2016a)</span>
  * **surrogate** = **interpretable** models trained to **approximate** the predictions of the underlying black box model
  * **local** = focuses on training surrogate models to explain **individual** predictions
* **model-agnostic** = can be used for any ML model
* works for all tabular data, text and images 
* most popular XAI method
 <style>
/* Reset table styling provided by theme */
table, tr, td, th {
  all: unset;

  /* Override contextual styling */
  border: 0 !important;
  background: transparent !important;
}
table { display: table; }
tr { display: table-row; }
td, th { display: table-cell; }

/* ...and layout freely :) */
table {
  width: 100%;
}
td {
  text-align: center;
  vertical-align: middle;
}
</style>

<table>
  <tr>
    <td valign="top"> <span style="font-size:20px"><br><br>(Stiglic et al. 2020)</span></td>
    <td valign="top"><img src="pictures/toy_lime.jpeg" width=380></td>
  </tr>
 </table>

 <!-- 
![width:400px  margin:0](pictures/toy_lime.jpeg)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="font-size:20px;">(Stiglic et al. 2020)</span>
-->
 <!-- ![width:400px right](pictures/toy_lime.jpeg) -->

<!--
<p float="left">
  <img src="/img1.png" width="100" />
  <img src="/img2.png" width="100" /> 
  <img src="/img3.png" width="100" />
</p>
-->
---

## LIME: how does it proceed ?

<u>Aim</u>: identify an interpretable model over the interpretable representation that is locally faithful  <span style="font-size:20px;">(Ribeiro et al. 2016b)</span>

<u>For any black-blox model, the procedure in brief</u>:

1. select an **instance of interest**

2. **perturb the dataset**

3.  get **new predictions** from the black-box model for the perturbed samples

4. **weight** the perturbed samples according to their proximity to the instance of interest

5. train a **weighted, interpretable model** on the perturbed dataset 

6. **explain the prediction** by interpreting the local model



 <span style="font-size:20px;">([Molnar 2021, §5.8](https://christophm.github.io/interpretable-ml-book/lime.html))</span>

---

## LIME: how does it proceed ?

<u>Mathematically speaking...</u>

Obtain the explanation $ξ(x)$ by solving:

$$
ξ(x) = \underset{g∈G}{argmin}\textrm{ }L(f,g,Πx) + Ω(g) 
$$

<span style="font-size:20px;">

$G$ = class of potentially interpretable models; $g$ = explanation model; $f$ =model being explained;  $Π_{x}(z)$ = proximity measure; $L(f,g,Πx)$ = fidelity function; $Ω$ = complexity measure

</span>

Estimate $L$ by generating perturbed samples around $x$, making predictions with the black box model $f$ and weighting them according to $Π_{x}$  <span style="font-size:20px;">(Ribeiro et al. 2016b)</span>


$\rightarrow$ minimize $L(f,g,Πx)$ while having $Ω(g)$ be low enough &nbsp;
(*fidelity-interpretability trade-off*)

---

## LIME: how does it proceed ?

<u>... and visually speaking</u>

<table>
  <tr>
    <td valign="top" text-align="right"> procedure:</td>
    <td valign="top"><img src="pictures/art2_fig1.png" width=380></td>
  </tr>
  <tr>
    <td valign="top" text-align="right"> output:</td>
    <td valign="top"><img src="pictures/linardatos_fig4.png" width=500></td>
  </tr>
 </table>

<!--
![width:300px center](pictures/art2_fig1.png)
![width:500px center](pictures/linardatos_fig4.png)
-->


---

## LIME weaknesses

<br>

* **inconsistency**: different explanations can be generated for the same prediction
  * caused by the randomness in generating perturbed samples that are used for the training of local surrogate models
  * smaller sample size = greater uncertainty 
  * limits its usefulness in critical applications (e.g. medical domain)

******* ADD FIGURE 1

---

## LIME weaknesses

<br>

* **inconsistency**: different explanations can be generated for the same prediction
  * caused by the randomness in generating perturbed samples that are used for the training of local surrogate models
  * smaller sample size = greater uncertainty 
  * limits its usefulness in critical applications (e.g. medical domain)

* **unrobustness** to kernel settings: challenge of defining the "neighbourhood" of the instance of interest (step 4)
    * no effective way to find the best kernel settings
    * best strategy for now: "trial-error" (biases !)


---
  
## BayeLIME

<br>

* Bayesian modification of LIME

* a “Bayesian principled weighted sum” of the prior knowledge and the estimates based on new samples 
<br>


<u>Upsides</u>:

$\rightarrow$ **improves consistency** by averaging out randomness <br>
$\rightarrow$  **improves robustness** by averaging out effects from kernels<br>
$\rightarrow$  **improves explanation fidelity** by combining diverse information

---
  
## BayeLIME: LIME in a Bayesian way

* bayesian linear regression framework
* β is the coefficient vector of m features, and 
* α is the precision parameter (reciprocal of the variance) representing noise in this linear assumption. Then,
* µ_0  the prior mean vector of $\beta$

* µ_n  the posterior mean vector

βMLE = (XTWX)−1XTWy is the Maximum Likelihood Estimates (MLE) for the linear regression model [Bishop, 2006] on the weighted samples

µn = (λIm + αXTWX)−1λImµ0 + (λIm + αXTWX)−1αXTWXβMLE (9)

* µ_n  the posterior mean vector
is a weighted sum ofµ0 and βMLE – A Bayesian combination of prior knowledge and the new observations. 
the weights are proportional to 
  1. the **"pseudo-count" of prior sample size** based on which we form our prior estimates $µ_{0}$
  2. the **"accurate-actual-count" of observation sample size**, i.e. the actual observation of the *n* new samples scaled by the precision *α*

---
  
## BayeLIME in clear


special case of a single feature instance (m = 1) with a simplified kernel function that returns a constant weight
(µn with 1 feature)

can be simplified as:
?n i=1 x2 i ≈ n(1 +
λ
αwcn µ0 + βMLE (12)

---
## BayeLIME: procedure recap

1. form the **prior estimate** of $µ_{0}$ based on $λ$ data points 
2. collect *n* new samples and consider their precision ($α$) and weights ($w_{c}$) for forming a **MLE estimate** $β_{MLE}$
3. **combine $µ_{0}$ and $β_{MLE}$** according to their proportions of the effective samples size ($λ$ and $αw_{c}n$, respectively)
4. calculate the **posterior precision** captured by all effective samples (i.e. $λ + αw_{c}n$)



---
## BayeLIME: choice of the priors

* **non-informative** priors
   - $µ_{0}$: zero mean vector 
   - $λ$ and $α$: fitted with Bayesian model selection 

* **partial informative** priors
   - $µ_{0}$ and $λ$: known distribution
   - $α$: fitted with Bayesian model selection
* **full informative** priors (ideal scenario)
   - $µ_{0}$, $λ$ and $α$: known distribution


---

## Methods: BayeLIME validation

<!--
<ol style="line-height:200%">
<li> <b>consistency</b> improvement (vs. LIME)</li>
<li> <b>robustness</b> to kernel settings improvement (vs. LIME)</li>
<li> explanation <b>fidelity</b> improvement (vs. XAI methods)</li>
</ol>
-->

<br>

 RQ1. <b>consistency</b> improvement (vs. LIME)<br>
 RQ2. <b>robustness</b> to kernel settings improvement (vs. LIME)<br>
 RQ3. explanation <b>fidelity</b> improvement (vs. XAI methods)<br>

<br>

<u>Datasets</u>:
* Boston house-price dataset
* breast cancer Wisconsin dataset 
* a set of CNNs pretrained on the ImageNet and GTSRB

---




## Methods: (in)consistency


* **Kendall's W**
  * measure the agreement among raters (i.e. repeated explanations in our case)
  * ranges from 0 (no agreement) to 1 (complete agreement)
  * procedure:
    - select a set of BayLIME explainers with **different options and prior parameters**
    - for each, iterate the explanation of the given instance *k* times, and quantify the inconsistency 
---


## Methods: (in)consistency
Kendall's W considers the discrete ranks of features: cannot discriminate explanations with the same ranking of features but different importance vectors
* metric based on the **index of dispersion** (IoD) of each feature in repeated runs
  * weights the IoD of the rank of each feature by its importance
---

## Methods: robustness to kernel settings

* define a kernel width settings interval $[l_{lo}, l_{up}]$ 
* randomly sample from that interval 5000 **pairs of kernel width parameters**
* for each pair, calculate the "distance" of the 2 explanations
* obtain a sample set of ratios between the 2 distances of explanations and the kernel width pair
* its **median value** provides insights on the general robustness

---
## Methods: explanation fidelity

**actual causality** as an indicator for the explanation fidelity 

* 2 causal metrics
   - **deletion**: decrease in the probability of the predicted label when starting with a complete instance and then gradually removing top-important features
     - good explanation = sharp drop (low AUC as a function of the fraction of removed features)
   - **insertion**: increase in the probability as more and more important features are introduced
     - good explanation = higher AUC
* **neural backdoors**

---
## Methods: how to obtain prior knowledge ?

* explanations of a set of **similar instances** (RQ1+RQ2)
  - the average importance of each feature in that set collectively forms the prior mean vector
 * **XAI techniques** (RQ3a)
   - explanations obtained from other XAI explainers 
   - here: GradCAM results as priors
* **Validation and Verification (V&V) methods** (RQ3b)
   - direct analysis of the behaviour of the underlying AI model
   - e.g. detection tools may provide prior knowledge on possible backdoor triggers
   - here:  NeuralCleanse results as priors

---

## Results: consistency improvement

* non-informative BayLIME indistinguishable from LIME
  * monotonic trends as *n* increases for both LIME and non-informative BayLIME
* by contrast, BayLIME with partial/full informative priors "averages out" the sampling noise


<!-- <img src="pictures/fig2.png" width="600" align=left> -->

<!-- <p style="margin-left:10cm"><style>
p {
  margin-left: 10cm;
}
</style>
<p> -->



![width:880px left](pictures/fig2.png)



---
## Results: consistency improvement

How different priors affect consistency ?

* use the auxiliary of the factor $λ/α$ ("regularization coefficient")
 * when $α\simeq0$: huge penalty on the data
 * when $λ\simeq0$: no penalty on the data
* $λ/α = 20$: identical curves for BayLIME with full informative priors 
* when $λ/α$ increases to 200: stronger ability of averaging out sampling noise (higher Kendall's W)

when $n → +∞$,  all converge to the measurement based on MLE

---
## Results: robustness to kernel settings improvement

* similar robustness for LIME and BayLIME with non-informative priors 
* either partial or full prior knowledge improves robustness 
<br>

![width:400px center](pictures/fig3.png)

---
## Results: robustness to kernel settings improvement

How varying the $λ$ and $α$ affects the robustness ? 

* contribution from the priors (independent from kernel setting) $\leftrightarrow$  contribution from the new data (sensitive to kernel settings), to the posteriors (cf. $λ/α$)


---
## Results: explanation fidelity - XAI methods

* better performance than GradCAM and LIME

![width:500px center](pictures/fig4_top.png)

---
## Results: explanation fidelity - XAI methods

* by varying *n* (average scores):
  * better than SHAP and LIME, converging when *n* increases
  * GradCAM better only when *n* is extremely small
  

![width:600px center](pictures/fig4_bottom.png)

$\Rightarrow$ **better fidelity in the middle and most practical range of *n*** 


---
## Results: explanation fidelity - V&V methods

* NeuralCleanse yields reversed triggers as the approximation of backdoor, which are far from perfect
* even directly apply LIME on an attacked image may provide a better IoU than NeuralCleanse. 


![width:700px center](pictures/fig5_table2.png)

$\Rightarrow$ **better fidelity after considering both the reversed triggers and a surrogate model** 

<!--
---
## Conclusion: uniqueness of BayLIME

* previous attempts for improving LIME  <span style="font-size:20px;">(e.g. modified sampling method, deterministic methods instead of random perturbations, etc.)</span>; in comparison, BayLIME:
  * embeds prior knowledge in a Bayesian way
  * can deal with strict efficiency constraints <span style="font-size:20px;">(small $n$)</span>
  * improves explanation fidelity

* two other XAI techniques with Bayesian flavour:
  * using a global approximation, requires large sample size  <span style="font-size:20px;">(Guo et al. 2018)</span>
  * using the posterior credible intervals to determine an ideal sample size $n$  <span style="font-size:20px;">(Slack et al. 2020)</span>

$\Rightarrow$ **BayLIME is the first to exploit prior knowledge for better consistency, robustness to kernel settings and explanation fidelity**
-->


---
## Conclusion 

<u>BayLIME</u>:
* is the first to exploit prior knowledge for better consistency, robustness to kernel settings and explanation fidelity
* improves over LIME
  * the prior knowledge is independent from the causes of inconsistency and unrobustness (thus benefits both properties)
  * improve fidelity by including additional useful information 
* performs better than V&V and other XAI methods


$\Rightarrow$ a way to obtain better explanations of AI models

$\Rightarrow$ a (Bayesian) way to inject knowledge in AI model interpretation (but defining good priors remains challenging !)

---
## Appendix: inconsistency metric

![width:800px center](pictures/fig9_inconsistency.png)


---
## Appendix: inconsist. metric vs. Kendall's W

![width:400px center](pictures/fig10_inconsistency_vs_kendall.png)

---
## Appendix: IoU vs. AMD

![width:600px center](pictures/table4_IoU_vs_AMD.png)

---
## Appendix: 


---
## Appendix: 

---
## Appendix: 

---
## Appendix: 

---
## Appendix: 

---
## Appendix: 
