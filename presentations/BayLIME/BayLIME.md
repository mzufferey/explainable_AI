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

TO CHANGE WITH https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/


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
  
## BayeLIME: LIME in a Bayesian way

* Bayesian linear regression framework  
* for $n$ samples $X$ and $m$ features, the **posterior distribution of the mean vector of the coefficient vector $\beta$** is given by:			
$$
 µ_n = (λI_m + αX^TWX)^{−1}λI_mµ_0 +\\(λI_m + αX^TWX)^{−1}αX^TWXβ_{MLE}
$$

<span style="font-size:20px;">

  $µ_n$ = the posterior mean vector of $\beta$; $µ_0$  = the prior mean vector of $\beta$; 
  $β$ = the coefficient vector for the features; $β_{MLE}$ = ML estimates for the linear regression model; 
  $α$ = the precision parameter (noise in the linear assumption); $\lambda$ = the precision parameter that governs the prior

</span>


$\rightarrow$ a <u>weighted sum of $µ_0$ and $β_{MLE}$</u>: **Bayesian combination of prior knowledge and the new observations**


---
  
## BayeLIME: LIME in a Bayesian way


<span style="font-size:20px;">

$$
 µ_n = (λI_m + αX^TWX)^{−1}λI_mµ_0 +\\(λI_m + αX^TWX)^{−1}αX^TWXβ_{MLE}
$$

</span>					

one sample and one feature case:


<span style="font-size:20px;">

$$
 µ_n = \frac{\lambda}{\lambda+\alpha w_c n }\mu_0 + \frac{\alpha w_c n}{\lambda + \alpha w_c}\beta_{MLE}
$$

</span>					


$\Rightarrow$ a <u>weighted sum of $µ_0$ and $β_{MLE}$</u> where the weights are proportional to 
  1. $\lambda I_m$= the **"pseudo-count" of prior sample size** ($\lambda$) based on which $µ_{0}$ is formed
  <br>
  2. $αX^TWX$ = the **"accurate-actual-count" of observation sample size**, i.e. the actual observation of the *n* new samples 
  ($X^TWX$) scaled by their precision ($\alpha$)

---
## BayeLIME: procedure recap

1. before the new experiment, form the **prior estimate** of $µ_{0}$ based on $λ$ data points 
<br>
2. in the experiments, collect *n* new samples and consider their precision ($α$) and weights ($w_{c}$) for forming a **ML estimate** $β_{MLE}$
<br>
3. **combine $µ_{0}$ and $β_{MLE}$** according to their proportions of the effective samples size ($λ$ and $αw_{c}n$, respectively)
<br>
4. calculate the **posterior precision** captured by all effective samples (i.e. $λ + αw_{c}n$)


---
## BayeLIME: options for the priors

1. **non-informative** priors
   - $µ_{0}$: zero mean vector 
   - $λ$ and $α$: fitted with Bayesian model selection 

2. **partial informative** priors
   - $µ_{0}$ and $λ$: known distribution
   - $α$: fitted with Bayesian model selection
3. **full informative** priors (ideal scenario)
   - $µ_{0}$, $λ$ and $α$: known distribution


---
  
## BayeLIME: upsides


The combination between prior knowledge and new observations allows BayLIME to address LIME's weaknesses as it

$\rightarrow$ **improves consistency** by averaging out randomness <br>
$\rightarrow$  **improves robustness** by averaging out effects from kernels<br>
$\rightarrow$  **improves explanation fidelity** by combining diverse information


METTRE UNE IMAGE -> COMMENT LE PROUVER !! sth like REALLY ? believe
un truc genre voir pour croire saint

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
  * measure the agreement among raters (here: repeated explanations) 
  <br>
  * ranges from 0 (no agreement) to 1 (complete agreement) 
  <br>
  * cannot discriminate explanations with the same ranking of features but different importance vectors


---

## Methods: (in)consistency

* **Kendall's W** 
<br>
* new metric **based on the index of dispersion** (IoD) of each feature 
  * a weighted sum of IoD of each feature's ranks in repeated explanations
 
 
---

## Methods: (in)consistency

* **Kendall's W** 
<br>
* new metric **based on the index of dispersion** (IoD) 
<br>
* <u>procedure</u>:
    - select a set of BayLIME explainers with **different options and prior parameters**
    - for each, iterate the explanation of the given instance *k* times, and quantify the inconsistency 



---

## Methods: robustness to kernel settings
* <u>procedure</u>:
  * define a kernel width settings interval $[l_{lo}, l_{up}]$ 
  <br>
  * randomly sample from that interval 5000 **pairs of kernel width parameters**
  <br>
  * for each pair, calculate the "distance" of the 2 explanations
  <br>
  * obtain a sample set of ratios between the 2 distances of explanations and the kernel width pair
  <br>
  * its **median value** provides insights on the general robustness
      ($\rightarrow$ weaker empirical global Lipschitz value)
      

---
## Methods: explanation fidelity

* 2 causal metrics
   1. **deletion**: decrease in the probability of the predicted label when starting with a complete instance and then gradually removing top-important features
     - good explanation = sharp drop (low AUC as a function of the fraction of removed features)
   2. **insertion**: increase in the probability as more and more important features are introduced
     - good explanation = higher AUC
     <br>
* neural backdoors
  * blackdoor triggers $\rightarrow$ ground truth
  * metrics: IoU (and AMD)
  
---
## Methods: how to obtain prior knowledge ?

* explanations of a set of **similar instances** (RQ1+RQ2)
  - the average importance of each feature in that set collectively forms the prior mean vector
  <br>
 * **XAI techniques** (RQ3a)
   - explanations obtained from other XAI explainers 
   - here: GradCAM results as priors
  <br>
* **Validation and Verification (V&V) methods** (RQ3b)
   - direct analysis of the behaviour of the underlying AI model
   - e.g. detection tools may provide prior knowledge on possible backdoor triggers
   - here:  NeuralCleanse results as priors

---

## Results: consistency improvement



<!-- <img src="pictures/fig2.png" width="600" align=left> -->

<!-- <p style="margin-left:10cm"><style>
p {
  margin-left: 10cm;
}
</style>
<p> -->
![width:880px left](pictures/fig2.png)


* non-informative BayLIME indistinguishable from LIME, with monotonic increasing trends as *n* increases 
* by contrast, BayLIME with partial/full informative priors "averages out" the sampling noise
* $\lambda/\alpha$: large (resp. small) value = the prior is dominated by the prior knowledge (resp. new data)
  * when it increases, the ability of averaging out sampling noise is even stronger.  
* when $n → +∞$,  all converge to MLE





---
## Results: robustness to kernel settings improvement


![width:400px center](pictures/fig3.png)

<br>

* similar (un)robustness for LIME and BayLIME with non-informative priors 
* either partial or full prior knowledge improves robustness 
* increased robustness for large $\lambda/\alpha$


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
