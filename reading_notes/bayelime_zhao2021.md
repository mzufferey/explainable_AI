## Bayesian Local Interpretable Model-Agnostic Explanations - Zhao et al. 2021

#### <u>Intro</u>

Extension to the LIME framework, BayLIME exploits prior know- ledge and Bayesian reasoning to improve both the **consistency** in repeated explanations of a single prediction and the **robustness** to kernel settings



**Explainable AI (XAI)** = a research field that aims at improving the trust and transparency of AI; goal: provide good explanations



several definitions of "goodness"; most important criterion: **fidelity** = how truthfully the explanation represents the unknown behaviour underlying AI decision 



Bayesian extension to the Local Interpretable Model-agnostic Explanations (**LIME**) = most popular method of the class of XAI methods that use local surrogate models for explaining individual predictions



°NB: A global surrogate model is an interpretable model that is trained to approximate the predictions of a black box model (Molnar 2021)°



LIME is one of the few XAI techniques that work for all tabular data, text and images 



LIME most significant weakness: **lack of consistency** in repeated explanations of a single prediction and **robustness** to kernel settings. 



BayLIME utilises a Bayesian local surrogate model =  a **“Bayesian principled weighted sum” of the prior knowledge and the estimates based on new samples** (as LIME)

- weights = the parameters with dedicated meanings that can be either automatically fitted from samples (Bayesian model selection) or elicited from application-specific prior knowledge (e.g., V&V methods)



BayLIME significantly improves over LIME: because **the prior knowledge is independent from the causes of  inconsistency and unrobustness** (thus benefits both properties) and also includes additional useful information that improves the fidelity



LIME: the time required to produce an explanation is dominated by the complexity of the black-box AI model. To improve LIME’s efficiency, we cannot effectively tune other LIME arguments, rather the best option is to **limit the number of queries** made to the AI model (which, indeed, is very costly when the AI model is deep) when generating labels for the *n* samples 



#### <u>Consistency</u>

LIME **inconsistency** (= different explanations can be generated for the same prediction)

- caused by the randomness in generating perturbed samples (of the instance under explanation) that are used for the training of local surrogate models (a smaller sample size leads to greater uncertainty of such randomness; but enlarging sample size might be impractical in real-world applications)
- limits its usefulness in critical applications (e.g. medical domain)



Compare  BayLIME with LIME on their inconsistency: **Kendall’s W** 

- measure the agreement among raters (i.e. repeated explanations in our case)
- ranges from 0 (no agreement) to 1 (complete agreement)
- considers the discrete ranks of features => cannot discriminate explanations with the same ranking of features but different importance vectors



Complement Kendall’s W in such corner cases with a new metric based on the **index of dispersion** (IoD) of each feature in repeated runs

- weights the IoD of the rank of each feature by its importance

  

#### <u>Robustness to kernel settings</u>

LIME issue in defining the right **locality** (= the “neighbourhood” on which a local surrogate is trained to approximate the AI model) 

- challenging to choose a kernel setting in LIME to properly define a neighbourhood is challenging
- explanations can be easily turned around by changing kernel settings



LIME uses an **exponential smoothing kernel to define the neighbourhood** with a given kernel width that determines how large the neighbourhood is. 

- a smaller kernel width = an instance needs to be closer to influence the local model (vice versa)
- no effective way exists to find the best kernel settings
- best strategy for now:  try different kernel settings and see if the explanations make sense (-> subject to errors/bias)
- an explanation may be easily turned around by changing the kernel settings



introduce a **weaker empirical notion of the robustness to kernel settings**:

- after a few trials, e.g., by cross-validation, it is not hard to figure out empirically a bound [llo, lup] as the **range of all possible kernel** settings in which we may perturb the kernel width to compute the robustness -> focus on the **median value** which is a much easier quantity to estimate than the optimised solution 

#### <u>Explanation fidelity</u>

**actual causality** as an indicator for the explanation fidelity in the following two different ways.

1. 2 causal metrics (no need for large-scale human annotation and avoid human bias)
   - **deletion**: measures the **decrease in the probability of the predicted label** when starting with a complete instance and then gradually removing top-important features (according to the ranked list obtained from XAI methods) 
     - rationale: the removal of the “cause” (important features) will force the underlying AI model to change its decision
     - good explanation capturing real causality will be indicated by a sharp drop (= low Area Under the probability Curve AUC) as a function of the fraction of removed features 
   - **insertion**: measures the increase in the probability as more and more important features are introduced
     - higher AUC indicating a better explanation. The two metrics not only 
2. **neural backdoors**
   - The major difficulty of evaluating fidelity is the **lack of ground truth** behaviour of the underlying AI model. As **the most important features that cause deliberate misclassification**, backdoor triggers provide such ground truth and therefore should be highlighted by a good XAI method

#### <u>BayLIME: bayesian retrofit of LIME</u>



##### <u>Embedding prior knowledge in LIME</u>

The local surrogate model is **trained on weighted samples perturbed around the instance of interest**, with weights representing their respective proximity to the instance 

=> **modify the posterior estimate** which is now essentially **a weighted sum of µ~0~ and β~MLE~**: a **Bayesian combination of prior knowledge and the new observations**. The weights are proportional to: 

1. the **“pseudo-count” of prior sample size** based on which we form our prior estimates µ~0~
2. the **“accurate-actual-count” of observation sample size**, i.e. the actual observation of the *n* new samples scaled by the precision *α*.



In brief, prior embedding as follows:

1. based on λ data-points prior to our new experiment, we **form our prior estimate of µ~0~**
2. in the experiments, collect *n* new samples; consider their precision (α) and weights (w~c~) for **forming a MLE estimate β~MLE~**
3.  **combine µ~0~ and β~MLE~ according to their proportions of the effective samples size** (λ and *αw~c~n*, respectively)
4. confidence in the new posterior estimate captured by all effective samples, i.e. *λ + αw~c~n* (the **posterior precision**)

It follows that

- Smaller λ means lower confidence in the prior knowledge, thus the posteriors are mostly dominated by the new observations, and vice versa. 
  - When λ ~ 0: µ~n~  (the posterior estimate on  β)  reduces to MLE, i.e. “let the data speak for themselves”.
  - If n ~ 0 (or equivalently α ~ 0, w~c~ ~ 0), then the β~MLE~ estimate from the new data is negligible and the prior knowledge dominates the posteriors.



**Why integrating prior knowledge via the "weighted sum" improve the three aforementioned properties ?** are: 

1) **“average out” the randomness** (β~MLE~ is function of the *n* randomly perturbed samples that causes inconsistency, µ~0~ independent from the cause) => **improve consistency**
2) “average out” the effect from kernels (β~MLE~ is a function of the weights*W* that depends on the choice of kernel settings, µ~0~ independent from kernel settings)  => thus **improve robustness**. 
3) µ0 normally contains added **diverse information** to β~MLE~ (black-box queries) that benefits the **explanation fidelity**



##### <u>The BayLIME framework</u>

 (µ0 = mean vector; λ = precision)

1. **non-informative priors**
   - µ~0~: zero mean vector
   - λ and α: fitted with Bayesian model selection . 
     - use the established algorithm to fit α and λ from data folllowing the algorithm of [MacKay, 1992, Tipping, 2001]. 
     - implicit solutions for λ and α: they are obtained by starting with initial values and then iterating over some interval equations until convergence.
2. **partial informative priors**
   - µ~0~ and λ: assume a known complete prior distribution 
   -  α: fitted from the data
     - similarly as 1), modify the Bayesian model selection algorithm by iterating α (but with fixed λ in this case) to maximise the log marginal likelihood until convergence
       BayLIME with full informative priors. 
3. **full informative priors**
   - µ~0~, λ and α: assumed known (ideal scenario)
     - direct implementation of the closed form of µ~n~ (the posterior estimate on β is)

##### **Obtaining prior knowledge** 

1. **Validation and Verification (V&V) methods** 
   - directly analyse the behaviour of the underlying AI model  -> may in- dicate the importance of certain features
   - e.g. explaining a prediction made by an infected model, (imperfect) detection tools may provide prior knowledge on possible backdoor triggers
2. **XAI techniques** 
   - explanations by other XAI explainers based on fundamentally different theories to LIME (e.g., gradient-based vs perturbation- based, global vs local) may provide useful prior knowledge
   - presumably, the drawbacks of individual XAI methods can be mitigated by the “hybrid” solution provided by BayLIME
3. **Explanations of a set of similar instances** (to the instance under explanation) 
   - still not rigorous enough (e.g., how to decide what is a “similar” instance ?), but serves as a first illustrative example



#### Evaluation

##### RQ1: **consistency** improvement  ?

- select a set of BayLIME explainers with **different options and prior parameters**
- for each, iterate the explanation of the given instance *k* times, and quantify the inconsistency according to **Kendall’s W**. 
- treat the sample size *n* as a variable (to confirm earlier observation that inconsistency is an even severer problem when *n* has to be limited for efficiency) 

##### RQ2: **robustness to kernel settings** improvement ?

- first, define an interval [l~lo~, l~up~] as the empirical bounds of all possible kernel width settings for the given application
- randomly sample from that interval 5000 **pairs of kernel width parameters** (fix the perturbed sample size to a sufficiently large number - indicated by RQ1 - to minimise the impact of inconsistency)
- for each pair, calculate the “distance” of the 2 explanations
- obtain a sample set of ratios between the 2 distances of explanations and the kernel width pair, on which statistics provides insights on the general robustness, e.g., the **median value**

The prior knowledge used in RQ1 and RQ2 is obtained from previous LIME explanations on a set of similar instances. 

- µ~0~ (prior mean vector): formed by the average importance of each feature in that set 
- λ: implied by the number of similar instances 
- α: unknown (then fitted from data) or assigned empirically (based on previous fitting of α on similar instances)

##### RQ3:  **explanation fidelity** improvement compared to some state-of-the-art XAI methods

2 scenarios (other diverse XAI and V&V methods respectively)

1. using diverse **XAI**

- gradient-based GradCAM to obtain “grey-box” information of the underlying CNN as the prior
- combine such extra knowledge with the “black-box” evidence as utilised by LIME
- compare the explanation fidelity of BayLIME with LIME, GradCAM and SHAP, via the **deletion and insertion** metrics

2. using **V&V methods**

- aim at explaining the behaviour of infected CNNs (BadNet and TrojanAttack, both trained on GTSRB with backdoors)
- approximate the location of triggers with the backdoor detector NeuralCleanse 
- BayLIME considers such knowledge as priors to provide a better explanation on the attacking behaviour, compared to NeuralCleanse and LIME applied solely

For RQ3, choose the more practical option – BayLIME with partial priors, i.e., α fitted, intuitive rationales for  µ~0~ and λ.

#### **Results and discussion** 

##### RQ1 (consistency)

**Kendall’s W** as a function of the perturbed sample size *n*

- Kendall’s W increases quickly but very low consistency when *n* is relatively small (e.g., *n* < 200)
- non-informative BayLIME indistinguishable from LIME (both only exploit the information from the    randomly generated samples)
- randomness: the results of sampling cannot be duplicated if the process were repeated
- the more sparse the samples are, the greater randomness presents in the dataset
-  monotonic trends as *n* increases for both LIME and non-informative BayLIME 
- by contrast, **BayLIME with partial/full informative priors can “average out” the sampling noise** in the new generated data by combining the information from the priors



How effectively BayLIME with different priors affects the consistency ?

- use the auxiliary of the factor **λ/α** 
  - **regularization coefficient** in Bayesian linear regressors
  - larger value penalises more on the training data - to control over-fitting
  - when α~0, 
    - λ/α → +∞
    - huge penalty on the data
    - => posteriors dominated by the prior knowledge
  - when λ~0, 
    - λ/α → 0 
    - no penalty on the data
    - => the posteriors dominated by the new observations

- λ/α = 20:
  - identical curves for BayLIME with full informative priors 
  - once λ/α fixed, the “proportion” of contributions to the posteriors by the priors and the new data is fixed 
  - = given *n* samples, the ability of “averaging out” sampling noise by the priors is fixed
- when λ/α increases to 200
  - ability of averaging out sampling noise even stronger
  - Kendall’s W measurements in this case > that in the case of λ/α = 20.
- BayLIME with partial informative priors: smaller λ results in worse consistency
  - smaller λ implies less contributions from the priors to the posteriors
  - = less ability to average out the randomness in the samples.

As *n* increases, uni-modal pattern with a minimum point for BayLIME with partial/full informative priors 

- tension between the perfectly consistent prior knowledge (does not change at all in repeated explanations) and quite consistent MLE based on large samples
- there must be a “balance-point” in-between that compromises both ends of the tension, yielding a minimised consistency
- when n → +∞,  convergence of all plots (to the measurement based on MLE using infinitely large samples)



##### RQ2 (robustness to kernel settings)

LIME and BayLIME with non-informative priors exhibit similar robustness, since there is no prior knowledge being considered, rather the data solely determines the ex- planations of both. 

Obvious **improvement on the robustness to kernel settings** when either partial or full prior knowledge is taken into account, we observe an . 

How varying the λ and α affects the robustness ? Boils down to how much contribution from the priors (independent from kernel setting), compared with the contribution from the new data (sensitive to kernel settings), to the posteriors (cf. λ/α)


##### RQ3a (explanation fidelity - XAI)

GradCAM provides fundamentally different grey-box information as priors to the black-box LIME. 

As expected, BayLIME performs better than both GradCAM (the prior) and LIME (new observations), thanks to its unique advantage of incorporating diverse information (cf. AUC scores of the dele- tion and insertion metrics).

Statistics on the average scores (varying *n*):

- BayLIME performs better than SHAP and LIME, converging trend when *n* increases
- GradCAM is not a function of *n* and only performs better in the corner cases when *n* is extremely small (even < than the number of features). 

=> compared to the other 3 XAI methods, BayLIME has **better fidelity in the middle and most practical range of *n*** 



##### RQ3b (explanation fidelity - V&V)

NeuralCleanse yields reversed triggers as the approximation of backdoor, which are far from perfect. Even directly apply LIME on an attacked image may provide a better IoU than NeuralCleanse. 

**BayLIME performs the best after considering both the reversed triggers and a surrogate model as LIME** 



#### Related work

Previous works for improving LIME

- modify the perturbed sampling method of LIME to cope with the correlation between features
- reinforcement learning to select a subset of perturbed samples to train the local surrogate model
- local linear regressors along with random forests that determines weights of perturbed instances to optimise the local linear model.
- replace the random perturbation with deterministic methods 
- employ an auto-encoder as a better weighting function for the local surrogate model

In comparison to these, BayLIME:

-  Bayesian way of embedding prior knowledge
  - direct experimental comparisons between methods not sensible: prior knowledge plays an important role in BayLIME that cannot be represented by other two methods)
- can deal with strict efficiency constraints (i.e., with small perturbed sample size n) 
- may improve explanation fidelity

Only two other model-agnostic XAI techniques with a Bayesian flavour:

-  derive generalised insights for an AI model through a global approximation (by a Bayesian non-parametric regression mixture model with multiple elastic nets) (Guo et al., 2018) 
  - fits conjugate priors from data,
  - normally requires a large sample size for the inference on a large number of model parameters.
- use the posterior credible intervals to determine an ideal sample size *n*  (Slack et al., 2020)

BayLIME is the first to exploit prior knowledge for better consistency, robustness to kernel settings and explanation fidelity.



#### Discussion

How defining of good priors ?

- prior that truly/accurately reflects the unknown behaviour of the AI model?
  - sensible, but due to the lack of ground truth behaviours of the black-box AI model, we would never know the prior is good or not with certainty
- prior that gives an explanation that looks good to human users? 
  - “cheating” in a Bayesian sense: we cannot rig a prior to get a result we like – and this is not the purpose of XAI methods

However, we can be fairly confident that a given prior is good in some practical cases (e.g., derived from a V&V tool that was shown to be reliable in previous uses), so that it can be utilised by BayLIME.

What if we used a bad prior?

- BayLIME might end up providing an explanation that is “consistently and robustly bad”, which seems to imply that BayLIME is only useful when we are certain that the priors are good. 
- However, in practice, we can never be certain if a prior is good or bad, rather the prior is simply a piece of evidence to us. 
- Against the Bayesian spirit but also unwise to discard any evidence without sound reasons. If there is a proof that the evidence (either the priors or the new observations) is not trustworthy, then certainly we should not consider it in our inference – in this sense, all Bayesian methods depend on good priors, not just BayLIME. 
- So with some caution in deriving priors/evidence from trustworthy ways (a separate engineering problem that is normally out of the scope of a Bayesian method itself), we may ease the concern of BayLIME being “consistently and robustly bad”.

#### Conclusion and future work

The Bayesian mechanism of BayLIME helps in improving the consistency in repeated explanations of a single prediction, robustness to kernel settings, and the explanation fidelity.

In future work, we will investigate 

- more practical ways of obtaining priors 
- how to leverage the posterior confidence for better explanations
- develop novel use cases of BayLIME

