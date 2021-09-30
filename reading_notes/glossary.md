## global surrogate

interpretable model trained to approximate the predictions of a black box model

- We can draw conclusions about the black box model by interpreting the surrogate model.

*(https://christophm.github.io/interpretable-ml-book/)*

## intrinsic interpretability

machine learning models that are considered interpretable due to their simple structure, such as short decision trees or sparse linear models

*(https://christophm.github.io/interpretable-ml-book/)*



## local surrogate 

interpretable models that are used to  explain individual predictions of black box machine learning models (e.g. Local interpretable model-agnostic explanations - LIME)

- <a href="LIME_def_molnar2021.md">summary</a> of the explanation from Molnar 2021

*(https://christophm.github.io/interpretable-ml-book/)*

## post-hoc interpretability

application of interpretation methods after model training (e.g. permutation feature importance); can also be applied to intrinsically interpretable models

*(https://christophm.github.io/interpretable-ml-book/)*

## model-agnostic

separated from the machine learning model (e.g. model-agnostic interpretation methods) 

*(https://christophm.github.io/interpretable-ml-book/)*



## RLIPP score

To quantitatively determine important subsystems for drug response prediction, we adopted the Relative Local Improvement in Pre- dictive Power (RLIPP) score as described previously for DCell (Ma et al., 2018). Briefly, for each subsystem in DrugCell we constructed and compared two different L2-norm penalized linear regression models of drug response local to that subsystem. The first regression model predicts drug response using the neuron values that represent the state of the subsystem under the different genotypes. The second regression model predicts drug response using the neuron values that represent the states of the subsystem’s children. Both models are optimized to predict drug response, but with consecutive layers of neurons located at and below the subsystem of interest in DrugCell. Performance is calculated as the Spearman correlation (rho) between the actual and predicted drug responses for each of the two alternative linear regression models (AUC). TheRLIPPscore is then definedas the ratio ofSpearman rho of the first linear model to that of the second linear model. RLIPP > 1 reflects that the state of the parent subsystem has more predictive power for drug response than the mere concatenation of the states of its children, indicating the importance of the parent subsystem in learning.

Kuenzi et al. 2020
