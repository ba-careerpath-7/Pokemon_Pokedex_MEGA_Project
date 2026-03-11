# Pokemon_Pokedex_MEGA_Project




### Pokemon Pokedex MEGA Project from Resume

READ ME: To view the project, press: Pokemon_Pokedex_MEGA_Project_(main).ipynb

NOTE: Since this project is MASSIVE, the file of "Pokemon_Pokedex_MEGA_Project_(main).ipynb" can take a minute to load! If it fails to load, refresh the website and it should hopefully load!

NOTE: If the project does not load, here is a Kaggle Link: https://www.kaggle.com/code/brettean7/pokemon-pokedex-mega-project-preview-some-models

## Overview:
This MEGA Project explores Pokemon data using Python, data science, and machine learning. The goal is to clean the Pokedex data, explore the data, create visualizations, build machine learning models, and get insights from both exploration and models.

Key Goals:

* Classify whether a future Pokemon is Legendary
* Predict total base stats for future Pokemon
* Discover patterns in Pokemon using distinct base stats.

Coding Language Used: Python

Python Packages Used:
NumPy, pandas, matplotlib, seaborn, scikit-learn, XGBoost, TensorFlow/Keras, Plotly, YellowBrick, SciPy 

Repository Contents:

* Contains data cleaning, filtering, aggregates, EDA, and machine learning experiments
* Plots and visualizations
* Insights, observations, and discoveries


## Models implemented:

Each Model contains:
- A explanation of how the model works.
- The code of the model.
- Insights or results. (Some models such as KNN or PCA contain visuals.)

### Classification 📂:
1. KNN
2. Naive Bayes
3. Logistic Regression
4. Logistic Ridge
5. Logistic Lasso
6. SVM 
7. Classification Tree
8. Classification Random Forests
9. Classification XGBoost
10. Classification Neural Network 



### Regression 📈:
1. Linear Regression
2. Ridge
3. LASSO
4. PCR 
5. PLS 
6. SVR 
7. Regression Tree
8. Regression Random Forests
9. Regression XGBoost
10. Regression Neural Network 
11. KNN Regressor



### Un-Supervised 🧩:

1. PCA
2. K Means
3. Spectral Clustering
4. t-SNE
5. MDS 
6. LLE 
7. DBSCAN
8. GMM 
9. HAC 
10. Isolation Forests


---


## 🐲 Project: Multivariate Pokedex Analytics & Scarcity Modeling

### 1. The Business Problem (The "Hook")

**The Challenge:** In a dataset with over 1,000 observations and 30+ features, how do we identify "Scarcity" (Legendary status) and predict "Performance" (Base Stats) when the data is heavily skewed and contains hidden archetypes?
**The Goal:** Move beyond intuition-based "guessing" to provide mathematical evidence for Pokemon classification, performance forecasting, and anomaly detection.

---

### 2. The Methodology (The "How")

I implemented a **Model-Competition Framework** across 32 architectures to identify the most robust predictors of Pokemon attributes.

* **Feature Engineering:** Evaluated physical attributes (height/weight) vs. statistical attributes, discovering that physical metrics are surprisingly weak predictors of power.
* **Handling Class Imbalance:** With only **6%** of the population being "Legendary," I prioritized models (like XGBoost) that handle sparse target variables without overfitting.
* **Stack:** `Python`, `XGBoost`, `Scikit-Learn`, `TensorFlow`, `YellowBrick` (Visual Diagnostics).

---

### 3. Key Insights (The "So What?")

* **Scarcity Modeling (Classification):** **XGBoost** emerged as the superior classifier (**F1: 0.606**). I identified two "Noise Categories" that make classification difficult: **Mythical Pokemon** (which mimic Legendary traits) and **Pseudo-Legendaries** (High-stat regulars like Garchomp).
* **Performance Forecasting (Regression):** **SVR (Support Vector Regression)** provided the most stable predictions (**rMSE: 56.14**). While height and weight show a positive correlation with power, they are statistically "weak predictors," suggesting that "Power" in this universe is a latent variable not tied to physical mass.
* **Distributional Analysis:** Discovered a **Bi-modal Distribution** in base stats (peaks at 300 and 500). This suggests a "Dual-Tier" design philosophy rather than a standard Bell Curve.
* **Independence of Variables:** Correlation analysis revealed a **near-zero relationship between Speed and Defense**. This debunked the "Heavy = Slow" intuition, proving that high-defense assets can maintain high velocity.
* **The "Outlier" Protocol:** While basic stats find "weak" Pokemon, I used **Isolation Forests** and **HAC (Hierarchical Clustering)** to mathematically define outliers like **Meltan**, who sits in a unique statistical cluster separate from other Mythical entities.

---

### 4. Visual Evidence & Patterns

* **PCA & t-SNE:** Dimensionality reduction confirmed that Legendary and high-stat Pokemon occupy a distinct "Cluster of Power" (Positive PC1).
* **Feature Importance:** Data-driven evidence confirmed **Electric types** as the velocity leaders and **Fairy types** as the Special Defense leaders—validating fan intuition with statistical significance.

---

### 5. Technical Log (30+ Models)

<details>
<summary><b>Click to expand the Comprehensive Model Library</b></summary>

* **Classification:** XGBoost (Winner), SVM, Neural Networks, Naive Bayes, Logistic (Lasso/Ridge).
* **Regression:** SVR (Winner), Random Forest, PCR, PLS, Ridge/Lasso.
* **Unsupervised:** PCA, t-SNE, Isolation Forests, GMM, Spectral Clustering, HAC.

</details>

---



