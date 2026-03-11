# 🐉Pokemon Pokedex MEGA Project

ALERT: To view the project, press: Pokemon_Pokedex_MEGA_Project_(main).ipynb

NOTE 1: Since this project is MASSIVE, the file of "Pokemon_Pokedex_MEGA_Project_(main).ipynb" can take a minute to load! If it fails to load, refresh the website and it should hopefully load!

NOTE 2: If the project does not load, here is a Kaggle Link: https://www.kaggle.com/code/brettean7/pokemon-pokedex-mega-project-preview-some-models

NOTE 3: In the end of this ReadMe, I will list my methodology and the Machine Learning Models I used.

---

## 1. ⭐ Why is this called the **MEGA Project**?

Before we begin, notice that this is called the **MEGA project**. The name has been inspired by MEGA evolution, a special mechanic introduced in Generation 6 of Pokemon. And it's great timing to use the word MEGA since a recent Pokemon game called *Pokemon Legends: Z-A* arrived in October 16, 2025. *Pokemon Legends: Z-A*'s theme is based on Generation 6 Pokemon and brings back the MEGA evolution mechanic.


The first reason this is a **MEGA Project** is because it has many parts: Data cleaning, data visualization, aggregate discoveries, exploratory data analysis,  classification models, regression models, and pattern models, and insights of Pokemon before and after discoveries in the data.

The second reason this is a **MEGA Project** is because I will aim to cater towards  everyone. If this is your first time seeing data science or if you are a veteran in data science, this project is for you. I will be explaining the simplest concepts to theories behind the models and data science concepts we use! There will also be many times where I explain concepts with real world examples such as airport data, NBA data, the population in the United States, financial scenarios, and more. Not everyone is a Pokemon fan, but there may be someone who is a NBA fan or a finance enthusiast.

The third reason this is a **MEGA Project** is because, well this is a looooooooooooooooonnnnnnnnnnnnnnnnnnngggggggggggggggg Project. 




## 2. 💵 The Business Problem: How do we identify minority populations and predict continous values?

**The Challenge:** In a dataset with over 1,000 observations and 30+ features, how do we identify "Scarcity of objects" (Legendary status) and predict "Strength oj objects" (Base Stats) when the data is heavily skewed and contains hidden archetypes? Lets do this in a fun way using Pokemon data!

**The Goal:** Move beyond intuition-based "guessing" to provide mathematical evidence for Pokemon classification, performance forecasting, and anomaly detection.

---



## 3. 💡 Key Insights and Final Conclusions

### **Scarcity Modeling (Classification):**
 **XGBoost** emerged as the superior classifier (**F1: 0.606**). I identified two "Noise Categories" that make classification difficult: **Mythical Pokemon** (which mimic Legendary traits) and **Pseudo-Legendaries** (High-stat regulars like Garchomp).

Bar plot of 10 classification models' F1 scores:
<img width="1447" height="707" alt="github_pokemon_1" src="https://github.com/user-attachments/assets/5af1a833-5551-45a9-bca5-8f2bffc4ba95" />

### **Strength Estimation (Regression):**

**SVR (Support Vector Regression)** provided the most stable predictions (**rMSE: 56.14**). While height and weight show a positive correlation with power, they are statistically "weak predictors," suggesting that "Power" in this universe is a latent variable not tied to physical mass.

3D Hyperparamter plot for SVR, SVR's performance can depend on its hyperparameters:
<img width="910" height="931" alt="github_pokemon_12_SVR" src="https://github.com/user-attachments/assets/8c0ec533-f8a6-4df3-a900-7c17c8fe2737" />

The values that give the SVR the lowest rMSE value is: $C = 100$ and $\gamma = 1$

As a refresher:
$C$ is the cost. The more higher the $C$, the more higher the penalty will be for any points outside tolerance zone. 
$\gamma$ tunes the complexity of the hyper plane. A more complex hyperplane is not always good, since it may perform well on the training set but predict testing set observations poorly due to overfitting.


Bar plot of 11 regression models' rMSE scores:
<img width="1424" height="705" alt="github_pokemon_2" src="https://github.com/user-attachments/assets/195cc965-c59d-4bcb-a592-e5c63ea426dd" />

  
### **Distributional Analysis:**

Discovered a **Bi-modal Distribution** in total base stats (peaks at 300 and 500). This suggests a "Dual-Tier" design philosophy rather than a standard Bell Curve.

Histogram of Pokemon' total base stats:
<img width="1247" height="702" alt="github_pokemon_3" src="https://github.com/user-attachments/assets/f5d8dcf5-c796-47b4-a9eb-5ab073e9b736" />

### **Independence of Predictor Variables:** 

Correlation analysis revealed a **near-zero relationship between Speed and Defense**. This debunked the "Heavy = Slow" intuition, proving that high-defense assets can maintain high velocity.

Correlation Matrix of Individual Pokemon base stats:
<img width="920" height="837" alt="github_pokemon_4_EDITED" src="https://github.com/user-attachments/assets/fbb185ff-ae47-4169-9b0d-69cc370aa87e" />


### **Dimensionality Reduction and Clustering:** 

PCA and K-Means confirmed that Legendary and high-stat Pokemon occupy a distinct "Cluster of Power" (Postive PC1 values).

2D PCA plot with Legendary labels:
<img width="1241" height="711" alt="github_pokemon_8" src="https://github.com/user-attachments/assets/0ffd03c6-7bb4-41eb-be12-07266fc70148" />

2D PCA plot with four K-Means cluster labels:
<img width="1241" height="710" alt="github_pokemon_9" src="https://github.com/user-attachments/assets/0360691b-2884-4844-98bf-2c6cd3d39f65" />

The averages the base stat values of each K-Means cluster:

<img width="708" height="190" alt="github_pokemon_10" src="https://github.com/user-attachments/assets/f22b3c32-0604-442d-a314-3a9aa8f4665a" />

Misc averages and proportions of each K-Means cluster:

<img width="591" height="185" alt="github_pokemon_11" src="https://github.com/user-attachments/assets/db7c8b3d-f813-4f9b-ac35-27e3493369a7" />

Notice that cluster 3 contained the strongest Pokemon on average since the average total base stat is 577.488! 
And recall that the proportion of Legendary Pokemon is biggest in cluster 3! 
Therefore, most Legendary Pokemon and strong Pokemon tend to be in postive PC1 areas.



### **The "Outlier" Protocol:**

While basic stats find "weak" Pokemon, I used **HAC (Hierarchical Agglormative Clustering)** to mathematically define outliers like **Meltan**, who sits in a unique statistical cluster separate from other Mythical entities. Furthermore, Isolation Forests showed that the outliers are any Pokemon with a extremely high or low stat like Attack or HP.

HAC dendrogram plot of Meltan being it's own cluster (outlier):
<img width="1499" height="817" alt="github_pokemon_5_EDITED" src="https://github.com/user-attachments/assets/11eec101-5f40-42c2-9742-63c5267e0a83" />
In this cut off line or any cut line, Meltan was bound to be isolated as it's own cluster.


The first 20 Pokemon Outliers based on Isolation Forest:
<img width="903" height="654" alt="github_pokemon_6" src="https://github.com/user-attachments/assets/a40b9fed-f522-4ed8-97a6-49fbca07185f" />

There are a total of 52 Pokemon that are outliers if we believe that 5% of the data has outliers or anomalies. 
Any Pokemon with a extremely high or low individual base stat like attack is considered as a outlier since our predictor variables uses only Pokemon individual base stats.  

3D Anomaly Plot, based on 3D PCA and Isolation Forest Anomaly Scores:  
<img width="960" height="963" alt="github_pokemon_7" src="https://github.com/user-attachments/assets/a0d03c2b-f6c5-4055-b589-ff8f517d6d00" />

Each point represents a Pokemon.
Notice a Pokemon is more likely to be a anomaly if it is farther away from the main group of Pokemon. 
Anomalies are denoted by a darker color. 






---

## 4. 📔 The Methodology of what I did: 

### Firstly, I did exploratory data analysis.

I tried to find any interesting details in the Pokedex data. For instance, when grouping Pokemon by their types, it reveals that Fighting Pokemon had the highest average attack stat. 

Plots of predictor against response variables or predictors against other predictors were made.

### Secondly, regression, classification, and unsupervised machine learning models were created.

I implemented a **Model-Competition Framework** across 31 architectures to identify the most robust models of explaining and prediciting Pokemon characteristics. 

* **Feature Engineering:** Evaluated physical attributes (height/weight) vs. statistical attributes, discovering that physical metrics are surprisingly weak predictors of Pokemon's total base stats. (Total base stats show how strong a Pokemon is.)
  
* **Handling Class Imbalance:** With only **6%** of the population being "Legendary," I prioritized models (like XGBoost) that handle sparse target variables without overfitting.

* **Stack:** Python (NumPy, pandas, matplotlib, seaborn, scikit-learn, XGBoost, TensorFlow/Keras, Plotly, YellowBrick, SciPy)

### Thirdly, I tried to gather insights about which machine learning model had the best metric and if we found any discoveries about Pokemon data as a whole. 

For a refresher, check them out at point 3! [point 3!](#-3--key-insights-and-final-conclusions)

---

## 5. 💻 Technical Log (30+ Models)

<details>
<summary><b> Click this Arrow to expand the Comprehensive Model Library! </b></summary>



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

</details>

---



