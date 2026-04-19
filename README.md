# Customer Segmentation and Multiclass Classification Project
  
**Author:** Sanket Madaan  

---

## 1. Project Overview

This project investigates customer segmentation and purchasing behavior using both predictive modeling and unsupervised learning. The work is structured into two main parts: multiclass classification for customer segments and data visualization plus clustering-driven segmentation.

The dataset includes demographic attributes, purchasing behavior, and campaign responses. The main goals are:
- To build and compare multiclass classification models for existing customer segments.
- To explore the data visually and construct alternative clusters.
- To compare clustering algorithms on both real and synthetic datasets and derive business insights.

---

## 2. Dataset and Preprocessing

### 2.1 Classification Dataset

For the classification task, the dataset contains three numerical features (Age, Work Experience, Family Size) and six categorical features (Gender, EverMarried, Graduated, Profession, SpendingScore, Var1). Numerical features are imputed with the median and standardized, while categorical features are imputed with the most frequent value and one‑hot encoded.

The target is a four-class customer segment label: A, B, C, and D. Segment A and D are the largest groups, while Segment C is the smallest, leading to moderate class imbalance.

### 2.2 Visualization and Clustering Dataset

For the visualization and clustering component, preprocessing is carried out in Power BI’s Power Query. Key steps include:
- Dropping rows with missing income values after confirming their fraction is small and randomly distributed.
- Engineering an Age feature from Year of Birth and a TotalSpending feature by summing spending across product categories.
- Removing outliers, such as unrealistically high income (>150,000) and biologically implausible ages, as well as invalid marital status values.

After cleaning, the dataset contains 2,198 customers with complete and validated information.

---

## 3. Multiclass Classification

### 3.1 Class Imbalance Considerations

The class distribution shows moderate imbalance: Segment A and D are larger, while Segment C has only about 55% of the samples of Segment A. Imbalance can bias models toward majority classes and especially harms minority class recall.

To mitigate this, models use:
- Class weighting (class_weight = "balanced") in SVM and Random Forest.
- Stratified train–validation splits to maintain class proportions.

Evaluation uses per-class precision, recall, and F1 score, along with macro and weighted averages, since accuracy alone can be misleading under imbalance.

### 3.2 One-vs-All SVM

The One-vs-All (OVA) strategy trains one binary SVM per class against all others (A vs rest, B vs rest, etc.). Using stratified 80–20 split, OVA SVM reaches around 30.71% validation accuracy, which is above random and majority-class baselines.

Key observations:
- Segment A has the weakest recall (~0.18), despite being the largest class, suggesting strong confusion with other segments.
- Segment C shows relatively high recall (~0.42) due to class weighting but lower precision, indicating many false positives.
- Macro and weighted F1 scores are close (~0.31 vs ~0.30), showing no extreme favoritism toward any one class.

### 3.3 One-vs-One SVM

The One-vs-One (OVO) strategy trains a separate SVM for each pair of classes, resulting in six binary classifiers for four classes. Prediction uses voting across pairwise classifiers.

Performance:
- Validation accuracy improves to about 32.88%, a relative gain of roughly 7% over OVA.
- Segment C’s recall jumps to about 0.55, demonstrating better minority-class capture under pairwise training.
- Segment D also sees improved precision and F1.

The confusion matrix shows stronger diagonal entries, especially for C and D, reflecting more refined decision boundaries.

### 3.4 Random Forest Classification

A Random Forest classifier is built using decision trees with bootstrap sampling, random feature subsets, and balanced class weights. The key hyperparameter explored is the number of trees.

- Validation accuracy peaks at around 31.79% at 20 trees; performance plateaus or slightly degrades with more trees.
- Segment A attains its best recall under Random Forest (~0.40), outperforming both SVM strategies for this class.
- Segment C performs worse than under OVO in terms of recall, showing that Random Forest is less effective for the smallest class in this setting.

Feature importance analysis indicates Age as the dominant predictor, followed by Work Experience and Family Size, with several categorical features (e.g., Var1 categories, gender, profession) contributing meaningfully.

### 3.5 Overall Classification Comparison

Across OVA-SVM, OVO-SVM, and Random Forest:
- OVO-SVM achieves the highest accuracy and the most balanced macro and weighted F1 scores.
- Random Forest offers interpretability and strong performance for the majority class but weaker minority-class recall compared to OVO.
- OVA-SVM is simplest but suffers more from imbalance and overlapping class distributions.

---

## 4. Exploratory Data Analysis in Power BI

Power BI is used for initial exploration and business-facing visualization.

### 4.1 Demographic Insights

- Age distribution is centered around about 50 years, with most customers between 40 and 80. Younger customers (<40) are underrepresented, indicating a potential growth opportunity.
- Married and cohabiting customers dominate, reflecting a base of settled households.
- Educational attainment is high, with a majority having undergraduate degrees or above, suggesting a relatively affluent and sophisticated customer base.

### 4.2 Economic and Behavioral Patterns

- Middle-income customers (30,000–70,000) form the largest group, aligning with a mid-market positioning.
- Income is strongly positively correlated with total spending, but with substantial variance, indicating that lifestyle and preferences modulate spending.
- There is a negative correlation between income and web visits per month, implying that lower-income customers browse more frequently and are more price-conscious.
- Purchases across store, web, and catalog channels are positively correlated, showing that omnichannel customers tend to spend more overall.
- Wine purchases are highly correlated with total spending, acting as an anchor category for high-value customers.
- Having children at home is negatively correlated with total spending, suggesting budget constraints in family segments.

---

## 5. K-Means Clustering on Real Customers

### 5.1 Methodology

K-means clustering is applied to the cleaned real dataset with all 24 standardized features (demographic, spending, channels, campaign responses). Several cluster counts are evaluated: k = 2, 5, 7, and 9, using K-means++ initialization.

Cluster quality is assessed using:
- Inertia (within-cluster sum of squares).
- Silhouette score (cohesion and separation).

### 5.2 Choosing the Number of Clusters

- Inertia decreases monotonically as k increases, with diminishing returns beyond k = 7 (elbow at k = 7).
- Silhouette score is highest at k = 2 (≈0.285) and drops significantly for larger k values.

Given that inertia alone can be misleading, the silhouette criterion is prioritized, and k = 2 is selected as the optimal number of clusters.

### 5.3 Cluster Profiles

The k = 2 solution splits customers into approximately 40% and 60% groups.

**Cluster 0 – Premium High-Value Customers (≈40%):**  
- Older (around 58 years), higher income (~71k), and much higher total spending (~1,242).  
- Strong preference for premium categories (wines, meat), with spending 6–10 times that of the other cluster in several categories.  
- Heavy omnichannel usage: higher counts of web, store, and catalog purchases, especially catalog.  

**Cluster 1 – Value-Conscious Standard Customers (≈60%):**  
- Lower income (~38k) and low annual total spending (~185).  
- Spending focuses on necessities rather than premium categories.  
- Lower catalog usage, with most activity through web and store channels.  

From a business perspective, Cluster 0 is the core revenue driver deserving premium treatment (VIP programs, personalized offers), while Cluster 1 requires value-focused strategies and efficient mass-marketing.

---

## 6. Comparative Clustering on Synthetic Datasets

To understand clustering algorithm behavior under controlled conditions, four synthetic datasets are constructed: compact spherical clusters, skewed elongated clusters, hierarchical subclusters, and well-separated clusters. Five algorithms are evaluated: K-means, hierarchical clustering with single/complete/average linkage, and DBSCAN.

### 6.1 Evaluation Metrics and Parameter Tuning

- The elbow method consistently suggests k = 4 for k-based algorithms, matching the true number of clusters.
- DBSCAN parameters (eps and min_samples) are tuned via grid search to maximize silhouette score for each dataset.

### 6.2 Quantitative Results

Silhouette scores and inertia show that:
- DBSCAN often achieves the highest silhouette score on compact, skewed, and well-separated datasets, leveraging its density-based nature and independence from a fixed k.
- K-means and average linkage perform best on the subclusters dataset, where spherical assumptions are acceptable and densities are relatively homogeneous.
- On skewed data, K-means achieves low inertia but poorer silhouette than DBSCAN, highlighting the difference between compactness and real separation.

### 6.3 Algorithm Insights

- **K-Means**: Works well for spherical, similarly sized, and well-separated clusters; struggles with elongated shapes and varying densities.
- **Hierarchical clustering (average linkage)**: Offers robust performance across diverse datasets with interpretable dendrograms, while single and complete linkage show more extreme behaviors (chaining vs overly compact clusters).
- **DBSCAN**: Most versatile for arbitrary shapes, varying sizes, and outlier detection, and can discover the number of clusters automatically, but is sensitive to parameter choices and struggles with strongly varying densities.

---

## 7. Reflections on Power BI vs Programmatic Analysis

Power BI is highly effective for:
- Data cleaning via Power Query.
- Fast exploratory analysis and interactive dashboards.
- Communicating insights to non-technical stakeholders.

However, advanced analytics (K-means, silhouette score, PCA, grid search) require programmatic tools such as Python with scikit-learn and pandas. The ideal workflow combines both: Power BI for the “first mile” and “last mile,” and code-based analysis for the core modeling and validation.

---

## 8. Conclusions and Recommendations

From a modeling perspective:
- OVO-SVM provides the best overall trade-off among multiclass classifiers on this dataset, particularly for minority segment performance.
- Random Forest is competitive and offers clear feature importance, with Age emerging as the most influential predictor.
- Moderate overall accuracies (around 31–33%) reflect inherent difficulty due to overlap, imbalance, and limited sample size.

From a business perspective:
- The K-means clustering solution with two clusters identifies a premium high-value segment and a value-conscious segment, each requiring distinct marketing, product, and loyalty strategies.
- Key strategic levers include tiered loyalty programs, targeted campaigns by segment, optimized product portfolios (premium vs budget), and tailored channel strategies (omnichannel vs web/store emphasis).

This project demonstrates how combining supervised classification, unsupervised clustering, and visualization can deliver both technical understanding and actionable customer segmentation.
