# Deep-learning
### H&M Fashion Recommender System  

#### **Overview**  
This project focuses on developing a personalized fashion recommendation system for H&M, utilizing customer purchase history, product metadata, and image embeddings. The system follows a two-stage approach—retrieval and ranking—to efficiently identify and rank relevant fashion items for each customer. Key components of the project include data preprocessing, feature engineering, model training, and evaluation to ensure high-quality recommendations.  

---

### **Table of Contents**  
1. Introduction  
2. Dataset Description  
3. Methods  
   - Step 1: Data Loading and Initial Processing  
   - Step 1.2: Image Embedding  
   - Step 2: Data Splitting  
   - Step 3: Feature Engineering  
   - Step 4: Model Data Preparation  
   - Step 5: Model Architecture  
   - Step 6: Inference  
   - Step 7: Evaluation  
4. Results and Evaluation  

---

### **Introduction**  
The objective of this project is to build a recommendation system for H&M that offers personalized fashion suggestions based on customer purchase behavior, product characteristics, and visual features. The system employs a two-tower neural network to map customers and articles into a shared embedding space, enabling efficient retrieval and ranking of relevant items.  

Key aspects of the project include:  
- **Data preprocessing** to handle large datasets and ensure consistency.  
- **Feature engineering** to create meaningful representations of customers, products, and interactions.  
- **Model training** using a two-stage retrieval and ranking approach.  
- **Performance evaluation** using metrics like Precision@K, Recall@K, and NDCG@K.  

---

### **Dataset Description**  
The dataset comprises three primary components:  

1. **Articles (Products)**  
   - Contains metadata for each fashion item, including category, color, and graphical attributes.  
   - **Size**: 105,542 articles.  
   - **Key Features**: `article_id`, `product_code`, `product_type`, `color_group`, `graphical_appearance`, etc.  

2. **Customers**  
   - Includes demographic and engagement data for 1.37 million customers.  
   - **Key Features**: `customer_id`, `age`, `postal_code`, `club_member_status`, `fashion_news_frequency`, etc.  

3. **Transactions**  
   - Records customer purchases with timestamps and sales channel details.  
   - **Size**: 28 million transactions.  
   - **Key Features**: `customer_id`, `article_id`, `t_dat` (purchase date), `price`, `sales_channel_id`.  

4. **Image Embeddings**  
   - Product images are represented as 2048-dimensional vectors using **ResNet152**.  
   - These embeddings capture visual similarities between fashion items.  

---

### **Methods**  

#### **Step 1: Data Loading and Initial Processing**  
- Load and preprocess customer, article, and transaction data.  
- Handle large-scale data using batch processing and parallel techniques.  
- Standardize data types (e.g., convert IDs to strings, transform dates).  
- Reduce the dimensionality of image embeddings using **PCA**, preserving 95% variance.  

#### **Step 1.2: Image Embedding**  
- Process product images using **ResNet152**, resizing them to 224x224 pixels.  
- Generate 2048-dimensional embeddings and store them as `.npy` files.  
- Apply **PCA** to reduce dimensionality while retaining key visual features.  

#### **Step 2: Data Splitting**  
- Split data chronologically to maintain the temporal nature of transactions.  
- **Train (70%)** → **Validation (15%)** → **Test (15%)** to ensure real-world applicability.  

#### **Step 3: Feature Engineering**  
- Develop meaningful features for customers, articles, and interactions.  

  **Customer Features**:  
  - Purchase patterns (total transactions, average purchase price, frequency).  
  - Recent activity (purchases in the last 30 days).  

  **Article Features**:  
  - Popularity metrics (total sales, unique customers).  
  - Recent performance (sales in the last 30 days).  

  **Interaction Features**:  
  - Temporal aspects (day of the week, month, weekend indicator).  

#### **Step 4: Model Data Preparation**  
- Generate **negative samples**: For every actual purchase, 4 non-purchased items are randomly selected.  
- Standardize numerical features and encode categorical variables.  
- Convert customer and article IDs to integer indices for model input.  

#### **Step 5: Model Architecture**  

1. **Retrieval Model**  
   - Two-tower neural network:  
     - **Customer Tower**: Maps customer features into a **32-dimensional embedding**.  
     - **Article Tower**: Maps article features into a **32-dimensional embedding**.  
   - Uses **ReLU activation** and **layer normalization** for better generalization.  

2. **Ranking Model**  
   - Merges customer and article embeddings to predict purchase likelihood.  
   - Uses **dense layers with ReLU activation** and a **sigmoid output layer**.  

**Training Configuration**:  
- **Optimizer**: Adam (learning rate = 0.001).  
- **Loss Function**: Combined retrieval and ranking loss.  
- **Evaluation Metrics**: AUC, Precision, Recall.  

#### **Step 6: Inference**  
- Retrieve candidate articles using the retrieval model.  
- Rank the candidates using the ranking model.  
- Exclude previously purchased items to maintain recommendation novelty.  
- Output the **top K** recommendations for each customer.  

#### **Step 7: Evaluation**  
- Assess the system’s performance using multiple metrics:  
  - **Precision@K** – Measures recommendation accuracy.  
  - **Recall@K** – Assesses how well recommendations cover user preferences.  
  - **NDCG@K** – Evaluates ranking quality.  

---

### **Results and Evaluation**  

#### **Training Performance**  
- The model demonstrates **consistent improvement over 500 epochs**.  
- **Precision**: Ranges between **0.85 and 0.925**.  
- **AUC**: Stays consistently above **0.85**, indicating strong performance.  

#### **Inference Performance**  
- Successfully generates personalized recommendations.  
- Effectively filters out previously purchased items to ensure novelty.  

---

### **Conclusion**  
This project successfully develops a personalized **fashion recommendation system for H&M**, leveraging customer purchase history, product metadata, and image embeddings. The **two-tower neural network architecture** ensures efficient retrieval and ranking of relevant fashion items. Evaluation metrics validate the model’s **effectiveness in providing personalized and accurate recommendations**, making it a valuable tool for improving customer shopping experiences.
