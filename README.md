# Driver Activity Recognition using Deep Belief Networks (DBN)

## Overview
This project focuses on **Driver Activity Recognition** using **Deep Belief Networks (DBN)**. It utilizes **Restricted Boltzmann Machines (RBM) and Logistic Regression** to classify driver activities based on extracted features from the **StateFarm Distracted Driver Dataset**. The primary goal is to identify and predict driver behaviors that could contribute to potential risks while driving.

## Algorithm Explanation
A **Deep Belief Network (DBN)** is a type of deep neural network composed of stacked **Restricted Boltzmann Machines (RBMs)**. RBMs are energy-based models used for unsupervised feature learning.

### **Working Mechanism:**
1. **Pretraining (Unsupervised Learning):**
   - The RBM learns hierarchical features from input data by adjusting weights based on patterns in the dataset.
   - The **BernoulliRBM** in this project uses 100 hidden units and a learning rate of 0.01 to extract meaningful representations.
   
2. **Fine-tuning (Supervised Learning):**
   - The output of the RBM is passed to a **Logistic Regression** classifier.
   - Logistic Regression maps the learned representations to specific driver activity classes.
   
## Dataset
- **Training Set:** Preprocessed driver activity features extracted from the dataset.
- **Test Set:** Unseen driver activity data for performance evaluation.

## Implementation Steps
1. **Load the dataset**
2. **Preprocess data:** Standardization and label encoding.
3. **Train the DBN:**
   - Train the RBM to learn feature representations.
   - Use Logistic Regression to classify driver activities.
4. **Evaluate performance:** Compute accuracy, precision, recall, and F1-score.

## Performance Metrics
Due to dataset limitations, the model's accuracy is relatively low:
- **Train Accuracy:** 30%
- **Test Accuracy:** 19%
- **Overall Accuracy:** 19%
- **Precision:** 20%
- **Recall:** 19%
- **F1-Score:** 19%

### **Why is Accuracy Low?**
1. **Small Dataset:** The model requires a larger dataset for better generalization.
2. **Limited Features:** More features could enhance classification performance.
3. **Hyperparameter Tuning:** Adjusting RBM parameters and optimizing the pipeline could improve results.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/driver-activity-recognition.git
   cd driver-activity-recognition
   ```
2. Install dependencies:
   ```bash
   pip install numpy scikit-learn
   ```
3. Run the script in **Google Colab** or locally:
   ```bash
   python driver_activity_recognition.py
   ```

## Future Improvements
- Expanding dataset size for better generalization.
- Hyperparameter tuning for **RBM and Logistic Regression**.
- Exploring additional deep learning architectures.
