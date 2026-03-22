
import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def preprocess_data(train_df, test_df):
    """Preprocesses training and testing dataframes."""
    X_train_full = train_df.drop(['ID', 'Segmentation'], axis=1)
    y_train_full = train_df['Segmentation']
    X_test_final = test_df.drop('ID', axis=1)

    # Define preprocessing pipelines
    numerical_features = X_train_full.select_dtypes(include=np.number).columns.tolist()
    print(f"Numerical features: {numerical_features}")
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = X_train_full.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Categorical features: {categorical_features}")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # Create the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')

    # Fit on the full training data and transform all sets
    preprocessor.fit(X_train_full)
    X_train_full_processed = preprocessor.transform(X_train_full)
    X_test_processed = preprocessor.transform(X_test_final)

    return X_train_full_processed, y_train_full, X_test_processed, preprocessor


class OneVsOneClassifier:
    def __init__(self, base_estimator, **kwargs):
        self.base_estimator = base_estimator
        self.estimator_kwargs = kwargs
        self.estimators_ = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_, y_mapped = np.unique(y, return_inverse=True)
        class_pairs = list(combinations(range(len(self.classes_)), 2))

        for class1_idx, class2_idx in class_pairs:
            indices = np.where((y_mapped == class1_idx) | (y_mapped == class2_idx))
            X_pair, y_pair = X[indices], y_mapped[indices]
            
            estimator = self.base_estimator(**self.estimator_kwargs)
            estimator.fit(X_pair, y_pair)
            self.estimators_[(class1_idx, class2_idx)] = estimator
        return self

    def predict(self, X):
        predictions = np.array([est.predict(X) for est in self.estimators_.values()]).T
        final_preds_indices = [np.bincount(pred.astype(int)).argmax() for pred in predictions]
        return self.classes_[final_preds_indices]


def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def print_metrics(y_true, y_pred, classes, strategy_name):
    """Print detailed classification metrics."""
    print(f"\n{'='*60}")
    print(f"Classification Report - {strategy_name}")
    print(f"{'='*60}")
    report = classification_report(y_true, y_pred, labels=classes, target_names=classes, digits=4)
    print(report)
    
    with open(f'{strategy_name.lower().replace(" ", "_")}_metrics.txt', 'w') as f:
        f.write(f"Classification Report - {strategy_name}\n")
        f.write("="*60 + "\n")
        f.write(report)
    print(f"Metrics saved to {strategy_name.lower().replace(' ', '_')}_metrics.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <test_file_path>")
        sys.exit(1)
    
    test_file = sys.argv[1]
    train_file = 'Customer_train.csv'

    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure train.csv and test file exist.")
        sys.exit(1)

    # Preprocess all data first
    X_full, y_full, X_test_final_processed, preprocessor = preprocess_data(train_df, test_df)

    # --- Validation Step ---
    print("\n" + "="*60)
    print("ONE-VS-ONE (OVO) STRATEGY - VALIDATION")
    print("="*60)
    
    # Split the FULL processed training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    # Check class distribution
    print("\nClass Distribution in Training Set:")
    print(pd.Series(y_train_split).value_counts().sort_index())
    print("\nClass Distribution in Validation Set:")
    print(pd.Series(y_val_split).value_counts().sort_index())

    val_classifier = OneVsOneClassifier(SVC, class_weight='balanced')
    val_classifier.fit(X_train_split, y_train_split)
    val_predictions = val_classifier.predict(X_val_split)
    
    accuracy = accuracy_score(y_val_split, val_predictions)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    
    # Get unique classes
    classes = np.unique(y_full)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val_split, val_predictions, classes,
                         'Confusion Matrix - One-vs-One (OVO)',
                         'confusion_matrix_ovo.png')
    
    # Print detailed metrics
    print_metrics(y_val_split, val_predictions, classes, 'One-vs-One (OVO)')

    # --- Final Training and Prediction Step ---
    print("\n" + "="*60)
    print("TRAINING ON FULL DATA AND PREDICTING FOR SUBMISSION")
    print("="*60)
    
    # Train a new classifier on the ENTIRE training dataset
    final_classifier = OneVsOneClassifier(SVC, class_weight='balanced')
    final_classifier.fit(X_full, y_full)

    # Predict on the official test set
    final_predictions = final_classifier.predict(X_test_final_processed)

    pd.DataFrame({'predicted': final_predictions}).to_csv('ovo.csv', index=False)
    print("\nFinal predictions saved to ovo.csv")
    print("="*60)
