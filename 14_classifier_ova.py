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


def preprocess_data(train_df, test_df):
    """Preprocesses training and testing dataframes."""
    X_train_full = train_df.drop(['ID', 'Segmentation'], axis=1)
    y_train_full = train_df['Segmentation']
    X_test_final = test_df.drop('ID', axis=1)

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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')

    preprocessor.fit(X_train_full)
    X_train_full_processed = preprocessor.transform(X_train_full)
    X_test_processed = preprocessor.transform(X_test_final)

    return X_train_full_processed, y_train_full, X_test_processed, preprocessor


class OneVsAllClassifier:
    def __init__(self, base_estimator, **kwargs):
        self.base_estimator = base_estimator
        self.estimator_kwargs = kwargs
        self.estimators_ = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, 0)
            estimator = self.base_estimator(**self.estimator_kwargs)
            estimator.fit(X, y_binary)
            self.estimators_[cls] = estimator
        return self

    def predict(self, X):
        decision_scores = np.array([est.decision_function(X) for est in self.estimators_.values()]).T
        return self.classes_[np.argmax(decision_scores, axis=1)]


def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
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

    X_full, y_full, X_test_final_processed, preprocessor = preprocess_data(train_df, test_df)

    
    print("\n" + "="*60)
    print("ONE-VS-ALL (OVA) STRATEGY - VALIDATION")
    print("="*60)
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)


    print("\nClass Distribution in Training Set:")
    print(pd.Series(y_train_split).value_counts().sort_index())
    print("\nClass Distribution in Validation Set:")
    print(pd.Series(y_val_split).value_counts().sort_index())

    val_classifier = OneVsAllClassifier(SVC, class_weight='balanced')
    val_classifier.fit(X_train_split, y_train_split)
    val_predictions = val_classifier.predict(X_val_split)
    
    accuracy = accuracy_score(y_val_split, val_predictions)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    
    # Get unique classes
    classes = np.unique(y_full)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val_split, val_predictions, classes,
                         'Confusion Matrix - One-vs-All (OVA)',
                         'confusion_matrix_ova.png')
    
    # Print detailed metrics
    print_metrics(y_val_split, val_predictions, classes, 'One-vs-All (OVA)')

    # --- Final Training and Prediction Step ---
    print("\n" + "="*60)
    print("TRAINING ON FULL DATA AND PREDICTING FOR SUBMISSION")
    print("="*60)
    
    final_classifier = OneVsAllClassifier(SVC, class_weight='balanced')
    final_classifier.fit(X_full, y_full)

    final_predictions = final_classifier.predict(X_test_final_processed)

    pd.DataFrame({'predicted': final_predictions}).to_csv('ova.csv', index=False)
    print("\nFinal predictions saved to ova.csv")
    print("="*60)
