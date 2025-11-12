"""
Test cross-validation with imbalanced data to verify StratifiedKFold fix.
"""
import numpy as np
import pandas as pd
from src.tools.advanced_training import perform_cross_validation

print("=" * 80)
print("CROSS-VALIDATION STRATIFICATION TEST")
print("=" * 80)

# Create imbalanced dataset (90% class 0, 10% class 1)
np.random.seed(42)
n_samples = 100

# Features
X = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'feature3': np.random.randn(n_samples)
})

# Imbalanced target (90/10 split)
y = pd.Series([0] * 90 + [1] * 10, name='target')

# Combine into dataset
df = pd.concat([X, y], axis=1)

# Save to file
test_file = './test_data/imbalanced.csv'
df.to_csv(test_file, index=False)

print(f"\n📊 Dataset Info:")
print(f"   Total samples: {n_samples}")
print(f"   Class distribution:")
print(y.value_counts())
print(f"   Class ratio: {(y == 1).sum() / len(y) * 100:.1f}% minority class")

print(f"\n🔄 Testing cross-validation with default cv_strategy='kfold'...")
print(f"   (Should auto-switch to StratifiedKFold for classification)")

try:
    result = perform_cross_validation(
        file_path=test_file,
        target_col='target',
        model_type='random_forest',
        task_type='classification',
        cv_strategy='kfold',  # Default strategy
        n_splits=5,
        random_state=42
    )
    
    print(f"\n✅ SUCCESS! Cross-validation completed without errors")
    print(f"\nResults:")
    print(f"   OOF Accuracy: {result['oof_metrics']['accuracy']:.4f}")
    print(f"   Mean CV Accuracy: {result['cv_summary']['mean_accuracy']:.4f}")
    print(f"   Std CV Accuracy: {result['cv_summary']['std_accuracy']:.4f}")
    
    print(f"\n📋 Fold-by-fold scores:")
    for fold in result['fold_scores']:
        print(f"   Fold {fold['fold']}: Accuracy={fold['accuracy']:.4f}, F1={fold['f1']:.4f}, Samples={fold['samples']}")
    
    print(f"\n✅✅✅ TEST PASSED - Stratification working correctly!")
    
except ValueError as e:
    print(f"\n❌ FAILED with ValueError: {e}")
    print(f"   This means stratification is NOT working - some folds have single class")
    
except Exception as e:
    print(f"\n❌ FAILED with {type(e).__name__}: {e}")

print("\n" + "=" * 80)
