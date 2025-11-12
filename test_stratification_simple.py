"""
Test cross-validation stratification fix - Standalone test.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

print("=" * 80)
print("CROSS-VALIDATION STRATIFICATION TEST")
print("=" * 80)

# Create imbalanced dataset (90% class 0, 10% class 1)
np.random.seed(42)
n_samples = 100

# Features
X = np.random.randn(n_samples, 3)

# Imbalanced target (90/10 split)
y = np.array([0] * 90 + [1] * 10)

print(f"\n📊 Dataset Info:")
print(f"   Total samples: {n_samples}")
print(f"   Class 0: {(y == 0).sum()} samples ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"   Class 1: {(y == 1).sum()} samples ({(y == 1).sum() / len(y) * 100:.1f}%)")

# Test 1: Regular KFold (should fail with imbalanced data)
print(f"\n\n❌ Test 1: Regular KFold (BEFORE FIX)")
print(f"   This will likely fail because some folds might have only one class...")

try:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        print(f"   Fold {fold_idx + 1}: Train classes={np.unique(y_train_fold)}, Val classes={np.unique(y_val_fold)}")
        
        model.fit(X[train_idx], y_train_fold)
        y_pred = model.predict(X[val_idx])
        
    print(f"   ✅ KFold succeeded (got lucky with data distribution)")
    
except ValueError as e:
    print(f"   ❌ KFold FAILED: {str(e)[:100]}")
    print(f"   This is the bug you encountered!")


# Test 2: StratifiedKFold (should always work)
print(f"\n\n✅ Test 2: StratifiedKFold (AFTER FIX)")
print(f"   This guarantees each fold has both classes...")

try:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):  # Note: y is passed here
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        print(f"   Fold {fold_idx + 1}: Train classes={np.unique(y_train_fold)}, Val classes={np.unique(y_val_fold)}")
        
        model.fit(X[train_idx], y_train_fold)
        y_pred = model.predict(X[val_idx])
        
    print(f"\n   ✅ StratifiedKFold SUCCESS!")
    print(f"   All folds have both classes - metrics will work correctly")
    
except ValueError as e:
    print(f"   ❌ StratifiedKFold FAILED: {e}")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("The fix changes perform_cross_validation to:")
print("  - Auto-detect classification tasks")
print("  - Use StratifiedKFold for classification (not regular KFold)")
print("  - This ensures every fold has samples from all classes")
print("  - Prevents 'Invalid classes inferred' errors")
print("=" * 80)
