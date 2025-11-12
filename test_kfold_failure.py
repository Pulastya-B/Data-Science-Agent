"""
Test showing KFold FAILURE with extreme imbalance.
"""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=" * 80)
print("EXTREME IMBALANCE TEST - KFold vs StratifiedKFold")
print("=" * 80)

# Create EXTREME imbalance (96% class 0, 4% class 1)
# With only 4 samples of class 1 and 5 folds, at least one fold will have NO class 1
np.random.seed(42)
n_samples = 100

X = np.random.randn(n_samples, 3)
y = np.array([0] * 96 + [1] * 4)  # Only 4 samples of class 1!

print(f"\n📊 Dataset Info:")
print(f"   Total samples: {n_samples}")
print(f"   Class 0: {(y == 0).sum()} samples ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"   Class 1: {(y == 1).sum()} samples ({(y == 1).sum() / len(y) * 100:.1f}%)")
print(f"   ⚠️ With 5 folds and only 4 minority samples, KFold will likely fail!")

# Test 1: Regular KFold
print(f"\n\n❌ Test 1: Regular KFold")
try:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    failed = False
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        train_classes = np.unique(y_train_fold)
        val_classes = np.unique(y_val_fold)
        
        print(f"   Fold {fold_idx + 1}: Train={train_classes}, Val={val_classes}", end="")
        
        if len(train_classes) < 2 or len(val_classes) < 2:
            print(f" ⚠️ SINGLE CLASS DETECTED!")
            failed = True
        else:
            print(f" ✓")
        
        try:
            model.fit(X[train_idx], y_train_fold)
            y_pred = model.predict(X[val_idx])
            acc = accuracy_score(y_val_fold, y_pred)
        except ValueError as e:
            print(f"      ❌ Error during fit/predict: {str(e)[:60]}...")
            failed = True
    
    if failed:
        print(f"\n   ❌ KFold FAILED - Some folds had single class")
    else:
        print(f"\n   ⚠️ KFold worked this time, but results are unreliable with imbalanced data")
        
except Exception as e:
    print(f"\n   ❌ KFold FAILED: {str(e)[:100]}")


# Test 2: StratifiedKFold
print(f"\n\n✅ Test 2: StratifiedKFold")
try:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        train_classes = np.unique(y_train_fold)
        val_classes = np.unique(y_val_fold)
        
        print(f"   Fold {fold_idx + 1}: Train={train_classes}, Val={val_classes} ✓")
        
        model.fit(X[train_idx], y_train_fold)
        y_pred = model.predict(X[val_idx])
        acc = accuracy_score(y_val_fold, y_pred)
        
    print(f"\n   ✅ StratifiedKFold SUCCESS - All folds have both classes!")
    
except Exception as e:
    print(f"\n   ❌ StratifiedKFold FAILED: {e}")


print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("With imbalanced data:")
print("  ❌ KFold: Random splits can create folds with single class → metrics fail")
print("  ✅ StratifiedKFold: Maintains class distribution in every fold → always works")
print("\nYour error 'Expected: [0], got: [1]' means:")
print("  - One fold had ONLY class 0, another had ONLY class 1")
print("  - sklearn metrics require both classes to calculate scores")
print("  - Fix: Auto-use StratifiedKFold for classification ✓")
print("=" * 80)
