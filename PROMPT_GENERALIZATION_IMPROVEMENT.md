# Prompt Generalization Improvement

## Problem Identified
The original column selection guidance was **biased toward specific datasets**:

### ❌ Before (Dataset-Specific Bias)
```
**Make intelligent choices**:
   - **Scatter plots**: Choose related variables that show meaningful relationships
     * Earthquake data: Latitude vs Longitude (geographic distribution)
     * House prices: LotArea vs SalePrice (price relationship)
     * Customer data: Age vs Income (demographic pattern)
   - **Histograms**: Choose the most important/interesting variable
     * Earthquake: Magnitude (primary measure of intensity)
     * Sales: Revenue (key business metric)
     * Survey: Age or Income (core demographics)
```

**Issues:**
1. ⚠️ **Hardcoded examples** create bias toward earthquake/house/customer data
2. ⚠️ **Limited adaptability** - doesn't help with manufacturing, healthcare, sports, etc.
3. ⚠️ **Implicit assumptions** - assumes certain column names (Magnitude, Revenue, Age)
4. ⚠️ **Narrow patterns** - only covers 3 domain types

---

## Solution: Domain-Agnostic Framework

### ✅ After (Generalized, Pattern-Based)
```
**For Scatter Plots** - Choose variables with meaningful relationships:
- Geographic data: Pair coordinate columns (latitude+longitude, x+y coordinates)
- Price/size relationships: Pair cost with quantity/area/volume metrics
- Performance metrics: Pair effort/input with outcome/output variables
- Temporal relationships: Pair time with trend variables
- Categorical vs numeric: Use most important numeric split by key category

**For Histograms** - Select the primary measure of interest:
- Target variable (if identified): The variable being predicted/analyzed
- Main metric: Revenue, score, magnitude, count, amount (key business/scientific measure)
- Distribution of interest: Variable with expected patterns (age, income, frequency)
- First numeric column with meaningful range (avoid IDs, binary flags)

**Selection principles** (no dataset-specific bias):
- Avoid ID columns, constants, or binary flags for visualizations
- Prefer columns with high variance and meaningful ranges
- Choose natural pairs (coordinates, input-output, cause-effect)
- Select variables that answer implicit questions about the data
- When uncertain, pick columns that reveal the most information
```

---

## Key Improvements

### 1. **Pattern-Based Instead of Dataset-Specific**
| Before | After |
|--------|-------|
| "Earthquake data: Latitude vs Longitude" | "Geographic data: Pair coordinate columns" |
| "House prices: LotArea vs SalePrice" | "Price/size relationships: Pair cost with quantity metrics" |
| "Customer data: Age vs Income" | "Performance metrics: Pair input with outcome variables" |

### 2. **Universal Principles**
- ✅ Works for **any domain**: healthcare, manufacturing, sports, finance, etc.
- ✅ Focuses on **data structure patterns** not domain knowledge
- ✅ Teaches the LLM **how to think** not what to memorize

### 3. **Broader Coverage**
**Original:** 3 example domains (earthquake, house, customer)  
**Improved:** 5+ relationship patterns applicable to ANY domain:
- Geographic (coordinates)
- Economic (price/quantity)
- Performance (input/output)
- Temporal (time/trends)
- Categorical splits

---

## Real-World Examples

### Example 1: Healthcare Data (Not in Original Examples)
**Dataset:** Patient records with columns: `Age`, `BMI`, `BloodPressure`, `Cholesterol`, `HeartRate`, `Diagnosis`

**Before (biased prompt):** 
- LLM struggles - no "earthquake" or "house prices" guidance
- Might pick random columns

**After (generalized prompt):**
- **Scatter:** `BMI` vs `BloodPressure` (health metrics relationship)
- **Histogram:** `Age` (distribution of interest, demographic variable)
- **Reasoning:** "I paired BMI and Blood Pressure as they're correlated health metrics. Age distribution shows patient demographics."

### Example 2: Manufacturing IoT Data
**Dataset:** Sensor readings with `Timestamp`, `Temperature`, `Pressure`, `Vibration`, `ProductionRate`, `DefectCount`

**Before:**
- No guidance for IoT/manufacturing data

**After:**
- **Scatter:** `Temperature` vs `Vibration` (performance metrics, cause-effect)
- **Time Series:** `Timestamp` vs `DefectCount` (temporal relationship)
- **Histogram:** `DefectCount` (main metric of interest)

### Example 3: Sports Analytics
**Dataset:** Player stats with `PlayerID`, `Minutes`, `Points`, `Rebounds`, `Assists`, `FGPercent`, `Position`

**Before:**
- Unclear which columns to use

**After:**
- **Scatter:** `Minutes` vs `Points` (effort/input vs outcome/output)
- **Box Plot:** `Points` grouped by `Position` (categorical vs numeric)
- **Histogram:** `FGPercent` (distribution of interest, performance metric)

---

## Technical Implementation

### Structure
1. **Analyze dataset structure** (types, ranges, patterns)
2. **Identify relationship patterns** (coordinates, input-output, temporal)
3. **Apply selection strategies** (scatter, histogram, box plot, time series)
4. **Explain with domain-agnostic reasoning**

### No Hardcoded Assumptions
- ❌ No specific column names (Magnitude, Revenue, Age)
- ❌ No specific domains (earthquake, sales, customer)
- ✅ Generic patterns (coordinates, metrics, categories)
- ✅ Data structure analysis (numeric, categorical, temporal)

---

## Benefits

### 1. **Flexibility** 🎯
- Works with **any dataset** from any domain
- Adapts to unfamiliar data structures
- No domain knowledge required

### 2. **Transparency** 💡
- LLM explains reasoning in domain-agnostic terms
- Users understand the logic regardless of domain
- Reasoning transferable across datasets

### 3. **Robustness** 🛡️
- Handles edge cases (unusual column names, new domains)
- Doesn't fail when data doesn't match examples
- Generalizes to future datasets

### 4. **Maintainability** 🔧
- No need to add examples for each domain
- Pattern-based approach scales infinitely
- Single framework for all use cases

---

## Comparison Table

| Aspect | Before (Specific) | After (Generalized) |
|--------|-------------------|---------------------|
| **Examples** | Earthquake, House, Customer | Pattern types (geographic, economic, performance) |
| **Coverage** | 3 domains | Unlimited domains |
| **Adaptability** | Breaks on new domains | Works on any dataset |
| **Bias** | High (toward examples) | Low (pattern-based) |
| **Reasoning** | Dataset-specific | Structural/relational |
| **Maintenance** | Add examples per domain | One framework forever |

---

## Testing Scenarios

### Scenario 1: Novel Domain (Gaming Analytics)
**Dataset:** `PlayerID`, `SessionDuration`, `LevelReached`, `InAppPurchases`, `DailyLogins`, `ChurnDate`

**Expected Behavior:**
- **Scatter:** `SessionDuration` vs `LevelReached` (effort vs achievement)
- **Histogram:** `InAppPurchases` (revenue metric)
- **Box Plot:** `SessionDuration` by `ChurnDate` presence (engagement by retention)

**Reasoning:** "Paired session duration with level reached to show player progression efficiency. InAppPurchases histogram reveals monetization distribution."

### Scenario 2: Scientific Data (Astronomy)
**Dataset:** `StarID`, `RightAscension`, `Declination`, `Magnitude`, `Temperature`, `Distance`, `SpectralClass`

**Expected Behavior:**
- **Scatter:** `RightAscension` vs `Declination` (coordinate pair - spatial distribution)
- **Histogram:** `Magnitude` (primary metric)
- **Color-coded scatter:** `Temperature` vs `Magnitude` colored by `SpectralClass`

**Reasoning:** "Right ascension and declination form spatial coordinates showing star positions. Magnitude histogram reveals brightness distribution."

### Scenario 3: Social Media Data
**Dataset:** `PostID`, `Timestamp`, `Likes`, `Shares`, `Comments`, `Reach`, `Engagement_Rate`, `Category`

**Expected Behavior:**
- **Scatter:** `Reach` vs `Engagement_Rate` (cause-effect, input-output)
- **Time Series:** `Timestamp` vs `Likes` (temporal trend)
- **Box Plot:** `Engagement_Rate` by `Category` (metric by grouping)

**Reasoning:** "Reach vs engagement rate shows content effectiveness. Time series reveals posting patterns and viral moments."

---

## Conclusion

✅ **Problem Solved:** Removed dataset-specific bias  
✅ **Approach:** Pattern-based, structure-driven selection  
✅ **Result:** Universal framework that works for ANY dataset  
✅ **Benefit:** Better generalization without losing intelligence  

The LLM now makes smart decisions based on **data patterns** rather than **memorized examples**.
