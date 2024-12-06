# Data Cleansing Process

## A. Handle Missing Values

1. **Gender**

   - **Action:** Missing values filled with `'Other'`.
   - **Result:** Unique values are now `'male'`, `'female'`, and `'other'`.

2. **Enrolled University**

   - **Action:** Filled with `'no_enrolement'`.

3. **Education Level**

   - **Action:** Dropped missing values (2% of data).

4. **Major Discipline**

   - **Action:** Filled with `'Unknown'`.

5. **Experience**

   - **Action:** Dropped missing values (0.3% of data).

6. **Company Size**

   - **Action:** Filled with `'0'` as placeholder.

7. **Company Type**

   - **Action:** Filled with `'Unknown'`.

8. **Last New Job**
   - **Action:** Dropped missing values (2.2% of data).

---

## B. Handle Duplicate Data

- **Result:** No duplicate data found in the dataset.

---

## C. Handle Outliers

1. **City Development Index**

   - Outliers retained as they may provide meaningful insights related to job search.

2. **Training Hours**
   - Extreme outliers (>200 hours) identified, potentially distorting data.
   - **Action:** Log transformation applied to normalize distribution.

---

## D. Feature Transformation

1. **City Development Index**

   - **Action:** No transformation required as data distribution is suitable for analysis.

2. **Training Hours**
   - **Action:** Applied logarithmic transformation for normalization.

---

## E. Feature Encoding

1. **Label Encoding**

   - Applied to:
     - **Ordinal categorical data:** Features with inherent order.
     - **Binary categorical data:** Features with only two unique categories.

2. **One-Hot Encoding**
   - Applied to non-ordinal categorical data to create dummy variables.

---

## F. Imbalanced Data

- **Observation:** 24.5% of employees are likely to seek new jobs.
- **Action:** Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the target labels, resulting in a ratio of **1:2** for `1` (seeking a new job) and `0` (not seeking a new job).
