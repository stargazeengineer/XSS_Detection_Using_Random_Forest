import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 讀取資料集
df = pd.read_csv("XSS_Analyize_Using_ML/XSS_dataset.csv", encoding='utf-8-sig')

# 把 Label 拿出來
labels = df['Label']

# Label 編碼成數字（如果還沒是數字）
label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
y = labels.map(label_mapping)

# 建立反轉對照（數字 → 字串）
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# 移除 Label 欄位
df = df.drop(columns=['Label'])

# 合併所有欄位成文字
df['combined'] = df.astype(str).agg(' '.join, axis=1)

# 向量化文字
vectorizer = CountVectorizer(min_df=1, max_df=0.9)
X = vectorizer.fit_transform(df['combined']).toarray()

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 訓練 Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 預測
y_pred = rf.predict(X_test)

# 評估模型
print("----- Evaluation Report -----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(
    y_test, y_pred,
    target_names=[str(inv_label_mapping[i]) for i in sorted(inv_label_mapping.keys())]
))

# --- Label 分佈圖（類別分佈） ---
plt.figure(figsize=(8, 6))
sns.countplot(x=y.map(inv_label_mapping))
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# --- 句子長度分佈 ---
df['length'] = df['combined'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
sns.histplot(df['length'], bins=30, kde=True)
plt.title("Sentence Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# --- 混淆矩陣 ---
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=[inv_label_mapping[i] for i in sorted(inv_label_mapping.keys())],
            yticklabels=[inv_label_mapping[i] for i in sorted(inv_label_mapping.keys())])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --- 特徵重要性圖 ---
importances = rf.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20
features = np.array(vectorizer.get_feature_names_out())[indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), features)
plt.xlabel('Importance Score')
plt.title('Top 20 Important Features')
plt.show()

# --- Classification Report 表格 ---
report = classification_report(y_test, y_pred, target_names=[str(inv_label_mapping[i]) for i in sorted(inv_label_mapping.keys())], digits=5, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y.map(inv_label_mapping))
plt.title("PCA Visualization of Dataset")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()
