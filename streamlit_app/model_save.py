import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

print("Bank Marketing Projesi - En İyi Model Kaydedici")
print("-----------------------------------------------")

# En iyi performansı gösteren model ve özellik setini belirleme
best_model_name = "Gradient Boosting"  # En yüksek F1 skoruna sahip model adı
best_feature_set = "RFE"  # En iyi özellik seti adı

print(f"Seçilen Model: {best_model_name}")
print(f"Seçilen Özellik Seti: {best_feature_set}")

# Veri setini yükleme
print("\nVeri yükleniyor...")
data_path = '../data/bank-full.csv'  
df = pd.read_csv(data_path, sep=';')

print(f"Veri seti boyutu: {df.shape}")

# Veri ön işleme
print("\nVeri ön işleme yapılıyor...")

# Kategorik değişkenleri dönüştürme
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
print(f"Kategorik değişkenler: {cat_cols}")

# Y değişkenini dönüştürme
le = LabelEncoder()
y = le.fit_transform(df['y'])
print(f"Hedef sınıflar: {le.classes_}")

# Özellik mühendisliği adımları
print("\nÖzellik mühendisliği yapılıyor...")

# Logaritmik dönüşümler
df['balance_log'] = np.log(df['balance'] + abs(min(0, df['balance'].min())) + 1)
df['duration_log'] = np.log(df['duration'] + 1)

# Binary değişkenleri dönüştürme
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    df[col+'_encoded'] = df[col].apply(lambda x: 1 if x == 'yes' else 0)

# One-hot encoding için 'unknown' değerini 'No Info' olarak değiştir
df.loc[df['poutcome'] == 'unknown', 'poutcome'] = 'No Info'
df.loc[df['contact'] == 'unknown', 'contact'] = 'No Info'

# One-hot encoding
df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'])

# DataFrame'deki gerçek sütun isimlerini kontrol et
print("\nOne-hot encoding sonrası sütun başlıkları:")
contact_cols = [col for col in df.columns if col.startswith('contact_')]
poutcome_cols = [col for col in df.columns if col.startswith('poutcome_')]
print(f"contact sütunları: {contact_cols}")
print(f"poutcome sütunları: {poutcome_cols}")

# Diğer özellikler
df['contacted_before'] = (df['previous'] > 0).astype(int)
df['pdays_long_ago'] = (df['pdays'] == -1).astype(int)
df['avg_duration_per_campaign'] = df['duration'] / df['campaign'].replace(0, 1)
df['campaign_success_rate'] = df['previous'] / df['campaign'].replace(0, 1)
df['previous_success_rate'] = df['campaign_success_rate'].copy()  # RFE'nin seçtiği değişken adı
df['campaign_success_rate'] = df['campaign_success_rate'].fillna(0)

df['day_of_week'] = df['day'] % 7
summer_months = ['month_jun', 'month_jul', 'month_aug']
df['is_summer'] = df[summer_months].any(axis=1).astype(int)

# RFE ile seçilen özellikleri manuel olarak tanımlama
print("\nRFE (Recursive Feature Elimination) ile özellik seçimi yapılıyor...")

# RFE tarafından seçilen özellikleri tanımla
selected_features_rfe = [
    'age',
    'day',
    'campaign',
    'balance_log',
    'duration_log',
    'pdays_long_ago',
    'avg_duration_per_campaign',
    'previous_success_rate',
    'housing_encoded',
    'loan_encoded'
]

# Özel durumlar için sütun isimlerini eşleştir
# poutcome_success
poutcome_success_col = [col for col in poutcome_cols if 'success' in col.lower()]
if poutcome_success_col:
    selected_features_rfe.append(poutcome_success_col[0])
else:
    print("Uyarı: 'poutcome_success' için uygun sütun bulunamadı.")

# poutcome_failure
poutcome_failure_col = [col for col in poutcome_cols if 'failure' in col.lower()]
if poutcome_failure_col:
    selected_features_rfe.append(poutcome_failure_col[0])
else:
    print("Uyarı: 'poutcome_failure' için uygun sütun bulunamadı.")

# contact_cellular
contact_cellular_col = [col for col in contact_cols if 'cellular' in col.lower()]
if contact_cellular_col:
    selected_features_rfe.append(contact_cellular_col[0])
else:
    print("Uyarı: 'contact_cellular' için uygun sütun bulunamadı.")

print(f"RFE için seçilen {len(selected_features_rfe)} özellik: {selected_features_rfe}")

# Seçilen özelliklere göre veri setini oluştur
X_set3 = df[selected_features_rfe]

# Veriyi eğitim ve test setlerine ayırma
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_set3, y, test_size=test_size, random_state=random_state, stratify=y
)
print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# Gradient Boosting modelini oluşturma ve eğitme
print("\nGradient Boosting modeli oluşturuluyor ve eğitiliyor...")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Model performansını değerlendirme
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Model performansı - Doğruluk: {accuracy:.4f}, F1 Skoru: {f1:.4f}")

# Modeli kaydetme
print("\nModel kaydediliyor...")
joblib.dump(gb_model, 'best_model.pkl')
print("Model başarıyla 'best_model.pkl' olarak kaydedildi.")

# Özellik isimlerini de kaydetme (tahmin için gerekecek)
feature_names = X_set3.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print("Özellik isimleri başarıyla 'feature_names.pkl' olarak kaydedildi.")

# Etiket kodlayıcıyı da kaydedelim (sınıf isimlerini almak için)
joblib.dump(le, 'label_encoder.pkl')
print("Etiket kodlayıcı başarıyla 'label_encoder.pkl' olarak kaydedildi.")

print("\nİşlem tamamlandı! Model ve gerekli dosyalar kaydedildi.")
print("Streamlit uygulamasını çalıştırmak için: streamlit run app.py") 