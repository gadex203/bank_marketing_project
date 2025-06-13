import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

# Streamlit uygulaması başlığı
st.title('Banka Pazarlama Kampanyası Tahmin Uygulaması')
st.write('Bu uygulama, bir müşterinin vadeli mevduat ürününe abone olup olmayacağını tahmin eder.')

# Model, özellik isimleri ve etiket kodlayıcıyı yükle
@st.cache_resource
def load_model_and_resources():
    try:
        model = joblib.load('best_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, feature_names, label_encoder
    except Exception as e:
        st.error(f"Model veya kaynak dosyaları yüklenirken hata oluştu: {str(e)}")
        return None, None, None

# Kullanıcı girişi için form oluştur
st.sidebar.header('Müşteri Bilgilerini Girin')

def user_input_features():
    age = st.sidebar.slider('Yaş', 18, 95, 40)
    job = st.sidebar.selectbox('Meslek', 
                             ('admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                              'management', 'retired', 'self-employed', 'services', 
                              'student', 'technician', 'unemployed'))
    marital = st.sidebar.selectbox('Medeni Durum', ('married', 'single', 'divorced'))
    education = st.sidebar.selectbox('Eğitim', ('primary', 'secondary', 'tertiary', 'Other'))
    default = st.sidebar.selectbox('Kredi Temerrüdü', ('no', 'yes'))
    balance = st.sidebar.slider('Bakiye', -8000, 100000, 1500)
    housing = st.sidebar.selectbox('Konut Kredisi', ('no', 'yes'))
    loan = st.sidebar.selectbox('Kişisel Kredi', ('no', 'yes'))
    contact = st.sidebar.selectbox('İletişim Tipi', ('cellular', 'telephone', 'No Info'))
    day = st.sidebar.slider('Gün', 1, 31, 15)
    month = st.sidebar.selectbox('Ay', ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
    duration = st.sidebar.slider('Son İletişim Süresi (saniye)', 0, 5000, 300)
    campaign = st.sidebar.slider('Bu Kampanya İçin İletişim Sayısı', 1, 60, 2)
    pdays = st.sidebar.slider('Önceki Kampanyadan Bu Yana Geçen Gün (-1 = hiç aranmamış)', -1, 999, -1)
    previous = st.sidebar.slider('Önceki Kampanyalarda İletişim Sayısı', 0, 20, 0)
    poutcome = st.sidebar.selectbox('Önceki Kampanya Sonucu', ('success', 'failure', 'other', 'No Info'))

    # Tüm girişleri bir sözlük olarak topla
    data = {
        'age': age,
        'job': job, 
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    
    # Sözlüğü pandas DataFrame'e dönüştür
    features = pd.DataFrame(data, index=[0])
    return features

# Veri ön işleme fonksiyonu
def preprocess_data(input_df, feature_names):
    """
    Kullanıcı girdilerini model girdilerine dönüştürür
    """
    # Kopya oluştur
    df = input_df.copy()
    
    # 1. Sayısal değişkenler için logaritmik dönüşüm
    if 'balance_log' in feature_names:
        # Sayıya çevirerek min işlemini gerçekleştir - Series hatası için
        balance_min = float(min(0, df['balance'].iloc[0]))
        df['balance_log'] = np.log(df['balance'] + abs(balance_min) + 1)
    
    if 'duration_log' in feature_names:
        df['duration_log'] = np.log(df['duration'] + 1)
    
    # 2. Binary değişkenler için kodlama
    if 'housing_encoded' in feature_names:
        df['housing_encoded'] = df['housing'].map({'no': 0, 'yes': 1})
    
    if 'loan_encoded' in feature_names:
        df['loan_encoded'] = df['loan'].map({'no': 0, 'yes': 1})
    
    # 3. Özellik mühendisliği özellikleri
    # Previous success rate - model için gerekli
    if 'previous_success_rate' in feature_names:
        # DataFrame içindeki campaign değerine güvenli erişim
        campaign_value = float(df['campaign'].iloc[0])
        if campaign_value == 0:
            campaign_value = 1
        
        # DataFrame içindeki previous değerine güvenli erişim
        previous_value = float(df['previous'].iloc[0])
        
        # previous_success_rate'i tek bir değer olarak hesapla
        df['previous_success_rate'] = previous_value / campaign_value
    
    # Contacted before - integer dönüşümünü düzeltelim
    if 'contacted_before' in feature_names:
        previous_value = float(df['previous'].iloc[0])
        df['contacted_before'] = 1 if previous_value > 0 else 0
    
    # pdays_long_ago - integer dönüşümünü düzeltelim
    if 'pdays_long_ago' in feature_names:
        pdays_value = float(df['pdays'].iloc[0])
        df['pdays_long_ago'] = 1 if pdays_value == -1 else 0
    
    # avg_duration_per_campaign
    if 'avg_duration_per_campaign' in feature_names:
        # DataFrame içindeki değerlere güvenli erişim
        duration_value = float(df['duration'].iloc[0])
        campaign_value = float(df['campaign'].iloc[0])
        if campaign_value == 0:
            campaign_value = 1
        
        df['avg_duration_per_campaign'] = duration_value / campaign_value
    
    # day_of_week
    if 'day_of_week' in feature_names:
        day_value = int(df['day'].iloc[0])
        df['day_of_week'] = day_value % 7
    
    # 4. Kategorik değişkenler için özel işleme
    # contact_cellular için özel işleme
    if 'contact_cellular' in feature_names:
        contact_value = str(df['contact'].iloc[0])
        df['contact_cellular'] = 1 if contact_value == 'cellular' else 0
    
    # poutcome_success için özel işleme
    if 'poutcome_success' in feature_names:
        poutcome_value = str(df['poutcome'].iloc[0])
        df['poutcome_success'] = 1 if poutcome_value == 'success' else 0
    
    # poutcome_failure için özel işleme
    if 'poutcome_failure' in feature_names:
        poutcome_value = str(df['poutcome'].iloc[0])
        df['poutcome_failure'] = 1 if poutcome_value == 'failure' else 0
    
    # 5. Sadece model için gerekli olan özellikleri seç
    processed_df = pd.DataFrame()
    
    # Türetilen özellikler listesi - bunlar için uyarı göstermeyelim
    derived_features = [
        'balance_log', 'duration_log', 'previous_success_rate',
        'contacted_before', 'pdays_long_ago', 'avg_duration_per_campaign',
        'day_of_week', 'housing_encoded', 'loan_encoded',
        'contact_cellular', 'poutcome_success', 'poutcome_failure'
    ]
    
    # Eksik olup türetilmeyen özellikler için uyarı verelim
    missing_features = []
    
    for feature in feature_names:
        if feature in df.columns:
            processed_df[feature] = df[feature]
        else:
            # Eğer türetilen bir özellik değilse uyarı listesine ekle
            if feature not in derived_features:
                missing_features.append(feature)
            # Her durumda eksik özelliklere 0 değeri ata
            processed_df[feature] = 0
    
    # Eksik özellikler varsa uyarı göster (türetilen özellikler hariç)
    if missing_features:
        st.warning(f"Aşağıdaki özellikler oluşturulurken varsayılan değerler atandı: {', '.join(missing_features)}")
    
    return processed_df

# Ana uygulamayı çalıştır
def run_app():
    # Model, özellik isimleri ve etiket kodlayıcıyı yükle
    model, feature_names, label_encoder = load_model_and_resources()
    
    if not model or not feature_names or not label_encoder:
        st.error("Model veya gerekli kaynaklar yüklenemedi.")
        return
    
    # Kullanıcı girdilerini al
    input_df = user_input_features()
    
    # Girdileri göster
    st.write('## Girilen Müşteri Bilgileri')
    st.write(input_df)
    
    # Tahmin butonu
    if st.button('Tahmin Yap'):
        # Veri ön işleme
        with st.spinner("Veriler işleniyor..."):
            try:
                processed_data = preprocess_data(input_df, feature_names)
                
                st.write("## İşlenmiş Veriler")
                st.write("Model için hazırlanan veriler:")
                st.write(processed_data)
                
                # Tahmin yap
                prediction = model.predict(processed_data)
                prediction_proba = model.predict_proba(processed_data)[0]
                
                # Sonucu göster
                st.write('## Tahmin Sonucu')
                
                # Tahmin edilen sınıfı orijinal etiketlere çevir
                predicted_class = label_encoder.inverse_transform([prediction[0]])[0]
                
                if predicted_class == 'yes':
                    st.success("**Tahmin: MÜŞTERİ VADELİ MEVDUAT ÜRÜNÜNE ABONE OLACAK** 💰")
                    st.balloons()
                else:
                    st.error("**Tahmin: MÜŞTERİ VADELİ MEVDUAT ÜRÜNÜNE ABONE OLMAYACAK** ❌")
                
                # Olasılıkları göster
                st.write("### Tahmin Olasılıkları")
                class_labels = label_encoder.classes_
                prob_df = pd.DataFrame({
                    'Sonuç': [f"{class_labels[0]} (Abone Olmayacak)", f"{class_labels[1]} (Abone Olacak)"],
                    'Olasılık': [prediction_proba[0], prediction_proba[1]]
                })
                
                st.bar_chart(prob_df.set_index('Sonuç'))
                
                # Ek bilgi
                st.info(f"Bu müşterinin vadeli mevduat ürününe abone olma olasılığı: {prediction_proba[1]:.2%}")
                
            except Exception as e:
                st.error(f"Tahmin sırasında bir hata oluştu: {str(e)}")
                st.error("Lütfen veri işleme adımlarını gözden geçirin.")

# Uygulamayı çalıştır
if __name__ == '__main__':
    run_app()