# Banka Pazarlama Kampanyası Tahmin Uygulaması

Bu proje, banka müşterilerinin vadeli mevduat ürününe abone olup olmayacaklarını tahmin etmek için makine öğrenmesi modelleri kullanıyor. Veri seti, Portekizli bir bankanın 2008-2010 yılları arasında gerçekleştirdiği doğrudan pazarlama kampanyalarına ait müşteri verilerinden oluşmaktadır.

## Proje Bileşenleri

1. **Veri Analizi ve Model Eğitimi**: `notebooks/proje.ipynb` dosyasında yer almaktadır.
2. **Model Dışa Aktarma**: `model_save.py` betiği ile en iyi model `.pkl` formatında dışa aktarılır.
3. **Streamlit Web Uygulaması**: `app.py` dosyası ile kullanıcıların etkileşimli olarak tahmin yapmasını sağlar.

## Kurulum

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## Kullanım

1. **Model Eğitimi ve Dışa Aktarma**:

```bash
python model_save.py
```

Bu komut, en iyi modeli eğitip `best_model.pkl` olarak kaydeder. Ayrıca `feature_names.pkl` ve `label_encoder.pkl` dosyalarını da oluşturur.

2. **Streamlit Uygulamasını Çalıştırma**:

```bash
streamlit run app.py
```

Tarayıcıda açılan uygulamada, sol taraftaki kontrol panelinden müşteri bilgilerini girin ve "Tahmin Yap" butonuna tıklayarak sonucu görüntüleyin.

## Proje Özellikleri

- Veri ön işleme ve özellik mühendisliği
- Özellik seçimi için Recursive Feature Elimination (RFE) kullanımı
- Gradient Boosting sınıflandırma algoritması
- Etkileşimli web arayüzü ile canlı ortam tahminleri

## Model Özellikleri

RFE özellik seçim yöntemi ile belirlenen en önemli özellikler:

- `age`: Müşterinin yaşı
- `day`: Ayın günü
- `campaign`: Bu kampanya boyunca yapılan iletişim sayısı
- `balance_log`: Bakiye değerinin logaritmik dönüşümü
- `duration_log`: İletişim süresinin logaritmik dönüşümü
- `pdays_long_ago`: Önceki kampanyadan beri uzun süre geçmiş olması
- `avg_duration_per_campaign`: Kampanya başına ortalama iletişim süresi
- `previous_success_rate`: Önceki iletişimlerin başarı oranı
- `poutcome_success`: Önceki kampanyanın başarılı olması
- `poutcome_failure`: Önceki kampanyanın başarısız olması
- `contact_cellular`: İletişim tipinin cep telefonu olması
- `housing_encoded`: Konut kredisi varlığı
- `loan_encoded`: Kişisel kredi varlığı

## Teknik Notlar

- `unknown` değerleri, model içinde `No Info` olarak yeniden adlandırılmıştır
- One-hot encoding sütun adlandırması, boşluk içerebilir (örn: `poutcome_No Info`)
- Streamlit uygulaması, her türlü giriş kombinasyonunu işleyecek şekilde tasarlanmıştır

## Veri Seti Özellikleri

Veri seti, müşterilerin demografik ve bankacılık davranışlarına ilişkin bilgiler içerir:

- **Yaş, meslek, medeni durum** gibi kişisel bilgiler
- **Bakiye, kredi durumu, konut kredisi** gibi finansal bilgiler
- **İletişim tipi, gün, ay** gibi kampanya iletişim bilgileri
- **Süre, kampanya sayısı, önceki sonuç** gibi kampanya performans ölçütleri

Hedef değişken: Müşterinin vadeli mevduat ürününe abone olup olmaması ('yes'/'no') 