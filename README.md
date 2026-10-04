# Energy Consumption Estimation

Python ile simüle edilmiş günlük tüketim verilerinde regresyon denemesi. Gerçek şirket, müşteri veya sayaç verisi içermez; gerçek tesis başarısı iddiası değildir.

## Yöntem

`train.py`, tarih sırası doğrulanmış günlük seride sıcaklık, fiyat, ay, haftanın günü ve önceki günün tüketimini kullanır. İlk %80 eğitim, son %20 testtir. Eksik değer doldurma yalnızca eğitim bölümünde öğrenilir. Ortalama baseline, Linear Regression ve Random Forest; MAE, RMSE ve R² ile karşılaştırılır.

Test gününün sıcaklık/fiyatı ve önceki günün **gözlenen** tüketimi kullanıldığı için sonuç bir adım ileri, kayan değerlendirmedir. Çok gün ileri tahmin veya yalnızca tahmin anında bilinen değişkenlerle üretim tahmini sayılmaz. Simüle veride ilişki zayıfsa modeller basit ortalamayı geçmeyebilir; başarı iddiası sonuçlara göre yapılmalıdır.

```sh
git clone https://github.com/hamzaguner0/Energy_Consumption_Estimation.git
cd Energy_Consumption_Estimation
python -m pip install -r requirements.txt
python train.py
```

Çıktılar `artifacts/metrics.json`, `artifacts/predictions.csv` ve yerel model dosyalarıdır. Yalnızca güvenilir, kendiniz oluşturduğunuz joblib/pickle dosyalarını yükleyin.

## Tarihsel dosyalar

`energy_consumption_analysis.ipynb` ilk keşif/öğrenme çalışmasıdır; rastgele bölümleme kullanır ve geleceğe tahmin başarısı olarak yorumlanmamalıdır. `energy_test_set.csv`, `nergy_train_set.csv` ve `predicted_test_set.csv` tarihsel çıktılardır; yazım hatalı dosya adı geçmiş uyumluluğu için korunmuştur. Eski `energy_consumption_model.pkl` güncel eğitim akışının doğrulanmış modeli değildir.

Veri sütunları: `Date`, `Energy_Consumption_kWh`, `Temperature_C`, `Electricity_Price_€/kWh`. Verinin simüle olduğu ilk proje açıklamasına dayanır; üretici kodu/lisans kökeni bu depoda ayrıca belgelenmemiştir.
