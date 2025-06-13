# Göz Tespiti CNN Modeli

Bu projede TensorFlow ve Keras kullanarak basit bir CNN modeli ile gözleri tespit etmeye çalışıyoruz. Elimizde göz içeren ve içermeyen fotoğraflar var; modelimiz bu fotoğrafları öğrenip yeni resimlerde göz olup olmadığını ayırt etmeye çalışıyor.

---

## Proje Hakkında

- Veri setimizi eğitim, doğrulama ve test olmak üzere üçe böldük.
- Görüntüleri modele uygun boyuta getirip, veri artırma teknikleri ile çeşitlendirdik.
- CNN modeli oluşturduk ve eğittik.
- Erken durdurma ve en iyi modeli kaydetme gibi yöntemlerle aşırı öğrenmeyi önledik.
- Test verisiyle modelin başarısını ölçtük.
- Son olarak, eğitimli modelle yeni fotoğraflarda göz olup olmadığını tahmin ettik.

---

## Veri Hazırlığı

Elimizde iki klasör var:  
- `eyes`: Göz içeren resimler  
- `random_images`: Göz içermeyen resimler  

Bu fotoğrafları rastgele şekilde %60 eğitim, %20 doğrulama, %20 test olarak farklı klasörlere kopyaladık. Ayrıca bozuk dosyalar varsa onları temizledik.

---

## Model Eğitimi

- Verileri 300x300 piksel boyutuna getirdik.
- Eğitim sırasında resimlere küçük dönüşümler uygulayarak (döndürme, kaydırma, zoom, vb.) çeşitlilik sağladık.
- CNN modelimizi oluşturduk ve 15 epoch boyunca eğittik.
- En iyi modeli `best_model.h5` olarak kaydettik.

---

## Değerlendirme ve Tahmin

- Test verisinde doğruluk, precision, recall ve F1 skorlarını hesapladık.
- Karışıklık matrisi ile sonuçları görselleştirdik.
- Yeni bir resim yükleyip, modelin tahminini aldık.
- Eğer model göz tespit ederse ekranda gösterdik.

---

## Nasıl Kullanılır?

```python
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('best_model.h5')

img = cv2.imread('test_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (300, 300))
img_normalized = img_resized / 255.0
input_tensor = np.expand_dims(img_normalized, axis=0)

prediction = model.predict(input_tensor)

if prediction > 0.5:
    print("Göz var!")
else:
    print("Göz yok.")
