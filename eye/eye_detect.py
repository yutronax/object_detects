import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import shutil
from PIL import Image
import cv2

# Resimlerin olduğu klasörleri ve dosyaları düzgün isimlendirmek için fonksiyon
def klasorolusturma(path, name):
    os.makedirs(path, exist_ok=True)
    liste = os.listdir(path)
    sayac = 0
    for i in liste:
        if i.lower().endswith((".jpg", ".jpeg", ".png")):
            eski_ad = os.path.join(path, i)
            yeni_ad = os.path.join(path, f"{name}{sayac}.jpg")
            os.rename(eski_ad, yeni_ad)
            sayac += 1

# Göz ve random resim klasörlerini düzenliyoruz
random_path = "/content/random_images/"
eye_path = "/content/eyes/"
klasorolusturma(eye_path, "eye")
klasorolusturma(random_path, "random")

# Eğitim, test ve doğrulama (validation) klasörlerini oluşturuyoruz
os.makedirs("/content/eye_detect/train/eyes", exist_ok=True)
os.makedirs("/content/eye_detect/train/random_images", exist_ok=True)
os.makedirs("/content/eye_detect/test/eyes", exist_ok=True)
os.makedirs("/content/eye_detect/test/random_images", exist_ok=True)
os.makedirs("/content/eye_detect/val/eyes", exist_ok=True)
os.makedirs("/content/eye_detect/val/random_images", exist_ok=True)

# Dosyaları train, test, val klasörlerine bölmek için fonksiyon
def fonk(asil_dosya, train, test, val):
    path_liste = os.listdir(asil_dosya)
    np.random.shuffle(path_liste)

    n = len(path_liste)
    trainler = path_liste[:int(n * 0.6)]
    valler = path_liste[int(n * 0.6):int(n * 0.8)]
    testler = path_liste[int(n * 0.8):]

    for i in testler:
        eski_adres = os.path.join(asil_dosya, i)
        shutil.copy2(eski_adres, os.path.join(test, i))
    for i in trainler:
        eski_adres = os.path.join(asil_dosya, i)
        shutil.copy2(eski_adres, os.path.join(train, i))
    for i in valler:
        eski_adres = os.path.join(asil_dosya, i)
        shutil.copy2(eski_adres, os.path.join(val, i))

# Klasör yolları
train_path_eye = "/content/eye_detect/train/eyes/"
test_path_eye = "/content/eye_detect/test/eyes/"
val_path_eye = "/content/eye_detect/val/eyes/"
train_path_random = "/content/eye_detect/train/random_images/"
test_path_random = "/content/eye_detect/test/random_images/"
val_path_random = "/content/eye_detect/val/random_images/"

# Verileri ayırıyoruz
fonk(eye_path, train_path_eye, test_path_eye, val_path_eye)
fonk(random_path, train_path_random, test_path_random, val_path_random)

# Bozuk dosyaları silmek için fonksiyon
def bozuk_dosyalari_sil(yol):
    for klasor, _, dosyalar in os.walk(yol):
        for dosya in dosyalar:
            dosya_yolu = os.path.join(klasor, dosya)
            try:
                img = Image.open(dosya_yolu)
                img.verify()  # sadece doğrulama yapıyoruz
            except Exception as e:
                print(f"❌ Silindi: {dosya_yolu} ({e})")
                os.remove(dosya_yolu)

# Bozuk dosyaları temizliyoruz
bozuk_dosyalari_sil("/content/eye_detect/train")
bozuk_dosyalari_sil("/content/eye_detect/val")
bozuk_dosyalari_sil("/content/eye_detect/test")

# Her klasörde kaç dosya kaldığını kontrol ediyoruz
print("val random: ", len(os.listdir(val_path_random)))
print("val eye: ", len(os.listdir(val_path_eye)))
print("train random: ", len(os.listdir(train_path_random)))
print("train eye: ", len(os.listdir(train_path_eye)))
print("test random: ", len(os.listdir(test_path_random)))
print("test eye: ", len(os.listdir(test_path_eye)))

train_path = "/content/eye_detect/train"
val_path = "/content/eye_detect/val"

# Veri artırma (augmentation) için ImageDataGenerator ayarları
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Piksel değerlerini 0-1 aralığına çekiyoruz
    rotation_range=40,       # Resimleri döndürerek çeşitlendiriyoruz
    width_shift_range=0.2,   # Yatay kaydırma
    height_shift_range=0.2,  # Dikey kaydırma
    shear_range=0.2,         # Kesme dönüşümü
    zoom_range=0.2,          # Yakınlaştırma
    horizontal_flip=True,    # Yatay çevirme
    fill_mode='nearest'      # Eksik pikselleri doldurma yöntemi
)

# Eğitim verisini generator ile yüklüyoruz
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(300, 300),  # Model giriş boyutu
    batch_size=128,
    class_mode='binary'      # İkili sınıflandırma (göz veya değil)
)

# Validation için sadece normalize ediyoruz, augmentation yok
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    val_path,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)

# CNN modelimizi oluşturuyoruz
cnn_model = tf.keras.models.Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=(300, 300, 3)),  # İlk konvolüsyon katmanı
    layers.MaxPooling2D(2, 2),  # Havuzlama ile boyut küçültme

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),            # Çok boyutlu çıktıyı tek vektöre çeviriyoruz
    layers.Dense(512, activation='relu'),  # Yoğun bağlantılı gizli katman
    layers.Dense(1, activation='sigmoid')  # Çıkış katmanı, 0-1 arası değer
])

cnn_model.summary()  # Modelin özetini yazdırıyoruz

# Modeli derliyoruz
from tensorflow.keras.optimizers import RMSprop
cnn_model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['acc']
)

# Eğitim sırasında erken durdurma ve en iyi modeli kaydetme için callback'ler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,             # 3 epoch üst üste düzelme olmazsa durdur
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Modeli eğitiyoruz
history = cnn_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)

# Eğitim ve doğrulama kayıplarını grafikle gösteriyoruz
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.figure(figsize=(12, 8))
plt.plot(history.history['acc'], label='train acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

# Test verisi için sadece normalize ediyoruz
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "/content/eye_detect/test",
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Confusion matrix için sıralı olması lazım
)

# Test setinde modeli değerlendiriyoruz
test_loss, test_acc = cnn_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Test verisindeki tahminlerimizi alıyoruz
y_pred_probs = cnn_model.predict(test_generator)

# Tahminleri 0.5 eşik değerine göre sınıflandırıyoruz
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Gerçek etiketleri alıyoruz
y_true = test_generator.classes

# Karışıklık matrisi hesaplıyoruz ve görselleştiriyoruz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
labels = list(test_generator.class_indices.keys())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.show()

# Diğer performans metriklerini yazdırıyoruz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))

# Kaydedilen en iyi modeli yüklüyoruz
model = tf.keras.models.load_model('best_model.h5')

# Test resmi üzerinde tahmin yapmak için hazırlık
img = cv2.imread('test_image.jpg')                      # Resmi OpenCV ile okuyoruz
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR'den RGB'ye çeviriyoruz

input_size = (300, 300)                                 # Model giriş boyutu
img_resized = cv2.resize(img_rgb, input_size)           # Resmi yeniden boyutlandırıyoruz
img_normalized = img_resized / 255.0                     # Piksel değerlerini normalize ediyoruz
input_tensor = np.expand_dims(img_normalized, axis=0)    # Batch dimension ekliyoruz

# Model ile tahmin yapıyoruz
prediction = model.predict(input_tensor)
print("Tahmin sonucu:", prediction)

# Tahmini 0.5 eşiğine göre sınıflandırıyoruz
y_pred_single = (prediction > 0.5).astype("int32").flatten()
print("Sınıf tahmini:", y_pred_single[0])

# Eğer göz tespit edildiyse resmi gösteriyoruz
if y_pred_single[0] == 1:
    plt.imshow(img_rgb)
    plt.title("Göz Tespit Edildi")
    plt.axis('off')
    plt.show()
else:
    print("Göz tespit edilmedi.")
