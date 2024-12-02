# Import library
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Membaca data
df = pd.read_csv("most popular.csv")
reviews = df["comment"].tolist()  # Kolom ulasan
labels = df["coresponding"].tolist()  # Kolom label (0 atau 1)

# Pastikan semua elemen dalam 'reviews' adalah string
reviews = [str(review) for review in reviews]

# Tokenisasi ulasan
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)  # Mengatasi error
sequences = tokenizer.texts_to_sequences(reviews)
word_index = tokenizer.word_index
print(f"Jumlah Kata Unik: {len(word_index)}")

# Padding sequences
max_length = 4  # Menggunakan panjang maksimal
x = pad_sequences(sequences, maxlen=max_length)
y = np.array(labels)

# Pisahkan data menjadi training dan testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Membangun model RNN
model = Sequential(
    [
        Embedding(input_dim=len(word_index)+1, output_dim=8, input_length=max_length),  # Layer embedding
        LSTM(50, activation="relu"),  # RNN Layer
        Dense(1, activation="sigmoid")  # Layer output
    ]
)
# Membuat model agar dapat di-*build* sebelum summary
model.build(input_shape=(None, max_length))

# Menampilkan arsitektur model
model.summary()

# Kompilasi model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Melatih model dengan data pelatihan
model.fit(x_train, y_train, epochs=10, batch_size=2, verbose=1)

# Melatih model dengan data pelatihan dan menyimpan history untuk grafik
# history = model.fit(x_train, y_train, epochs=10, batch_size=2, validation_data=(x_test, y_test), verbose=1)


# Prediksi pada data uji
y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Mengonversi prediksi menjadi 0 atau 1

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Akurasi: {accuracy * 100:.2f}%")

# Evaluasi model pada data uji
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nEvaluasi Model: Loss = {loss:.4f}, Akurasi = {accuracy*100:.2f}%")



# Prediksi ulasan baru
new_reviews = ["Pascal is RIP", "phyton, js and ts the best languages"]
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_x = pad_sequences(new_sequences, maxlen=max_length)
predictions = model.predict(new_x)
print(predictions)
# Output hasil prediksi untuk ulasan baru
print("Prediksi untuk ulasan baru:")
for review, prediction in zip(new_reviews, predictions):
    sentiment = "positif" if prediction > 0.5 else "negatif"
    print(f"Ulasan: {review}, Sentimen: {sentiment}")


# # Grafik akurasi dari hasil pelatihan
# plt.plot(history.history['accuracy'], label='Akurasi Pelatihan')
# plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
# plt.xlabel('Epoch')
# plt.ylabel('Akurasi')
# plt.legend()
# plt.show()