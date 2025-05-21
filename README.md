# Laporan Proyek Machine Learning - Faiza Adinda Fakhira Batubara

## Project Overview

Dalam era digital saat ini, jumlah pilihan makanan yang tersedia secara online sangat melimpah, mulai dari berbagai jenis masakan lokal hingga internasional. Hal ini seringkali membuat seseorang kebingungan dalam memilih makanan yang sesuai dengan preferensi mereka. Oleh karena itu, dibutuhkan sistem yang mampu memberikan rekomendasi makanan yang relevan dan personal, untuk meningkatkan pengalaman pengguna dalam menemukan makanan yang mereka sukai.

Salah satu pendekatan yang terbukti efektif dalam menyelesaikan permasalahan ini adalah penggunaan Sistem Rekomendasi (Recommendation System). Sistem rekomendasi merupakan sebuah Artificial Intelligence (AI), yang biasa berhubungan dengan Machine Learning dan Big Data untuk merekomendasikan suatu produk kepada konsumen. Sistem tersebut dapat didasari dari beberapa kriteria yaitu seperti riwayat pembelian, riwayat pencarian, dan faktor lainnya [1]. Sistem Rekomendasi memiliki beberapa jenis yaitu:

- Content Based Filtering

  Merupakan metode yang merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu

- Collaborative Filtering

  Merupakan metode yang merekomendasikan item dengan menggunakan informasi dari aktivitas pengguna

- Hybrid Recommender System

  Merupakan metode penggabungan antara Content Based Filtering dan Collaborative Filtering

Rubrik/Kriteria Tambahan (Opsional):
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding
### Problem Statements

- Banyak orang kesulitan menemukan makanan yang sesuai dengan preferensi pribadi mereka. Dalam jumlah makanan yang sangat banyak, mereka sering tidak tahu makanan apa yang cocok untuk tanpa harus melihat satu per satu.
- Sistem rekomendasi makanan yang ada saat ini masih cenderung sederhana, banyak yang hanya mengandalkan popularitas atau pencocokan kata kunci tanpa mempertimbangkan preferensi pengguna secara mendalam.
- Evaluasi performa sistem rekomendasi makanan diperlukan untuk memastikan bahwa rekomendasi yang dihasilkan bersifat akurat, relevan, dan dapat diandalkan dalam membantu pengguna menemukan makanan yang sesuai dengan preferensi mereka.

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Membangun sistem rekomendasi makanan yang mampu menyarankan makanan relevan berdasarkan karakteristik makanan. Ini ditujukan untuk mengatasi masalah pengguna yang tidak tahu apa yang ingin mereka coba dengan melihat makanan serupa dari segi konten.
- Mengembangkan sistem rekomendasi makanan yang lebih canggih dengan menggabungkan pendekatan content-based filtering dan collaborative filtering.
- Melakukan evaluasi terhadap model rekomendasi menggunakan metrik seperti cosine similarity score (untuk content-based filtering) dan  Mean Squared Error (MSE) atau  Mean Absolute Error (MAE) (untuk collaborative filtering) guna menilai kualitas saran yang dihasilkan serta efektivitas model dalam memberikan rekomendasi yang sesuai dengan kebutuhan pengguna.
  
    ### Solution statements
  Untuk mencapai goals-goals diatas, pendekatan yang digunakan adalah sebagai berikut:
    - Content-Based Filtering (CBF)

      Model Content-Based Filtering menyarankan makanan berdasarkan kemiripan konten antar makanan. Untuk implementasinya, data seperti nama makanan, deskripsi, dan kategori digabung menjadi satu kesatuan teks,        kemudian direpresentasikan secara numerik menggunakan teknik TF-IDF (Term Frequency–Inverse Document Frequency). Kemudian, cosine similarity digunakan untuk menghitung kemiripan antar makanan. Berdasarkan 
      makanan yang pernah disukai pengguna, sistem akan menyarankan makanan lain yang memiliki karakteristik serupa.

    - Collaborative Filtering

      Model Collaborative Filtering menyarankan makanan berdasarkan pola rating dari pengguna lain yang memiliki preferensi serupa. Dalam implementasinya, digunakan model deep learning dengan embedding layer 
      untuk mempelajari representasi dari pengguna dan makanan. Sistem kemudian memprediksi makanan mana yang kemungkinan besar akan disukai oleh pengguna berdasarkan riwayat interaksi sebelumnya. Berbeda 
      dengan CBF, pendekatan ini tidak bergantung pada deskripsi makanan, melainkan fokus pada pola perilaku pengguna dalam memberikan rating.
      

## Data Understanding
Pada proyek ini, digunakannya dataset [Food Recommendation System](https://www.kaggle.com/datasets/schemersays/food-recommendation-system/data) dari Kaggle. Dataset ini memiliki 2 file yaitu 1 dataset terkait makanan, bahan makanan, dan jenis makanan. Lalu, 1 dataset terkait rating yang diberikan setiap pengguna.

### Variabel-variabel pada Food Recommendation System dataset adalah sebagai berikut:
- ```Food_ID``` : Id unik makanan
- ```Name``` : nama makanan
- ```C_Type```: jenis makanan
- ```Veg_Non```: Id unik pengguna
- ```Describe```: bahan-bahan makanan
- ```User_ID```: Id unik pengguna
- ```Rating```: nilai rating (1 sampai 10)

 ### Exploratory Data Analysis
Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

Berikut ini adalah EDA yang dilakukan:
```python
  # Melihat info umum dataset (fitur, type data, jumlah entry, dan nilai null)
  print("Movies Info:")
  print(movie.info())
  print("\nRatings Info:")
  print(rating.info())
  ```

Output dari kode diatas yaitu:

```
Food Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 5 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Food_ID   400 non-null    int64 
 1   Name      400 non-null    object
 2   C_Type    400 non-null    object
 3   Veg_Non   400 non-null    object
 4   Describe  400 non-null    object
dtypes: int64(1), object(4)
memory usage: 15.8+ KB
None

Ratings Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 512 entries, 0 to 511
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   User_ID  511 non-null    float64
 1   Food_ID  511 non-null    float64
 2   Rating   511 non-null    float64
dtypes: float64(3)
memory usage: 12.1 KB
None
```

Dapat dilihat dari output kode diatas, untuk data Food memiliki 400 entries dengan 5 atribut dan untuk data Rating memiliki 512 entries dengan 3 atribut.

Selanjutnya dengan kode dibawah ini, kita akan melihat apakah ada missing value dan nilai duplikat pada dataset

```python
  # Melihat apakah ada missing value
print("Missing values in movies:\n", food.isnull().sum())
print("Missing values in ratings:\n", rating.isnull().sum())

# Melihat apakah ada nilai duplikat
print("Duplicated rows in movies:", food.duplicated().sum())
print("Duplicated rows in ratings:", rating.duplicated().sum())
  ```
Output kode diatas yaitu:

```
Missing values in movies:
 Food_ID     0
Name        0
C_Type      0
Veg_Non     0
Describe    0
dtype: int64
Missing values in ratings:
 User_ID    1
Food_ID    1
Rating     1
dtype: int64
Duplicated rows in movies: 0
Duplicated rows in ratings: 0
```

Terlihat dari output kode diatas, pada data ```food``` tidak memiliki missing value dan nilai duplikat. Sedangkan pada data ```rating``` memiliki missing value dan tidak memiliki nilai duplikat.

Lalu, kita akan melihat jumlah nilai-nilai unik pada kolom ```Name``` dan ```C_Type```
```phyton
# Melihat nilai unik makanan dan course makanan
print("Jumlah unique nama makanan:", food['Name'].nunique())
print("Jumlah unique Course:", food['C_Type'].nunique())
```

Output kode diatas yaitu:

```
Jumlah unique nama makanan: 400
Jumlah unique Course: 16
```

Terlihat dari output kode diatas dataset memiliki 400 nilai unik makanan dan 16 jenis makanan. Dengan begitu, dataset berpotensi mendukung pembuatan sistem rekomendasi yang personal dan kontekstual. 

Selanjutnya, kita akan memulai EDA untuk data rating. Pada code dibawah ini kita akan melihat statistik deskriptif untuk data rating
```phyton
# Melihat statistik deskriptif data rating
rating.describe()
```
Output kode diatas yaitu:

| Statistic |   User_ID  |   Food_ID  |   Rating   |
|-----------|------------|------------|------------|
| Count     | 511.000000 | 511.000000 | 511.000000 |
| Mean      | 49.068493  | 125.311155 | 5.438356   |
| Std       | 28.739213  | 91.292629  | 2.866236   |
| Min       | 1.000000   | 1.000000   | 1.000000   |
| 25%       | 25.000000  | 45.500000  | 3.000000   |
| 50%       | 49.000000  | 111.000000 | 5.000000   |
| 75%       | 72.000000  | 204.000000 | 8.000000   |
| Max       | 100.000000 | 309.000000 | 10.000000  |

Data statistik menunjukkan bahwa terdapat 100 pengguna dan 309 jenis makanan dengan total 511 interaksi rating. Rata-rata rating berada di angka 5.44 dengan nilai minimum 1 dan maksimum 10, serta standar deviasi sebesar 2.87 yang menandakan adanya variasi preferensi antar pengguna. Sebagian besar rating berada pada rentang menengah ke atas, yang mengindikasikan kecenderungan pengguna memberikan penilaian positif terhadap makanan. Hal ini menjadi dasar yang baik untuk pengembangan sistem rekomendasi dengan menggunakan metode Collaborative Filtering.

### Visualisasi Data
- Visualisasi distribusi kolom C_Type
  
  <p align="center">
  <img src="https://github.com/user-attachments/assets/f2d35827-ccab-46a1-b392-927dc6af47cd" alt="Distribusi C_Type" width="600">
  </p>
  Visualisasi menunjukkan bahwa makanan dengan kategori Indian mendominasi jumlah data, disusul oleh healthy food dan dessert. Ini mengindikasikan bahwa makanan khas India paling banyak direpresentasikan dalam 
  dataset.

- Visualisasi distribusi kolom Veg_Non
  
  <p align="center">
  <img src="https://github.com/user-attachments/assets/a8ea0e41-f133-4676-bbbc-ddff882799ae" alt="Distribusi C_Type" width="600">
  </p>
  Grafik ini memperlihatkan bahwa makanan vegetarian (veg) lebih banyak dibandingkan non-vegetarian (non-veg), yang menunjukkan preferensi yang lebih tinggi terhadap makanan vegetarian.

- Visualisasi distribusi kolom Rating
  
  <p align="center">
  <img src="https://github.com/user-attachments/assets/cea736e7-3f4c-4ae5-a689-54c0318b0925" alt="Distribusi C_Type" width="600">
  </p>
  Rating makanan bervariasi dari 1 hingga 10, dengan rating 3, 5, dan 10 menjadi yang paling banyak diberikan. Hal ini menandakan persebaran opini pengguna cukup luas, dengan kecenderungan terhadap rating 
  positif (tinggi).



## Data Preparation

  Pada tahap ini akan dilakukan cleaning data, yaitu menghapus data missing value. Melakukan cleaning data sangat penting karena dapat meningkatkan kualitas dan keakuratan data.

  Kode dibawah ini melakukan penghapusan untuk missing value:
  ```python
   # Menghapus missing value pada data rating
   rating.dropna(axis=0 ,inplace=True)

   # Cek kembali missing value
   print("Missing values in ratings:\n", rating.isnull().sum())
  ```

  Output kode di atas yaitu:
  ```
  Missing values in ratings:
  User_ID    0
  Food_ID    0
  Rating     0
  dtype: int64
  ```
  Dapat dilihat dari output di atas, bahwa data ```rating``` sudah bersih dari missing value

  Selanjutnya, kita akan mengubah tipe data dengan kode di bawah, untuk memastikan setiap kolom memiliki tipe data yang sesuai dengan fungsinya. Kolom ```User_ID``` dan ```Food_ID``` diubah ke tipe ```int``` 
  karena berisi nilai numerik diskrit sebagai identifikasi, sementara kolom ```Rating``` diubah ke tipe ```float``` agar dapat mendukung perhitungan statistik yang lebih fleksibel, seperti rata-rata atau 
  pembobotan dalam model rekomendasi.

  ```python
   # Mengubah type data User_ID, Food_ID, dan Rating
   rating['User_ID'] = rating['User_ID'].astype(int)
   rating['Food_ID'] = rating['Food_ID'].astype(int)
   rating['Rating'] = rating['Rating'].astype(float)
  ```

  Lalu, kode dibawah ini akan melakukan cleaning (pembersihan) data teks pada beberapa kolom di dataset makanan, yaitu ```Name```, ```C_Type```, ```Veg_Non```, dan ```Describe```. Tujuan utamanya adalah menghapus
  karakter non-alfanumerik (seperti tanda baca dan simbol), serta mengubah semua huruf menjadi huruf kecil. Langkah ini penting agar data lebih konsisten dan tidak terjadi duplikasi yang disebabkan 
  oleh perbedaan format penulisan, sehingga hasil analisis atau pemodelan menjadi lebih akurat.
  
  ```python
   def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())
    return ''

  # Menerapkan fungsi pembersihan teks ke kolom-kolom teks pada dataset
  food['Name'] = food['Name'].apply(clean_text)
  food['C_Type'] = food['C_Type'].apply(clean_text)
  food['Veg_Non'] = food['Veg_Non'].apply(clean_text)
  food['Describe'] = food['Describe'].apply(clean_text)
  ```

## Modeling and Result
Pada tahap ini, dilakukan pembangunan dua jenis sistem rekomendasi untuk menyelesaikan permasalahan dalam merekomendasikan makanan kepada pengguna. Dua pendekatan yang digunakan adalah Content-Based Filtering dan Collaborative Filtering, masing-masing dengan algoritma dan data yang berbeda.

1. Content Based Filtering (CBF)
   
    Pendekatan Content-Based Filtering dilakukan dengan memanfaatkan informasi deskriptif dari makanan, seperti nama, jenis makanan ```C_Type```, dan bahan-bahan makanan (```Describe```). Informasi ini digabungkan ke dalam satu kolom combined, kemudian dilakukan proses vektorisasi menggunakan ***TF-IDF (Term Frequency-Inverse Document Frequency)***. TF-IDF merupakan teknik yang mengukur seberapa penting suatu kata dalam sebuah dokumen relatif terhadap seluruh kumpulan dokumen. Kata-kata yang sering muncul di satu dokumen tetapi jarang muncul di dokumen lain akan memiliki bobot lebih tinggi, sehingga membantu membedakan karakteristik unik dari setiap makanan.

Langka-langkah yang dilakukan: 
- Menggabungkan fitur teks menjadi satu kolom gabungan untuk TF-IDF.
  ```python
   food['combined'] = food['Name'] + ' ' + food['C_Type'] + ' '  + ' ' + food['Describe']
  ```
- Menerapkan TfidfVectorizer untuk menghasilkan representasi vektor dari teks.
  ```python
   tfidf = TfidfVectorizer(stop_words='english')
   tfidf_matrix = tfidf.fit_transform(food['combined'])
  ```
- Menghitung kemiripan antar item menggunakan cosine similarity. Fungsi ```cosine_similarity``` digunakan untuk menghitung cosine similarity antara vektor TF-IDF, yang menghasilkan matriks kemiripan antar semua item makanan.
  ```python
   cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
  ```
- Menggunakan fungsi pencarian berbasis nama makanan untuk mengembalikan 5 rekomendasi teratas berdasarkan kemiripan konten.
   ```python
   def recommend_food(input_name, cosine_sim=cosine_sim):
    # Reset index untuk memastikan urutan index sesuai cosine_sim
    food_reset = food.reset_index()

    # Cari makanan yang mengandung input keyword (case-insensitive)
    matches = food_reset[food_reset['Name'].str.lower().str.contains(input_name.lower())]
    if matches.empty:
        return f"Makanan yang mengandung '{input_name}' tidak ditemukan dalam data."
    idx = matches.index[0]

    # Hitung similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 recommendation

    food_indices = [i[0] for i in sim_scores]

    return food_reset[['Name', 'C_Type', 'Describe']].iloc[food_indices]
  ```
Hasil output rekomendasi dari recommend_food('pasta') mengembalikan daftar makanan yang memiliki konten serupa dengan “pasta”.
```python
   recommend_food('pasta')
  ```


|                  Name                          |  C_Type |                               Describe                                  |                             
|------------------------------------------------|---------|-------------------------------------------------------------------------|
| melted broccoli pasta with capers and anchovies| french  | broccolibread crumbs anchovy fillets garlic capers                      |
| pasta with garlicscape pesto                   | italian | pistachios parmigianoreggiano cheesesalt and basil                      |
| cheese naan                                    | indian  | aall purpose flour yougurt cheese                                       |
| fish with white sauce                          | italian | fillet fish oil milk flour butter salt ground pepper                    |
| cheese chicken kebabs                          | indian  | chicken thais garlic paste garlic paste yellow chili powder             |


Dapat dilihat bahwa model berhasil merekomendasi kan 5 makanan
   
Kelebihan dan Kekurangan model ini adalah: 
- **Kelebihan**: model ini tidak memerlukan interaksi pengguna lain, lalu model ini juga dapat memberikan rekomendasi untuk pengguna baru (cold-start pada user).
- **Kekurangan**: model ini hanya cenderung merekomendasikan makanan yang mirip dengan inputan makanan yang diberikan user. Lalu, model ini juga tidak dapat belajar dari preferensi pengguna lain.
  
   
2. Collaborative Filtering (CF)
   
   Pendekatan kedua adalah Collaborative Filtering menggunakan neural network dengan model embedding. Model ini mempelajari hubungan antara pengguna dan makanan berdasarkan histori rating yang diberikan. Setiap     User_ID dan Food_ID diubah menjadi representasi numerik menggunakan LabelEncoder, lalu dipetakan ke dalam embedding layer.

Langka-langkah yang dilakukan: 
- Encode ```User_ID``` dan ```Food_ID```.
  ```python
  user_encoder = LabelEncoder()
  item_encoder = LabelEncoder()

  rating['user'] = user_encoder.fit_transform(rating['User_ID'])
  rating['item'] = item_encoder.fit_transform(rating['Food_ID'])
  ```
  Hal ini dilakukan karena model hanya dapat memproses input numerik, bukan string.
  
- Split data menjadi data latih dan data uji.
  ```python
   x = rating[['user', 'item']].values
   y = rating['Rating'].values

   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  ```
  Variabel input ```(x)``` didefinisikan sebagai pasangan nilai user dan item, yaitu representasi dari ID pengguna dan ID makanan yang akan digunakan sebagai fitur. Sementara itu, variabel target ```(y)``` 
  didefinisikan sebagai nilai Rating, yaitu rating yang diberikan pengguna terhadap makanan. Selanjutnya, dataset dibagi menjadi dua bagian menggunakan fungsi ```train_test_split```, dengan proporsi 80% data 
  untuk pelatihan (train) dan 20% untuk pengujian (test). Split dataset ini juga disetel dengan ````random_state=42```` agar hasil split data tetap konsisten saat dijalankan ulang.
  
- Membangun neural network dengan dua embedding layer (untuk user dan item). Lalu, menggabungkan representasi embedding dan tambahkan dense layers.
  ```python
   # Membangun model
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    embedding_size = 50

    user_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    user_vec = Flatten()(user_embedding)

    item_input = Input(shape=(1,))
    item_embedding = Embedding(num_items, embedding_size)(item_input)
    item_vec = Flatten()(item_embedding)
    
    # Concatenate + Dense layers
    concatenated = Concatenate()([user_vec, item_vec])
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1)(x)

    model = Model([user_input, item_input], output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  ```
  Model mendefinisikan dua input utama: satu untuk pengguna ```(user_input)``` dan satu untuk item atau produk ```(item_input)```. Masing-masing input ini diubah menjadi vektor berdimensi 50 menggunakan layer Embedding, yang bertujuan untuk menangkap karakteristik pengguna dan item dalam ruang vektor. Vektor hasil embedding kemudian diratakan dengan ```Flatten```, lalu digabungkan menggunakan ```Concatenate``` untuk membentuk satu vektor fitur gabungan. Vektor ini kemudian diproses oleh dua layer ```Dense``` berturut-turut berukuran 128 dan 64 neuron dengan aktivasi ReLU, masing-masing disertai dengan Dropout 0.3 untuk mencegah overfitting. Terakhir, layer ```Dense``` berukuran 1 digunakan untuk menghasilkan output prediksi rating. Model ini dikompilasi menggunakan optimizer Adam, fungsi loss ```mean_squared_error``` untuk melatih model, dan ```mean_absolute_error``` sebagai metrik evaluasi performa.
  
- Melakukan training model menggunakan fungsi loss ```mean_squared_error``` dan ```mean_absolute_error``` (MAE) sebagai metric tambahan.
```python
   # Train model
    model.fit([x_train[:,0], x_train[:,1]], y_train,
          validation_data=([x_test[:,0], x_test[:,1]], y_test),
          epochs=20, batch_size=16)
  ```
Output-nya adalah sebagai berikut.

<p align="center">
  <img src="https://github.com/user-attachments/assets/27776b90-62bd-4d4e-a88a-5403e58e1530" alt="Epoch training" width="600">
</p>

Hasil pelatihan model pada epoch ke-20 menunjukkan bahwa model memiliki performa yang baik pada data training dengan MAE sebesar 1.01, namun terdapat gap cukup besar dibandingkan MAE pada data validasi sebesar 2.83. Hal ini mengindikasikan potensi overfitting, di mana model terlalu menyesuaikan diri dengan data latih sehingga kurang optimal dalam memprediksi data baru. Meskipun begitu, error masih berada dalam skala yang wajar (skala rating 1–10), namun performa dapat ditingkatkan dengan regularisasi atau strategi early stopping.

- Membuat fungsi rekomendasi makanan kepada pengguna tertentu
  ```python
    def recommend_for_user(user_Id, top_n=10):
    user_idx = user_encoder.transform([user_Id])[0]
    all_items = np.arange(num_items)
    user_array = np.full_like(all_items, user_idx)

    predictions = model.predict([user_array, all_items], verbose=0).flatten()
    top_indices = predictions.argsort()[-top_n:][::-1]

    recommended_item_ids = item_encoder.inverse_transform(top_indices)

    return food[food['Food_ID'].isin(recommended_item_ids)][['Food_ID', 'Name', 'C_Type', 'Veg_Non']]
  ```
  Tes model dengan kode di bawah:
  ```python
    recommend_for_user('5')
  ```
  
  Hasil output rekomendasi dari recommend_for_user('5')
  
  | Food_ID |                  Name                    |    C_Type     | Veg_Non |
  |---------|------------------------------------------|---------------|---------|
  | 69      | banana and maple ice lollies             | dessert       | veg     |
  | 94      | chicken sukka                            | indian        | nonveg  |
  | 105     | chicken tenders                          | snack         | nonveg  |
  | 127     | cajun spiced turkey wrapped with bacon   | mexican       | nonveg  |
  | 139     | surmai curry with lobster butter rice    | thai          | veg     |
  | 172     | zucchini methi pulao                     | indian        | veg     |
  | 196     | bread dahi vada                          | indian        | veg     |
  | 273     | corn jalapeno poppers                    | mexican       | veg     |
  | 276     | apple and pear cake                      | healthy food  | veg     |
  | 282     | fruit cube salad                         | healthy food  | veg     |
  
  Dapat dilihat bahwa model berhasil merekomendasi kan 10 makanan pada user dengan userID 5

Kelebihan dan Kekurangan model ini adalah: 
- **Kelebihan**: model ini dapat menangkap pola kompleks dari preferensi pengguna, lalu model ini juga merekomendasi secara lebih personalized karena berdasarkan rating user lain yang mirip.
- **Kekurangan**: model ini tidak bisa memberikan rekomendasi jika pengguna baru belum memberikan rating (cold-start problem), lalu model ini membutuhkan interaksi pengguna yang besar.

Rubrik/Kriteria Tambahan (Opsional): 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada tahap evaluasi, digunakan beberapa metrik evaluasi untuk mengukur performa model rekomendasi yang telah dibangun. Metrik yang digunakan disesuaikan dengan jenis model dan tujuan dari sistem rekomendasi. Untuk model Content-Based Filtering (CBF), digunakan metrik Precision@K, sedangkan untuk model Collaborative Filtering (CF) digunakan metrik Mean Squared Error (MSE) dan Mean Absolute Error (MAE).

1. Evaluasi Content Based Filtering (CBF)
   Metrik evaluasi yang digunakan untuk menilai performa model Content-Based Filtering (CBF) adalah Precision@K. Metrik ini mengukur seberapa relevan rekomendasi yang diberikan oleh sistem terhadap preferensi pengguna, dalam hal ini berdasarkan kesamaan kategori makanan (C_Type) dengan makanan yang dijadikan input referensi. Dalam implementasinya, precision dihitung dengan membandingkan berapa banyak item dari rekomendasi yang termasuk dalam kategori yang sama dengan makanan input. Formula yang digunakan adalah:

   <p align="center">
    <img src="https://github.com/user-attachments/assets/41a479d9-67f5-4497-80b2-112bb092beab" alt="Epoch training" width="600">
   </p>

   Kode di bawah ini akan membuat fungsi untuk menghitung precision:
   ```python
    def precision_at_k(query_food_name, recommendations, food_data):
    matches = food_data[food_data['Name'].str.lower().str.contains(query_food_name.lower())]
    if matches.empty:
        return f"Makanan '{query_food_name}' tidak ditemukan dalam data."

    # Ambil C_Type dari hasil match pertama
    query_type = matches.iloc[0]['C_Type'].lower().strip()

    # Hitung jumlah rekomendasi yang punya C_Type yang sama
    relevant_count = sum(
        rec_type.lower().strip() == query_type
        for rec_type in recommendations['C_Type']
    )

    precision = relevant_count / len(recommendations)
    return precision

  ```

  Lalu, menghitung precision:
  ```python
    recs = recommend_food('kimchi', cosine_sim)
    precision = precision_at_k('kimchi', recs, food)
    print(f'Precision: {precision:.2f}')
  ```

  Output:
  ```
  Precision: 0.80
  ```
Suatu rekomendasi dikatakan relevan jika memiliki nilai C_Type yang sama dengan makanan acuan. Dalam pengujian dengan input makanan "kimchi", dari lima hasil rekomendasi teratas yang dihasilkan oleh sistem, sebanyak empat memiliki kategori yang sama. Sehingga, nilai Precision@5 yang dihasilkan adalah 0.80, atau 80%.
   
2. Evaluasi Collaborative Filtering (CF)
   
  Pada model Collaborative Filtering (CF) ini, metrik evaluasi yang digunakan adalah Mean Squared Error (MSE) dan Mean Absolute Error (MAE). MSE digunakan sebagai indikator utama karena mampu memberikan penalti lebih besar terhadap kesalahan prediksi yang tinggi, sehingga lebih sensitif terhadap outlier. Sementara itu, MAE digunakan sebagai pendukung untuk mengukur rata-rata besar kesalahan absolut antara nilai aktual dan prediksi.


   Metrik MSE dan MAE dihitung dengan rumus:

   <p align="center">
    <img src="https://github.com/user-attachments/assets/537846bb-29bd-4816-96e7-c72b5e939bb0" alt="Formula MAE and MSE" width="600">
   </p>

Pada tahap evaluasi model CF, dilakukannya visualisasi metrik. Kode di bawah ini akan membuat visualisasi metrik:
```python
   plt.plot(history.history['loss'])
   plt.plot(history.history['mae'])
   plt.title('model_metrics')
   plt.ylabel('root_mean_squared_error')
   plt.xlabel('epoch')
   plt.legend(['train', 'test'], loc='upper left')
   plt.show()
  ```

Hasil visualisasi:

  <p align="center">
    <img src="https://github.com/user-attachments/assets/67961471-1890-4331-9e36-980f7981895c" alt="Visualisasi metrik" width="600">
  </p>
Berdasarkan hasil visualisasi grafik pelatihan model collaborative filtering, terlihat bahwa nilai loss (mean squared error) pada data pelatihan dan pengujian menurun secara signifikan selama beberapa epoch pertama, lalu cenderung mendatar setelahnya. Hal ini mengindikasikan bahwa model berhasil belajar dengan baik dan mencapai konvergensi. Selisih antara train loss dan test loss juga tidak terlalu jauh, yang menunjukkan bahwa model tidak mengalami overfitting secara signifikan. Selain itu, nilai MSE pada data pengujian yang konsisten di kisaran rendah menandakan bahwa prediksi model cukup stabil dan dapat diandalkan untuk merekomendasikan item berdasarkan preferensi pengguna. Secara keseluruhan, performa model sudah cukup baik dan stabil untuk digunakan dalam sistem rekomendasi.

## Referensi
[1] NVIDIA, "What Is a Recommendation System?" [Online]. Available: https://www.nvidia.com/en-us/glossary/recommendation-system/. [Accessed: 20-May-2025].
