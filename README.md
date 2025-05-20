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

     Model Content-Based Filtering menyarankan makanan berdasarkan kemiripan konten antar makanan. Untuk implementasinya, data seperti nama makanan, deskripsi, dan kategori digabung menjadi satu kesatuan teks,   
     kemudian direpresentasikan secara numerik menggunakan teknik TF-IDF (Term Frequency–Inverse Document Frequency). Kemudian, cosine similarity digunakan untuk menghitung kemiripan antar makanan. Berdasarkan 
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
  ```phyton
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

  ```phyton
  # Mengubah type data User_ID, Food_ID, dan Rating
  rating['User_ID'] = rating['User_ID'].astype(int)
  rating['Food_ID'] = rating['Food_ID'].astype(int)
  rating['Rating'] = rating['Rating'].astype(float)
  ```

  Lalu, kode dibawah ini akan melakukan cleaning (pembersihan) data teks pada beberapa kolom di dataset makanan, yaitu ```Name```, ```C_Type```, ```Veg_Non```, dan ```Describe```. Tujuan utamanya adalah   
  menghapus karakter non-alfanumerik (seperti tanda baca dan simbol), serta mengubah semua huruf menjadi huruf kecil. Langkah ini penting agar data lebih konsisten dan tidak terjadi duplikasi yang disebabkan 
  oleh perbedaan format penulisan, sehingga hasil analisis atau pemodelan menjadi lebih akurat.
  ```phyton
  def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())
    return ''

  food['Name'] = food['Name'].apply(clean_text)
  food['C_Type'] = food['C_Type'].apply(clean_text)
  food['Veg_Non'] = food['Veg_Non'].apply(clean_text)
  food['Describe'] = food['Describe'].apply(clean_text)
  ```

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

Rubrik/Kriteria Tambahan (Opsional): 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

Rubrik/Kriteria Tambahan (Opsional): 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

---Ini adalah bagian akhir laporan---

Catatan:
- Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
