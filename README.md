# *"Rubber Guardian"* Tire Quality Detection and Classification

## Project Description
"Rubber Guardian" is a project created for Final Project in MSIB Startup Campus Track Artificial Intelligence. This product is an innovative solution designed to revolutionize the tire industry by ensuring uncompromised quality through advanced detection and classification technology. This cutting-edge system employs state-of-the-art artificial intelligence and machine learning algorithms to meticulously inspect and categorize tires during the manufacturing process. The primary objective is to enhance quality control, minimize defects, and ultimately improve overall safety and performance

## Contributor
| Full Name | Affiliation | Email | LinkedIn | Role |
| --- | --- | --- | --- | --- |
|Octa Miraz       |Universitas PGRI Yogyakarta         | octamiraz7@gmail.com |[Link](https://www.linkedin.com/in/octa-miraz-b58102145/)| Team Leader      |
|Alya Fauziah     |Institut Teknologi Telkom Purwokerto|alyafauziyah6175@gmail.com |[Link](www.linkedin.com/in/alyafauziyah/)|Team Member      |
|Andi Zulfikar    |Universitas Islam Bandung           |azfikar96@gmail.com|[Link](https://www.linkedin.com/in/andifikar/)                          |Team Member     |
|Hanina Nafisa A. |Universitas Sebelas Maret           |haninafisazka@gmail.com    |[Link](www.linkedin.com/in/haninanafisaazka/)|Team Member     |
|Risnaldy N.I     |UPN "Veteran" Jawa Timur            |risnaldy19@gmail.com       |[Link](www.linkedin.com/in/risnaldynovendra/)|Team Member     |
|Ryanza Aufa Y.   |UPN "Veteran" Jakarta               | ryanzaufay18@gmail.com    |[Link](www.linkedin.com/in/ryanza-aufa-yansa-669b0a221/)|Team Member    |

## Setup
### Prerequisite Packages (Dependencies)
- pandas==2.1.0
- openai==0.28.0
- google-cloud-aiplatform==1.34.0
- google-cloud-bigquery==3.12.0
- ...
- ...

### Environment
| | |
| --- | --- |
| CPU | Example: AMD Athlon™ Silver 3050U Processor |
| GPU | Example: AMD RADEON GRAPHIC |
| ROM | Example: 1TB HDD |
| RAM | Example: 8GB DDR4 |
| OS | Example: WINDOWS 10 home single language |

## Dataset
The dataset that we used is sourced from Kaggle, consisting of two classes: defective and good. There are 1028 images of defective tires and 828 images of good tires in the dataset.

However, the dataset we acquired has varied sizes. Therefore, we conducted preprocessing by resizing the images to 512x512. Additionally, we divided our dataset into training and testing sets. For the training set, we utilized 822 images of defective tires and 622 images of good tires. Meanwhile, for the testing set, we needed 206 images of defective tires and 166 images of good tires. Consequently, the total number of images we used amounts to 1856.

Moreover, we implemented data augmentation techniques to enhance the quality of our dataset. We applied vertical and horizontal flips, 90-degree rotation, and Gaussian blur during the augmentation process.

- Link Kaggle : [Dataset Rubber Guardian](https://bit.ly/Dataset-Rubber-Guardian)

Defective

<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/Defective%20(136).jpg" width="200" height="400">
<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/Defective%20(27).jpg" width="400" height="400">

Good

<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/good%20(351).jpg" width="200" height="400">
<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/good%20(637).jpg" width="400" height="400">

## Results
### Model Performance
We use ResNet50 and InceptionV3 architecture for the base architecture of our model. We do some modifications for get the best results, because when we try the model (without modify) the result is so bad. The accuracy shown is still very weak at around 50% by epoch 30 and shows no signs of increasing accuracy. Therefore we make modifications to the architecture of the base model and hyperparameter.

For the first model (ResNet50) we modify the fully connected layer model. We change fully connected model from Linear layer to Sequential layer so that we can combine the layers that we will add to the Sequential layer. On the Sequential layer we combine Linear layer, ReLU activation and Dropout layer. We adjust the input and output features in Linear layer to fit our dataset. Then, we add ReLU activation so that our model can understand nonlinearity relation between input and output features. And then, we add Dropout layer with the p = 0.5 so that our model can minimize overfit. For the optimizer we use SGD with the value of learning rate 0.001, momentum 0.9, weight_decay 0.003. We also added save checkpoint function to save the base state of our model when testing. The modifications showed good results with the best accuracy is 97-98%, loss 0.03-0.02 and the model can predict the quality of tyre from an image. But, our model is still overfitting although not as severe as before we modified.

The second model is used with Googlenet (Inception) The model utilizes a modified InceptionV3 for binary classification. The last output layer of InceptionV3 ('mixed8') is selected, followed by a Flatten layer. Two Dense layers (32 and 64 units) with ReLU activation and batch normalization are added. The final output is a Dense layer with one unit and sigmoid activation. The model is trained with the Adam optimizer (lr=5e-4) and binary crossentropy loss. EarlyStopping and ModelCheckpoint are employed as callbacks during training. Training is performed with a data generator for a specified number of epochs, and the model with the best performance is saved. As a result, the model can be used for binary classification of images. To mitigate overfitting or underfitting, a dropout layer is added, known to improve performance with an accuracy rate of 91%.

Describe all results found in your final project experiments, including hyperparameters tuning and architecture modification performances. Put it into table format. Please show pictures (of model accuracy, loss, etc.) for more clarity.

#### 1. Metrics
Inform your model validation performances, as follows:
- For classification tasks, use **Precision and Recall**.
- For object detection tasks, use **Precision and Recall**. Additionaly, you may also use **Intersection over Union (IoU)**.
- For image retrieval tasks, use **Precision and Recall**.
- For optical character recognition (OCR) tasks, use **Word Error Rate (WER) and Character Error Rate (CER)**.
- For adversarial-based generative tasks, use **Peak Signal-to-Noise Ratio (PNSR)**. Additionally, for specific GAN tasks,
  - For single-image super resolution (SISR) tasks, use **Structural Similarity Index Measure (SSIM)**.
  - For conditional image-to-image translation tasks (e.g., Pix2Pix), use **Inception Score**.

Feel free to adjust the columns in the table below.

| model | epoch | learning_rate | batch_size | optimizer | val_loss | val_precision class 0 | val_recall class 0 | val_precision class 1 | val_recall class 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet50 | 150 |  0.001 | 32 | SGD | 0.036 | 98% | 93% | 92% | 98% |
| InceptionV3 | 2500 | 0.00001 | 128 | SGD | 0.041 | 90.19% | 87.55% | ... |  ... | 

#### 2. Ablation Study
Any improvements or modifications of your base model, should be summarized in this table. Feel free to adjust the columns in the table below.

| model     | fully connected layer | top1_acc | top5_acc |
| ---       | ---                   | ---      | ---      |
| ResNet50  | Linear(2048, 1024) x1, Linear(1024, 512) x1, Linear(512, 256) x1, Linear(256, 2) x1, ReLU() x3, Dropout(0.5) x2 | 55.37% | 90.86% |
| InceptionV3 | Conv(3x3, 32) x3 | 72.11% | 76.84% |

#### 3. Training/Validation Curve
<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/acuracy.png">
 
### Testing
Testing with 5 defectice tyre images
<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/test_defect.png">

Testing with 5 good tyre images
<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/test_good.png">

### Deployment (Optional)
<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/deployment.png">

Watch [this video](https://drive.google.com/file/d/1e9xFEADsjgzXil1iHT1E0FhGk0OgVwq3/view?usp=sharing) for the detail


Rubber Guardian is a cutting-edge application builth with Streamlit and designed to provide real-time assessments of tire conditions, empowering users to determine whether their vehicle tires are in optimal or compromised states. This intuitive and user-friendly tool utilizes advanced technology to analyze tire conditions accurately, promoting safety and extending the lifespan of tires.

With Rubber Guardian, users can quickly and easily capture images of their vehicle tires using their smartphones or cameras. The application then analyzes these images instantly, providing a clear and concise evaluation of the tire's overall health. Users receive clear visual feedback on the condition of their tires through a simple and intuitive interface. The application categorizes the tire as either 'Good' or 'Poor,' offering users a quick and understandable assessment of their tire health.

## Supporting Documents
### Presentation Deck
Rubber Guardians's [Presentation Deck](https://drive.google.com/file/d/165V4WECq-jD6kIgnRtFFQz5yRRgFxheT/view?usp=sharing) 

### Business Model Canvas
<img src="https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/gambar/BMC.png">

[Detail](https://github.com/corbin2023/Final-Project-Startup-Campus-Rubber-Guardian-/blob/main/BMC%20Rubber%20Guardian.pdf)

#### Problem Statement
Kurangnya kesadaran masyarakat akan kualitas ban mobil dan potensi risiko keselamatan yang diakibatkan oleh ban yang rusak, penyakit ban, atau ketebalan ban yang tidak sesuai standar.

#### Mission Statement
Misi <b> "RuberGuardian" </b>  adalah meningkatkan kesadaran masyarakat terhadap kualitas ban dan keselamatan berkendara, dengan memberikan solusi otomatis untuk mendeteksi ban yang rusak dan meningkatkan kontrol kualitas di industri ban.

#### Key Partners
- Produsen ban
- Distributor ban
- Bengkel dan toko ban
- Perusahaan asuransi kendaraan
- Pusat riset dan pengembangan teknologi otomotif

#### Key Activities
- Pengembangan dan pemeliharaan model binary classification dan object detection
- Pembaharuan berkala mengenai perangkat keras dan perangkat lunak
- Pelatihan mitra bengkel dalam menggunakan teknologi "RuberGuardian"
- Pemantauan dan analisis data untuk meningkatkan akurasi deteksi
- Integrasi teknologi dalam proses kontrol kualitas industri ban
- Kampanye edukasi dan pemasaran untuk meningkatkan kesadaran masyarakat

#### Key Resources
- Tim pengembangan perangkat lunak dan kecerdasan buatan
- perekrutantim ahli teknis untuk pelatihan dan dukungan teknis
- Dataset ban yang representatif dan berkualitas
- Mitra industri untuk mendapatkan data dan dukungan
- Akses ke Google Colab Pro dan teknologi deteksi terkini
- Platform online 

#### Value Proposition
- Deteksi otomatis ban yang rusak untuk meningkatkan keselamatan berkendara
- Peningkatan kesadaran masyarakat akan risiko keselamatan yang terkait dengan kondisi buruk ban
- Meningkatkan proses kontrol kualitas industri ban
- Mengurangi kemungkinan kecelakaan akibat ban yang rusak
- Keamanan dan ketenangan pikiran bagi pemilik kendaraan
- Meningkatkan efisiensi operasional dengan memberikan informasi real time tentang kondisi ban
- Referensi ban berkualitas ke pelanggan

#### Stakeholder (Customers) Relationships
- Hubungan pelanggan yang responsif melalui aplikasi "RuberGuardian"
- Pelatihan dan dukungan teknis bagi mitra bengkel dan toko ban
- Kolaborasi dengan produsen ban untuk peningkatan teknologi

#### Stakeholder (Customers) Segments
- Pemilik kendaraan pribadi
- Perusahaan transportasi dan logistik
- Perusahaan asuransi
- Bengkel dan toko ban
- Perusahaan Ban

#### Channels
- Aplikasi seluler "RuberGuardian" untuk pengguna 
- Pelatihan dan promosi melalui mitra bengkel dan toko ban
- Kerjasama dengan perusahaan asuransi untuk mempromosikan keselamatan berkendara
- Pemasaran digital dan kampanye promosi online
- Partisipasi dalam pameran otomotif dan konferensi industri
- Penjualan perangkat keras dan perangkat lunak
- Dukungan teknis online

#### Cost Structure
- Biaya pengembangan dan pemeliharaan perangkat lunak dan model deteksi
- Biaya pemasaran dan promosi
- Biaya pelatihan mitra bengkel dan toko ban
- Biaya operasional untuk mengelola dataset dan infrastruktur teknologi

#### Revenue Streams
- Biaya langganan bulanan atau tahunan untuk pengguna 
- Biaya lisensi untuk bengkel, toko, ban, perusahaan transportasi dan perusahaan ban.
- Penjualan perangkat deteksi dan perangkat lunak ke perusahaan transportasi 
- Biaya pelatihan mitra

### Short Video
This is our short video that contained background project for [Rubber Guardian](https://drive.google.com/file/d/1hBedE9WZkkGswuJ2WN2ZxyYSEdUPnUk8/view?usp=sharing)

## References
Provide all links that support this final project, i.e., papers, GitHub repositories, websites, etc.
- Link: [https://...](https://www.kaggle.com/code/balmukund/pytorch-tyre-quality-classification-accuracy97-67)
- Link: [https://...](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification?select=Digital+images+of+defective+and+good+condition+tyres)

## Additional Comments
Provide your team's additional comments or final remarks for this project. For example,
1. ...
2. ...
3. ...

## How to Cite
If you find this project useful, we'd grateful if you cite this repository:
```
@article{
...
}
```

## License
For academic and non-commercial use only.

## Acknowledgement
This project entitled <b>"Rubber Guardian" Tire Quality Detection and Classification</b> is supported and funded by Startup Campus Indonesia and Indonesian Ministry of Education and Culture through the "**Kampus Merdeka: Magang dan Studi Independen Bersertifikasi (MSIB)**" program.
