Glaucoma Detection

This project implements a machine learning model to detect Glaucoma from retinal images. Glaucoma is a group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in the eye. Early detection through image analysis can help prevent vision loss.

🚀 Features

Automated detection of Glaucoma from retinal images

Preprocessing of images for enhanced model performance

CNN architecture for feature extraction and classification

Visualization of model predictions

🛠️ Installation

Clone the repository:

git clone https://github.com/littlhe/Glaucoma-Detection.git
cd Glaucoma-Detection

Install dependencies:

pip install -r requirements.txt

📊 Dataset

The model is trained on retinal images annotated for Glaucoma detection.

Ensure the dataset is placed in the data/ directory with the following structure:

data/
  train/
  test/
  labels.csv

🚀 Usage

Run the detection script:

python detect_glaucoma.py

The predictions will be saved in the results/ directory.

⚙️ Model Details

Convolutional Neural Network (CNN) architecture

Image preprocessing includes resizing, normalization, and data augmentation

📈 Results

Achieved an accuracy of XX% on the test set

ROC-AUC score: XX

🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

📬 Contact

GitHub: littlhe

Email: your-email@example.com
