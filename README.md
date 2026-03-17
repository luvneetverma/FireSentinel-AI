Wildfire Classification

A Deep Learning–based system for classifying and detecting wildfires using satellite imagery. The project leverages Convolutional Neural Networks (CNNs) to accurately distinguish between wildfire and non-wildfire regions. Designed to assist in early detection and environmental monitoring, this model supports real-time visualization through an interactive Gradio interface.

🚀 Key Features

🧠 Built and tested 100+ CNN models for wildfire classification.

🛰️ Processed over 100,000 satellite images from various sources.

🎯 Achieved 96% detection accuracy across diverse datasets.

🌍 Implemented image preprocessing, augmentation, and segmentation pipelines to enhance model generalization.

💻 Designed an interactive Gradio app for visualization, prediction, and alert generation.

⚙️ Modular design allowing experimentation with multiple architectures (ResNet, Xception, EfficientNet).

🧱 Tech Stack

Programming Language: Python

Deep Learning Frameworks: TensorFlow, Keras

Libraries: OpenCV, NumPy, Matplotlib, Scikit-learn

Tools: Google Colab, Gradio, Hugging Face Spaces

📂 Project Structure
Wildfire-Classification/
│
├── data/                     # Dataset (satellite imagery)
├── notebooks/                # Jupyter notebooks for experiments
├── models/                   # Trained model files
├── src/                      # Python scripts for training & preprocessing
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── app.py                    # Gradio interface script
├── requirements.txt          # Required dependencies
└── README.md                 # Project documentation

⚙️ Installation

Clone the repository

git clone https://github.com/Ruchi1963/Wildfire-Classification.git
cd Wildfire-Classification


Install dependencies

pip install -r requirements.txt


Run the Gradio app

python app.py

📊 Results
Metric	Value
Accuracy	96%
Precision	0.95
Recall	0.96
F1 Score	0.95
💡 Insights

Data augmentation significantly improved model robustness on unseen samples.

Fine-tuning pretrained architectures (ResNet50, Xception) enhanced feature extraction.

GAN-based augmentation improved class balance by 25%.

🌐 Demo

🚀 Try the live model here: Wildfire Classifier on Hugging Face

🏆 Acknowledgements

Satellite datasets sourced from open wildfire monitoring datasets.

TensorFlow and Keras for deep learning framework support.

Hugging Face and Gradio for deployment and visualization.

👨‍💻 Author

Ruchir Raj
📧 rajruchir18@gmail.com

