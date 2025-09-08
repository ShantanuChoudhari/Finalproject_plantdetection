Plant-Disease-Detection/
│
├── dataset/                 # PlantVillage dataset (leaf images)
├── models/                  # Saved trained models
├── app.py                   # Streamlit app file
├── train.py                 # Model training script
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── Plant_Disease_Detection_Report.pdf  # Detailed report



git clone https://github.com/ShantanuChoudhari/plant-disease-detection.git
cd plant-disease-detection
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
python train.py
streamlit run app.py

