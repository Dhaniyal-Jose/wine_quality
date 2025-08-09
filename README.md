Wine Quality Prediction – Feedforward Neural Network
📌 Overview
This project trains a Feedforward Neural Network (FNN) to predict wine quality using the wine_quality.csv dataset. The task is part of Task 1: Train a Deep Learning Model using tabular data without transformers.

The model is built with PyTorch (or TensorFlow, depending on your code), trained on features like acidity, sugar, chlorides, and alcohol content to predict the quality score of wine.

🎯 Task Requirements
Data Type: Tabular (wine_quality.csv)

Goal: Predict wine quality (regression problem)

Model: Feedforward Neural Network (no transformers)

Evaluation Metrics:

Mean Squared Error (MSE)

R² Score

Output: Save the trained model as wine_quality_model.pkl

📂 Project Structure
bash
Copy
Edit
AI_Projects/
│
├── WineQuality/
│   ├── wine_quality.csv            # Dataset
│   ├── train_wine_regression.py    # Training script
│   ├── wine_quality_model.pkl      # Saved trained model
│
├── venv/                           # Virtual environment (optional)
└── README.md                       # Project documentation
🧠 Model Details
Architecture:

Input Layer: Number of neurons = number of features in dataset

Hidden Layers: Fully connected layers with ReLU activation

Output Layer: Single neuron for regression output

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam optimizer

Epochs: Configurable in the script

Batch Size: Configurable in the script

📊 Dataset Information
The wine_quality.csv dataset contains physicochemical attributes of wine and the quality score given by wine tasters.

Example columns:

Feature	Description
fixed acidity	Tartaric acid content
volatile acidity	Acetic acid content
citric acid	Citric acid content
residual sugar	Sugar after fermentation
chlorides	Salt content
free sulfur dioxide	SO₂ in free form
total sulfur dioxide	Total SO₂
density	Wine density
pH	Acidity level
sulphates	Potassium sulphate content
alcohol	Alcohol percentage
quality	Quality score (target)

⚙️ Installation & Setup
1️⃣ Clone the repository (if applicable)

bash
Copy
Edit
git clone <repository_url>
cd AI_Projects
2️⃣ Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the training script

bash
Copy
Edit
python WineQuality/train_wine_regression.py
📈 Training Output
Example output after training:

javascript
Copy
Edit
Mean Squared Error: 0.4714212599197881
R² Score: 0.3404845408986822
Model saved as WineQuality/wine_quality_model.pkl
🗂 Model Saving
After training, the model is saved as:

bash
Copy
Edit
WineQuality/wine_quality_model.pkl
You can later load this model for predictions.

🔮 Next Steps
Implement a prediction script to take new wine data and predict its quality.

Tune hyperparameters for better accuracy.

Try feature scaling and normalization for improved performance.

📜 License
This project is for educational purposes and part of AI project tasks. You may modify and use it as needed.
