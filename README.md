# 🏡 House Pricing Prediction Using Neural Networks

This project leverages a PyTorch-based neural network to predict house prices based on tabular data. The model is trained using historical housing data, processed with feature scaling, and evaluated using standard regression metrics. It also includes a ready-to-use prediction pipeline and TensorBoard integration for training visualization.

---

## 📁 Project Structure

```
.
├── Houses_Pricing_NN.ipynb         # Jupyter Notebook for data preprocessing, training, and evaluation
├── data/
│   ├── train.csv                   # Training dataset with features and target values
│   └── test.csv                    # Test dataset (features only, no target)
├── models/
│   ├── scaler.pkl                  # Fitted StandardScaler used for preprocessing input features
│   └── trained_model.pt            # Trained PyTorch model saved after training
├── runs/
│   └── ...                         # TensorBoard logs for training visualization
├── submission.csv                  # Output file with predicted prices for the test dataset
├── requirements.txt                # List of Python dependencies for environment setup
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.11 or newer
- pip package manager

### Recommended Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

---

## 📦 Installing Dependencies

All required packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

If you made changes or added new packages, update the file with:

```bash
pip freeze > requirements.txt
```

---

## 📓 Usage

### Step 1: Launch the Notebook

```bash
jupyter notebook Houses_Pricing_NN.ipynb
```

### Step 2: Run Through the Notebook

- **Data Loading:** Reads training and test datasets.
- **Preprocessing:** Applies scaling and encoding.
- **Model Definition:** Builds a PyTorch feedforward neural network.
- **Training:** Optimizes using MSE loss and Adam optimizer.
- **Evaluation:** Measures performance on a validation split.
- **Export:** Saves the model and scaler, and generates a CSV of predictions.

---

## 🧠 Model Architecture

- **Input Layer:** Matches the number of features after preprocessing.
- **Hidden Layers:** Fully connected layers with ReLU activations.
- **Output Layer:** Single neuron predicting continuous house price.
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Epochs:** Configurable, typically 100+
- **Batch Size:** Usually 32 or 64

---

## 📊 Visualization with TensorBoard

During training, logs are saved under the `runs/` directory. To view them:

```bash
tensorboard --logdir=runs
```

Navigate to `http://localhost:6006` in your browser to inspect training/validation loss curves and metrics.

---

## 📦 Output Files

After running the notebook:

- `models/trained_model.pt`: Trained PyTorch model weights  
- `models/scaler.pkl`: Preprocessing scaler (required for inference)  
- `submission.csv`: Predicted house prices for the test dataset (ready for submission)

---

## ✅ Evaluation Metrics

Although not part of the submission, validation during training can include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics help gauge how well the model generalizes to unseen data.

---

## 🧾 Explanation: Step-by-Step Notebook Walkthrough

Below is a breakdown of what each cell in the `Houses_Pricing_NN.ipynb` notebook does and why it matters:

### **Step 1**: `from pathlib import Path`  
Used for OS-independent file handling.

### **Step 2**: `train_df = pd.read_csv("data/train.csv")`  
Loads the training data.

### **Step 3**: `y = train_df["SalePrice"]`  
Extracts the target variable (house price).

### **Step 4**: `full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)`  
Combines training and test datasets for uniform preprocessing.

### **Step 5**: `missing_data = full_df.isnull().sum()`  
Identifies missing values in the combined dataset.

### **Step 6**: `plt.figure(figsize=(12, 6))`  
Initializes a plot for visualizing data issues like missing values.

### **Step 7**: `numeric_features = full_df.select_dtypes(include=[np.number]).columns`  
Selects numerical columns for scaling.

### **Step 8**: `categorical_features = full_df.select_dtypes(include=[object]).columns`  
Identifies categorical columns for encoding.

### **Step 9**: `# One-hot encode categorical variables`  
Encodes categorical features into numerical format.

### **Step 10**: `X_train = full_df.iloc[: len(y), :]`  
Separates preprocessed training features from combined dataset.

### **Step 11–14**: Data normalization and feature scaling  
Ensures all features are on a similar scale for effective training.

### **Step 15**: Neural network architecture definition  
Defines the PyTorch model: input, hidden, and output layers.

### **Step 16**: Training setup  
Configures loss function and optimizer.

### **Step 17–20**: Model training loop  
Runs training for multiple epochs, calculates loss, and logs to TensorBoard.

### **Step 21**: Save model and scaler  
Persists trained model and scaler for reuse.

### **Step 22**: Make predictions on test data  
Generates predictions on the test set using the trained model.

### **Step 23**: Create submission file  
Exports the predictions to `submission.csv` in the correct format.

---

## 📝 License

This project is provided for learning and demonstration purposes. Feel free to use, modify, and extend as needed.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open an issue or submit a pull request!

---

## 📬 Contact

For any questions or feedback, feel free to reach out or open an issue on the repository.
