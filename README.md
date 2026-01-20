# 🛢️ US DOT Pipeline Accident Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Gradio](https://img.shields.io/badge/Interface-Gradio-yellow)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project analyzes the **US Department of Transportation (DOT) Pipeline Accidents dataset**. 

While initial exploration involved classifying accident causes, the final objective evolved to building a **Regression Model** to predict the severity of an accident. Specifically, the model predicts the **Unintentional Release (in Barrels)** based on factors like pipeline type, location, material failure details, and recovery efforts.

The project includes a full data science pipeline—from data cleaning and complex model comparison to an interactive **Gradio web interface**.

## 📂 Dataset
* **Source:** [Kaggle - US DOT Pipeline Accidents](https://www.kaggle.com/datasets/usdot/pipeline-accidents)
* **Description:** Records of pipeline accidents in the US from 2010 to present.
* **Target Variable:** `Unintentional Release (Barrels)`
* **Key Features Used:**
    * `Pipeline Type` (Above/Underground, Tank)
    * `Liquid Type` (Crude Oil, Biofuel, HVL, etc.)
    * `Location Data` (State, Latitude, Longitude)
    * `Cause Category & Subcategory` (e.g., Corrosion, Internal)
    * `Pipeline Shutdown` (Yes/No)
    * `Liquid Recovery` (Barrels recovered during cleanup)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/himadree-chaudhury/Oil-Pipeline-Accidents-Prediction.git](https://github.com/himadree-chaudhury/Oil-Pipeline-Accidents-Prediction.git)
    cd Oil-Pipeline-Accidents-Prediction
    ```

2.  **Install dependencies:**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

    *Contents of `requirements.txt`:*
    ```text
    pandas
    numpy
    scikit-learn
    gradio
    ```

3.  **Generate the Model:**
    Before running the app, you must train the model and generate the pickle file.
    ```bash
    python train_model.py
    ```
    *This will create a `linear_model.pkl` file in your directory.*

## 🚀 Usage

### Running the Web App
To launch the interactive prediction interface:

```bash
python app.py