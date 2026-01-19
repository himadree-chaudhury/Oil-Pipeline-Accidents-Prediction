# 🛢️ US DOT Pipeline Accident Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Gradio](https://img.shields.io/badge/Interface-Gradio-yellow)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project analyzes the **US Department of Transportation (DOT) Pipeline Accidents dataset**. The goal is to investigate historical accident data and build a Machine Learning model that predicts the **primary cause of an accident** (e.g., Corrosion, Excavation Damage, Equipment Failure) based on factors like location, pipeline type, costs, and commodity released.

The project includes a full data science pipeline—from data cleaning to model evaluation—and features an interactive **Gradio web interface** deployed to Hugging Face Spaces.

## 📂 Dataset
* **Source:** [Kaggle - US DOT Pipeline Accidents](https://www.kaggle.com/datasets/usdot/pipeline-accidents)
* **Description:** Records of pipeline accidents in the US from 2010 to present.
* **Key Features Used:**
    * `Pipeline Location` (Onshore/Offshore)
    * `Pipeline Type` (Above/Underground)
    * `Liquid Type` (Crude Oil, Refined Products, etc.)
    * `Net Loss (Barrels)`
    * `All Costs` (Financial impact)
    * `Liquid Ignition` & `Explosion` indicators

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/himadree-chaudhury/Oil-Pipeline-Accidents-Prediction.git
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

3.  **Data Setup:**
    Ensure `database.csv` is placed in the root directory of the project.

## 🚀 Usage

### Running the Web App
To launch the interactive prediction interface:

```bash
python app.py