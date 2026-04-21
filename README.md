# 🧠 MedEye: MRI-Based Brain Tumor Classification with Deep Learning & Grad-CAM

## 🚀 Overview

MedEye is an AI-powered diagnostic system designed to detect brain tumors from MRI scans using deep learning models. It integrates **CNN-based classification** with **Grad-CAM visualization** to provide explainable results, along with an **AI chatbot** for medical assistance.

---

## ✨ Features

* 🧠 Brain tumor classification using CNN (MobileNetV2, ResNet, etc.)
* 🔍 Grad-CAM visualization for model explainability
* 💬 AI-powered medical chatbot (Gemini API)
* 🖥️ Interactive UI built with Streamlit
* 📊 Performance visualization and results analysis

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **Deep Learning:** TensorFlow, Keras
* **Visualization:** Grad-CAM
* **AI Chatbot:** Gemini API
* **Others:** NumPy, OpenCV, Matplotlib

---

## 📁 Project Structure

```
MedEye-Project/
├── app.py
├── main.py
├── chatbot.py
├── chatbot_ui.py
├── Training.py
├── requirements.txt
├── README.md
├── Notebooks/
├── Results/
```

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Add API Key

Create a `.env` file:

```
API_KEY=your_api_key_here
```

### 4. Run the application

```
python -m streamlit run main.py 
```

---

## 📊 Model Details

* Pretrained CNN architectures used:

  * MobileNetV2
  * ResNet50
  * InceptionV3
  * Xception
* Best performing model selected based on validation accuracy
* Grad-CAM used to highlight tumor regions in MRI images

---

## 📷 Results

* Model accuracy and loss graphs available in `Results/`
* Grad-CAM heatmaps for interpretability
* Sample MRI predictions included

---

## ⚠️ Disclaimer

* This project is for **educational and research purposes only**
* It does **not provide medical diagnosis**
* Always consult a medical professional for real-world decisions

---

## 📌 Future Improvements

* Deploy model on cloud (AWS / Azure)
* Improve model accuracy with larger dataset
* Add real-time MRI scan integration
* Enhance chatbot with medical knowledge base

---

## 👩‍💻 Author

**Priya Panchal and Yashika Mehra**

B.E. ICT Student

