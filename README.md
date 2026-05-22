<h1 align="center">⚕️ Health-AI — an AI assistant for doctors during operations</h1>

<p align="center">A school research project about medical AI and patient monitoring 🩺</p>

---

## 👋 Hello!
I am Skoptsov Vasilii. 
This project was created by me for school and city science conferences.  
**Health-AI** is an autonomous system that monitors a patient’s condition during surgery using:

- ECG signals  
- camera video  
- additional sensor data  

The program detects possible complications and gives their probabilities in real time.

---

<h2>📌 System Components</h2>

1. 🫀 **ECG model** — a ResNet1D network for classifying lead-II ECG signals  
2. 📷 **Vision model** — a ResNet50 for detecting complications using the patient’s face  
3. 📈 **Sensor algorithm** — checks complications using external sensor readings  
4. 🔁 **Re-check algorithm** — compares AI outputs with rule-based logic  
5. 🎛️ **Final probability module** — combines all predictions  
6. 🖥️ **Interface program** — the main window with alerts and visualization  

---

<h2>🚀 How to Run</h2>

1. Install **PyCharm** or any Python IDE  
2. Install **Python 3.9**  
3. Install libraries:
pip install -r requirements.txt
4. Clone this repository:
git clone https: ?
cd Health-ai

---

<h2>📂 Main Files</h2>

- **window_final_healtAI.py** — main program with interface  
- **check_verdict_AI.py** — re-check module for AI decisions  
- **best_model.pth** — trained ECG model  
- **ML_ECG_cardiogramma.py** — code for ECG model training  
- **resnet50_classification.py** — code for vision model training  

---

<h2>🩺 What the System Can Detect</h2>

The program can classify **9 complications**:

1. AV block  
2. Fibrillation  
3. Myocardial infarction  
4. Hypoxia  
5. Allergic reaction  
6. Coagulation problems  
7. Shock  
8. Cyanosis  
9. Jugular vein swelling  

---

<h2>🧠 About the Models</h2>

### 🫀 ECG Model (ResNet1D)
- Trained for **50–60 epochs**  
- Dataset: **31,000 lead-II ECG signals**  
- Input:  
- 2,500 ECG points  
- 2,500 derivative points  
- You can test the system using **Arduino ECG sensors**  

### 📷 Vision Model (ResNet50)
- Trained on **5,000 photos**  
- Detects the face using OpenCV  
- Can work with your webcam  

### 🔁 Re-check Algorithm
This module compares:
- the AI predictions  
- the rule-based sensor algorithm  

Then a linear regression gives a **final probability** of the complication.

---

<h2>📄 More Information you can find in my documentation</h2>

👉 Full documentation: health_AI_DOC1.docx.

---

<h2>🏆It is project won in some Conferences</h2>
1) Всероссийская Сеченовская конференция(Сеченово): First place
2) Потенциал(МЭИ): Third place
3) Всероссийская конференция ЮНИОР(НИЯУ МИФИ) - 2nd place
4) Старт в медицину(Сеченово) - 3rd place
5) ЮНИОР(РНИУ Пирогово) - 2nd place
6) Высший пилотаж(НИУ ВШЕ) - 2nd place
7) Спектр(Бауманка) - 1st place
---
<h2>⚠️ Disclaimer</h2>

This program is **not a medical device**.  
It is not intended for diagnosis or commercial use.  
The author is not responsible for incorrect predictions.

---

<h2>💡 Future Plans</h2>

I plan to return to this project in **3–4 years**, when I study at university (around **2029**), and make a big update with more accurate models and better sensors.

---

<p align="center">Thank you for reading! 😊 Skoptsov Vasilii 2025-2026</p>



