# 🚀 AI Analysis and Visualization App

A Streamlit-based AI-powered application that automates dataset preprocessing by analyzing missing values, duplicates, and summary statistics. It utilizes Gemini 1.5 Flash to provide structured preprocessing recommendations for machine learning and deep learning models.

## 🛠️ Features

✅ Upload and analyze datasets (CSV, Excel, JSON, HTML, Parquet, Feather, Pickle, Stata, SPSS)  
✅ Automated missing value detection and handling suggestions  
✅ Duplicate record identification and resolution  
✅ Data visualization with histograms, count plots, and correlation heatmaps  
✅ AI-powered preprocessing recommendations using Gemini 1.5 Flash  
✅ FAISS-based embedding storage for recommendation tracking  
✅ Downloadable AI recommendations for further analysis  

## 📂 Project Structure

```bash
ml-preprocessing-app/
├── app.py                 # Main Streamlit application
├── .env                   # API key storage (not to be committed)
├── requirements.txt       # List of dependencies
├── README.md              # Documentation
└── .gitignore             # Ignore unnecessary files
```

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/MVenkatsai02/Data-Analysis-and-Visualization-Gen-AI
cd ml-preprocessing-app
```

### 2️⃣ Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 🖥️ Running Locally

### 🔹 Start Streamlit Application

```bash
streamlit run app.py
```

Your UI will be available at:

👉 http://localhost:8501

## 🌍 Deployment

### 4️⃣ Deploy on Streamlit Cloud

1. Push the project to GitHub.
2. Go to Streamlit Cloud → Click "Deploy an App".
3. Select GitHub repo → Set main file path to `app.py`.
4. Deploy and share your app link.

## 📌 Example Usage

1️⃣ Open the Streamlit UI.  
2️⃣ Upload a dataset.  
3️⃣ View missing values, duplicates, and data summary.  
4️⃣ Generate AI-powered preprocessing recommendations.  
5️⃣ Download recommendations for further analysis.  

## 🔧 Troubleshooting

💡 **Issue: API Key Not Found**  
✔️ Ensure `.env` file contains `GOOGLE_API_KEY=your_api_key_here`.  
✔️ Restart the Streamlit app after updating the `.env` file.

💡 **Issue: Missing Dependencies**  
✔️ Run `pip install -r requirements.txt`.  
✔️ Ensure virtual environment is activated.

## 🛡️ License

This project is open-source under the MIT License.

## 📩 Contact & Contributions

🔹 Feel free to fork this repo & contribute!  
🔹 Found a bug? Create an issue on GitHub.  
🔹 Questions? Reach out via email: venkatsaimacha123@gmail.com 

🚀 Built with ❤️ using Streamlit & Gemini AI 🚀

