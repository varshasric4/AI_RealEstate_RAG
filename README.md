# 🏡 AI Real Estate Advisor (RAG-Based)

An intelligent AI-powered real estate assistant that helps users search, analyze, and get recommendations for properties using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**.

---

## 🚀 Overview

The **AI Real Estate Advisor** is a conversational system designed to:

* Answer real estate-related queries
* Retrieve relevant property data efficiently
* Provide accurate and context-aware responses
* Assist users in making better buying/renting decisions

This project combines:

* 🔍 Semantic Search (FAISS)
* 🤖 Generative AI (LLMs)
* 🧠 Context Retrieval (RAG)

---

## 🧠 Key Features

* 💬 Chat-based AI assistant
* 🔍 Semantic search using embeddings
* 🧠 Context-aware answers via RAG pipeline
* ⚡ Fast document retrieval using FAISS
* 🎯 Query understanding & auto-correction
* 📍 Intelligent property suggestions
* 📊 Structured and meaningful responses

---

## 🏗️ System Architecture

```
User Query
   ↓
Embedding Generation
   ↓
Vector Database (FAISS)
   ↓
Top-K Relevant Documents
   ↓
LLM (OpenAI / Chat Model)
   ↓
Final AI Response
```

---

## 🔄 RAG Pipeline (Working)

1. User inputs a query
2. Query is converted into embeddings
3. FAISS retrieves similar documents
4. Relevant context is passed to LLM
5. LLM generates a final accurate response

👉 This improves accuracy and reduces hallucinations compared to normal LLMs.

---

## 🛠️ Tech Stack

### 👨‍💻 Backend

* Python
* LangChain
* OpenAI API (LLM)

### 📦 Libraries & Tools

* langchain
* langchain-openai
* langchain-community
* HuggingFaceEmbeddings
* FAISS

### 🎨 Frontend

* Gradio (Lightweight UI)

---

## 📂 Project Structure

```
AI_RealEstate_RAG/
│── app.py                  # Main application
│── config.py               # Configuration settings
│── search.py               # Retrieval logic
│── vectorstore/            # FAISS vector database
│── data/                   # Property dataset
│── utils/                  # Helper functions
│── requirements.txt        # Dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/varshasric4/AI_RealEstate_RAG.git
cd AI_RealEstate_RAG
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate      # Windows
```

```bash
source venv/bin/activate   # Mac/Linux
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Setup Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
python app.py
```

After running, open the **Gradio local link** in your browser.

---

## 💡 Example Queries

* Show me 2BHK apartments under 50 lakhs
* Affordable houses in Hyderabad
* Best properties near IT companies
* Luxury villas with swimming pool
* Suggest budget-friendly homes

---

## 📸 Sample Outputs

* Chat-based responses
* Context-aware answers
* Corrected user queries
* Relevant property suggestions

---

## 🎯 Why This Project?

✔ Real-world application of AI in real estate
✔ Demonstrates RAG pipeline (important for interviews)
✔ Combines AI + Retrieval + UI
✔ Strong portfolio project for internships/jobs

---

## 🔮 Future Improvements

* 📍 Map-based property visualization
* 🧠 Personalized recommendations
* 📊 Price prediction model
* 🌐 Real-time property API integration
* 📱 Full-stack deployment

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Submit a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👩‍💻 Author

**Varsha Sri Palakurthi**
🔗 GitHub: https://github.com/varshasric4

---

## ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork it
📢 Share it

---
