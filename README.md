---
title: Real Estate Prediction
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# House Sale Price Prediction and Intelligent AI Advisor

A multi-stage real estate intelligence project that combines machine learning price prediction with an agentic AI advisory system. The project utilizes the King County House Sales Dataset and advanced RAG (Retrieval-Augmented Generation) to provide data-driven investment advice.

---

## Live Demo
**Hosted on Hugging Face Spaces:** [Link to Space](https://huggingface.co/spaces/parthrajsingh/real-estate-analysis)

---

## Table of Contents

- [Overview](#overview)
- [Milestone 1: Price Prediction](#milestone-1-price-prediction)
- [Milestone 2: Intelligent AI Advisor](#milestone-2-intelligent-ai-advisor)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Deployment](#deployment)

---

## Overview

This project has evolved from a simple predictive model into a comprehensive decision-support system. It covers:

1. **Machine Learning Pipeline** - End-to-end data preprocessing and model training.
2. **Agentic AI Advisor** - A reasoning system built with LangGraph that analyzes market trends and property features.
3. **RAG Integration** - Retrieval-Augmented Generation using FAISS to ground AI advice in real-world market data and dataset statistics.

---

## Milestone 1: Price Prediction

The foundation of the project is a regression model that predicts house sale prices.
- **Preprocessing**: Median imputation for missing values, label encoding for waterfront views, and one-hot encoding for house conditions.
- **Model Comparison**: Evaluated Linear Regression vs. Random Forest Regression.
- **Result**: Random Forest achieved an R-squared score of **0.89**, significantly outperforming the baseline.

---

## Milestone 2: Intelligent AI Advisor

The latest update transforms the system into an agentic advisor that mimics real-world investor thinking.

### Key Features
- **Sequential Reasoning**: Uses LangGraph to orchestrate a workflow (Intake -> Prediction -> RAG -> Analysis -> Report).
- **Localized RAG**: Incorporates a vector database (FAISS) containing 2024-2025 Seattle market trends and specific insights derived from the King County dataset.
- **Investment Verdicts**: The AI generates explicit "Buy", "Hold", or "Avoid" recommendations based on the "Value Gap" between asking prices and predicted market values.
- **Real Comparable Search**: A custom tool that searches the 21,000+ record dataset to find the top 3 closest historical matches based on location and bedroom count.

---

## Dataset

| Property | Details |
|---|---|
| **File** | Data/houseDataset.csv |
| **Rows** | 21,609 |
| **Columns** | 21 |
| **Target Variable** | Sale Price |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Programming** | Python 3.10+ |
| **ML Engine** | scikit-learn, Pandas, NumPy |
| **AI Orchestration** | LangGraph, LangChain |
| **LLM Provider** | Groq (Llama 3.1) |
| **Vector DB** | FAISS |
| **Frontend** | Streamlit |
| **Deployment** | Docker, Hugging Face Spaces |

---

## Project Structure

```
GenAI_Project/
├── agent/             # LangGraph nodes, state, and tools
├── rag/               # RAG logic and indexed documents
├── Frontend/          # Streamlit application
├── Model/             # Trained joblib models and scalers
├── Data/              # House sales dataset
├── utils/             # Preprocessing and formatting utilities
├── Dockerfile         # Deployment configuration
└── requirements.txt   # Project dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Groq API Key (for the AI Advisor)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd GenAI_Project
   ```

2. **Set up environment**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_actual_key_here
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the advisor**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   streamlit run Frontend/app.py
   ```

---

## Deployment

This project is Dockerized for easy deployment on Cloud platforms or Hugging Face Spaces.

### Hugging Face Deployment
1. Create a new Space with the **Docker** SDK.
2. Link your GitHub repository.
3. Add `GROQ_API_KEY` as a **Secret** in the Space settings.
4. The Space will automatically build and deploy using the provided `Dockerfile`.

---

## License

This project is for educational purposes (Mid-Semester Examination).
