# MEDICAL-TRIAGE-ASSISSTANT-Real-Time-Emergency-Response-Triage-Assistant
Medical Triage Assistant - An intelligent RAG-based system that analyses patient symptoms against NDMA guidelines to provide immediate triage assessments (Level 0-3). Features FAISS vector search, LLM-powered analysis, and an interactive dashboard.

# 🏥 Medical Triage Assistant - RAG-Based Emergency Response System

An intelligent medical triage system that uses **Retrieval-Augmented Generation (RAG)** to analyze patient symptoms against NDMA (National Disaster Management Authority) guidelines and provide immediate triage assessments.

## 📋 Overview

This system helps healthcare professionals and first responders make rapid, informed triage decisions by:
- Extracting and understanding NDMA emergency protocols
- Analyzing patient symptoms and vitals
- Providing structured triage assessments (Level 0-3)
- Offering immediate action recommendations
- Tracking assessment history with analytics dashboard

## ✨ Features

### Core Capabilities
- **📄 PDF Document Ingestion**: Automatically extracts and indexes NDMA guidelines
- **🔍 Semantic Search**: FAISS-based vector search for relevant protocols
- **🤖 LLM-Powered Assessment**: DialoGPT-based analysis with structured output
- **⚠️ Emergency Detection**: Immediate Level 3 triage for critical symptoms
- **💾 Model Caching**: Fast loading with pickle-based persistence

### Web Interface
- **🏠 Assessment Portal**: Easy symptom input and vitals recording
- **📊 Analytics Dashboard**: Real-time statistics and triage distribution
- **📈 Visual Charts**: Interactive charts for assessment trends
- **🔄 Assessment History**: Track and review past triage decisions

### Triage Levels
| Level | Category | Response Time | Examples |
|-------|----------|---------------|----------|
| 3 | Emergency | Immediate | Heart attack, cardiac arrest, severe trauma |
| 2 | Urgent | 15-30 minutes | Chest pain, severe bleeding, breathing difficulty |
| 1 | Less Urgent | 1-2 hours | Minor fractures, moderate pain, fever |
| 0 | Non-urgent | Routine | Minor cuts, mild symptoms, checkups |


## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Framework** | Flask (Python) |
| **Vector Search** | FAISS |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **LLM** | Microsoft DialoGPT-medium |
| **PDF Processing** | pdfplumber |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Chart.js, Bootstrap |
| **Serialization** | Pickle |


