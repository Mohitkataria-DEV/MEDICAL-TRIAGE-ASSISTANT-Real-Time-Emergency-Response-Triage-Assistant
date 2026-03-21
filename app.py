from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import hashlib
import uuid
import traceback

# Add the model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# Import your models
from model.triage_model import TriageAssistant
from model.data_processor import DataProcessor

app = Flask(__name__)
app.secret_key = 'd7df7177bfa6e6a38ec03f63fe64c2f4eefba58c6d6fd19343a1c94b1a0e25cd'

# Initialize models with caching
print("=" * 50)
print("Initializing Medical Triage Assistant...")
print("=" * 50)

try:
    print("\n1. Loading Data Processor...")
    data_processor = DataProcessor(model_path='data_processor.pkl')
    print("   ✓ Data Processor loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading Data Processor: {e}")
    data_processor = None

try:
    print("\n2. Loading Triage Assistant...")
    triage_assistant = TriageAssistant(model_path='triage_assistant.pkl')
    print("   ✓ Triage Assistant loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading Triage Assistant: {e}")
    triage_assistant = None

print("\n" + "=" * 50)
print("Application Ready!")
print("=" * 50)

# Store assessments in memory (use database in production)
assessments_db = {}

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with statistics"""
    stats = {
        'total_assessments': len(assessments_db),
        'avg_confidence': calculate_avg_confidence(),
        'recent_assessments': get_recent_assessments(5),
        'triage_distribution': get_triage_distribution(),
        'models_loaded': {
            'data_processor': data_processor is not None,
            'triage_assistant': triage_assistant is not None
        }
    }
    return render_template('dashboard.html', stats=stats)

@app.route('/assess', methods=['POST'])
def assess():
    """Handle triage assessment request"""
    try:
        data = request.json
        query = data.get('symptoms', '')
        
        if not query:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Get vitals if provided (filter out None values)
        vitals = {}
        vital_fields = ['age', 'heart_rate', 'bp_systolic', 'oxygen', 'temperature', 'pain_level']
        for field in vital_fields:
            value = data.get(field)
            if value is not None and value != '':
                vitals[field] = value
        
        # Check if triage assistant is available
        if triage_assistant is None:
            # Use fallback assessment
            assessment = fallback_assessment(query, vitals)
            retrieved_docs = []
            raw_response = None
        else:
            # Perform assessment with actual model
            assessment, retrieved_docs, raw_response = triage_assistant.assess(
                query, 
                include_vitals=vitals if vitals else None
            )
        
        if assessment:
            # Generate unique ID for this assessment
            assessment_id = str(uuid.uuid4())[:8]
            
            # Store assessment
            assessments_db[assessment_id] = {
                'id': assessment_id,
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'vitals': vitals,
                'assessment': assessment,
                'retrieved_docs': retrieved_docs[:3] if retrieved_docs else [],
                'fallback': triage_assistant is None
            }
            
            return jsonify({
                'success': True,
                'assessment_id': assessment_id,
                'assessment': assessment,
                'retrieved_docs': retrieved_docs[:3] if retrieved_docs else [],
                'fallback_used': triage_assistant is None
            })
        else:
            return jsonify({'error': 'Failed to generate assessment'}), 500
            
    except Exception as e:
        print(f"Error in assess endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/assessment/<assessment_id>')
def view_assessment(assessment_id):
    """View a specific assessment"""
    if assessment_id in assessments_db:
        return render_template('assessment.html', 
                             assessment=assessments_db[assessment_id])
    return render_template('404.html'), 404

@app.route('/api/assessments')
def get_assessments():
    """API endpoint to get all assessments"""
    return jsonify(list(assessments_db.values()))

@app.route('/api/assessments/<assessment_id>')
def get_assessment(assessment_id):
    """API endpoint to get specific assessment"""
    if assessment_id in assessments_db:
        return jsonify(assessments_db[assessment_id])
    return jsonify({'error': 'Assessment not found'}), 404

@app.route('/api/stats')
def get_stats():
    """API endpoint for dashboard stats"""
    stats = {
        'total_assessments': len(assessments_db),
        'avg_confidence': calculate_avg_confidence(),
        'triage_distribution': get_triage_distribution(),
        'recent_trend': get_recent_trend(),
        'models_loaded': {
            'data_processor': data_processor is not None,
            'triage_assistant': triage_assistant is not None
        }
    }
    return jsonify(stats)

@app.route('/api/sample_queries')
def get_sample_queries():
    """Get sample queries from data processor"""
    try:
        if data_processor:
            samples = data_processor.get_sample_queries(5)
            return jsonify({'success': True, 'samples': samples})
        else:
            return jsonify({'success': False, 'error': 'Data processor not available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export/<assessment_id>')
def export_assessment(assessment_id):
    """Export assessment as JSON"""
    if assessment_id in assessments_db:
        return jsonify(assessments_db[assessment_id])
    return jsonify({'error': 'Assessment not found'}), 404

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear assessment history"""
    global assessments_db
    assessments_db = {}
    return jsonify({'success': True, 'message': 'History cleared'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'data_processor': data_processor is not None,
        'triage_assistant': triage_assistant is not None,
        'total_assessments': len(assessments_db),
        'timestamp': datetime.now().isoformat()
    })

# Helper functions
def calculate_avg_confidence():
    if not assessments_db:
        return 0
    confidences = []
    for a in assessments_db.values():
        conf = a['assessment'].get('confidence', 0)
        if conf:
            confidences.append(conf)
    return round(sum(confidences) / len(confidences), 2) if confidences else 0

def get_recent_assessments(n):
    sorted_assessments = sorted(
        assessments_db.values(), 
        key=lambda x: x['timestamp'], 
        reverse=True
    )
    return sorted_assessments[:n]

def get_triage_distribution():
    distribution = {'0': 0, '1': 0, '2': 0, '3': 0}
    for a in assessments_db.values():
        level = str(a['assessment'].get('triage_level', '0'))
        distribution[level] = distribution.get(level, 0) + 1
    return distribution

def get_recent_trend():
    recent = get_recent_assessments(10)
    return [{'date': a['timestamp'][:10], 'level': a['assessment']['triage_level']} 
            for a in recent]

def fallback_assessment(query, vitals=None):
    """Fallback assessment if model fails"""
    query_lower = query.lower()
    
    # Simple keyword-based triage
    if any(word in query_lower for word in ['unconscious', 'not breathing', 'cardiac', 'severe bleeding', 'heart attack']):
        level = '3'
        priority = 'High'
        urgency = 'Immediate'
        dept = 'Emergency'
        confidence = 85
    elif any(word in query_lower for word in ['chest pain', 'difficulty breathing', 'stroke', 'severe pain']):
        level = '2'
        priority = 'High'
        urgency = 'Very Urgent'
        dept = 'Emergency'
        confidence = 80
    elif any(word in query_lower for word in ['fever', 'broken', 'fracture', 'moderate pain', 'vomiting']):
        level = '1'
        priority = 'Medium'
        urgency = 'Urgent'
        dept = 'General Ward'
        confidence = 70
    else:
        level = '0'
        priority = 'Low'
        urgency = 'Routine'
        dept = 'OPD'
        confidence = 65
    
    return {
        'triage_level': level,
        'confidence': confidence,
        'priority': priority,
        'immediate_actions': ['Assess patient', 'Check vital signs', 'Document symptoms'],
        'department': dept,
        'urgency': urgency,
        'reasoning': f'Fallback assessment based on symptom analysis: {query[:100]}...',
        'risk_factors': 'Based on reported symptoms',
        'follow_up': 'Follow up as per department protocol',
        'timestamp': datetime.now().isoformat()
    }

if __name__ == '__main__':
    # Run the app
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("Dashboard: http://localhost:5000/dashboard")
    print("Health check: http://localhost:5000/health")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(debug=False, host='0.0.0.0', port=5000)