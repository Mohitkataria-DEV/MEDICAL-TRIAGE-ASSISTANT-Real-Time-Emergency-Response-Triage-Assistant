import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pickle
import os
import re
import warnings
from typing import List, Dict
import pdfplumber  # For PDF extraction

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message="Passing `generation_config` together with generation-related arguments")

class TriageAssistant:
    def __init__(self, model_path=None, load_cached=True, pdf_path='NDMA.pdf'):
        """Initialize the triage assistant with caching support and PDF extraction"""
        
        self.model_path = model_path or 'triage_assistant.pkl'
        self.pdf_path = pdf_path
        
        # Try to load cached model if requested
        if load_cached and os.path.exists(self.model_path):
            print(f"Loading cached model from {self.model_path}...")
            self._load_from_cache()
        else:
            print("Initializing fresh model...")
            self._initialize_fresh()
            # Cache the model if path is provided
            if model_path or load_cached:
                self.save_model()
        
        print("Triage Assistant initialized successfully!")
    
    def _initialize_fresh(self):
        """Initialize all components from scratch"""
        # Load embedding model
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index and chunks
        self.index = None
        self.chunks = []
        
        # Load SOP documents from PDF
        print("Loading SOP documents from PDF...")
        self._load_sop_from_pdf()
        
        # Load LLM
        print("Loading language model...")
        self._load_llm()
        
        print(f"Loaded {len(self.chunks)} document chunks from PDF")
    
    def _load_from_cache(self):
        """Load the entire model from cache file"""
        try:
            with open(self.model_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Restore all attributes
            self.embed_model = cached_data['embed_model']
            self.index = cached_data['index']
            self.chunks = cached_data['chunks']
            self.llm = cached_data['llm']
            self.tokenizer = cached_data['tokenizer']
            self.generator = cached_data['generator']
            self.pdf_path = cached_data.get('pdf_path', 'NDMA.pdf')
            
            print(f"Model loaded successfully from cache! ({len(self.chunks)} chunks)")
            
        except Exception as e:
            print(f"Error loading from cache: {e}")
            print("Falling back to fresh initialization...")
            self._initialize_fresh()
    
    def save_model(self):
        """Save the model to disk with disk space check"""
        try:
            # Check disk space
            import shutil
            total, used, free = shutil.disk_usage(".")
            if free < 100 * 1024 * 1024:  # Less than 100MB free
                print(f"Warning: Low disk space ({free / (1024**3):.1f} GB free). Skipping save.")
                return
            
            cached_data = {
                'embed_model': self.embed_model,
                'index': self.index,
                'chunks': self.chunks,
                'llm': self.llm,
                'tokenizer': self.tokenizer,
                'generator': self.generator,
                'pdf_path': self.pdf_path
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            print(f"Model saved successfully to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            print(f"Extracting text from {pdf_path}...")
            
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {i+1} ---\n"
                        text += page_text
                    print(f"  Processed page {i+1}/{len(pdf.pages)}")
            
            if not text.strip():
                print("Warning: No text extracted from PDF. Using fallback content.")
                return None
                
            print(f"Extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return None
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunk = ' '.join(chunk_words)
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} text chunks")
        return chunks
    
    def _load_sop_from_pdf(self):
        """Load and index SOP documents from PDF with emergency protocols"""
        
        # Try to extract from PDF first
        pdf_text = self._extract_text_from_pdf(self.pdf_path)
        
        if pdf_text:
            # Split into chunks
            self.chunks = self._chunk_text(pdf_text)
            
            # Add emergency protocols as additional chunks
            emergency_chunks = self._get_emergency_protocols()
            self.chunks.extend(emergency_chunks)
            
        else:
            # Fallback to hardcoded content if PDF extraction fails
            print("Using fallback SOP content (PDF extraction failed)")
            self.chunks = self._get_fallback_sop()
        
        # Create embeddings for all chunks
        print("Creating embeddings for document chunks...")
        embeddings = self.embed_model.encode(self.chunks, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        print(f"FAISS index created with {len(self.chunks)} chunks")
    
    def _get_emergency_protocols(self) -> List[str]:
        """Get emergency-specific protocols"""
        return [
            """CARDIAC EMERGENCY PROTOCOL - LEVEL 3 (IMMEDIATE):
            Heart attack/Myocardial Infarction symptoms:
            - Chest pain/pressure/squeezing sensation
            - Pain radiating to left arm, jaw, back, or stomach
            - Shortness of breath with chest discomfort
            - Cold sweat, nausea, lightheadedness
            - ANY suspicion of heart attack = EMERGENCY LEVEL 3
            Time to reperfusion directly impacts survival.
            ACTIONS: Immediate ECG, oxygen, aspirin, nitroglycerin, activate cath lab.""",
            
            """RESPIRATORY EMERGENCY PROTOCOL - LEVEL 3 (IMMEDIATE):
            Severe respiratory distress symptoms:
            - Inability to speak full sentences
            - Use of accessory muscles
            - Gasping or agonal breathing
            - Cyanosis (blue lips/fingers)
            - Oxygen saturation < 90%
            ACTIONS: High-flow oxygen, positioning, prepare for ventilation if needed.""",
            
            """STROKE PROTOCOL - LEVEL 3 (IMMEDIATE):
            Stroke symptoms (FAST assessment):
            - Facial drooping
            - Arm weakness
            - Speech difficulty
            - Time to call emergency
            ACTIONS: Immediate CT scan, neurological assessment, time window for thrombolytics is critical."""
        ]
    
    def _get_fallback_sop(self) -> List[str]:
        """Fallback SOP content if PDF extraction fails"""
        return [
            """Day One: Zero Hour - Control room sends message to everyone on List 1. 
            Officers assemble within 45 minutes. NCMC/NDMA establishes immediate contact 
            with all stakeholders.""",
            
            """First Response Team responsibilities: Report to State/District disaster 
            management setup. Establish basic office at disaster site. Attempt restoration 
            of power, communication, and water supply. Assess ground zero situation.""",
            
            """Triage Level 3 (Emergency): Immediate life threat, requires immediate 
            medical attention. Examples: Cardiac arrest, severe trauma, unconsciousness,
            heart attack symptoms (chest pain radiating to arm), severe breathing difficulty.""",
            
            """Triage Level 2 (Urgent): Serious condition but stable for short period. 
            Examples: Chest pain, severe bleeding, breathing difficulty.""",
            
            """Triage Level 1 (Less Urgent): Stable condition requiring medical attention 
            but can wait. Examples: Minor fractures, moderate pain, fever.""",
            
            """Triage Level 0 (Non-urgent): Minor conditions that can be treated routinely. 
            Examples: Minor cuts, mild symptoms, routine checkups."""
        ]
    
    def _load_llm(self):
        """Load the language model with proper generation config"""
        try:
            model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create generation config properly
            generation_config = GenerationConfig(
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.llm,
                tokenizer=self.tokenizer,
                generation_config=generation_config
            )
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.generator = None
    
    def retrieve_relevant_chunks(self, query, k=5):
        """Retrieve relevant document chunks"""
        if self.index is None:
            return []
            
        query_embedding = self.embed_model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                similarity = 1 / (1 + distances[0][i])
                retrieved.append({
                    'text': self.chunks[idx],
                    'score': float(similarity),
                    'index': int(idx)
                })
        
        return retrieved
    
    def create_prompt(self, query, retrieved_chunks, vitals=None):
        """Create structured prompt for assessment using retrieved PDF content"""
        
        context = "\n\n".join([f"[SOP {i+1}]: {chunk['text'][:500]}" 
                              for i, chunk in enumerate(retrieved_chunks[:3])])
        
        vitals_text = ""
        if vitals and any(vitals.values()):
            vitals_text = "PATIENT VITALS:\n"
            for key, value in vitals.items():
                if value:
                    vitals_text += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        prompt = f"""You are an expert emergency medical triage assistant. Based on the NDMA Standard Operating Procedures (SOPs) and medical guidelines below, provide a structured triage assessment.

STANDARD OPERATING PROCEDURES (from NDMA):
{context}

{vitals_text}
PATIENT SYMPTOMS: {query}

Based on the NDMA guidelines above, provide a complete triage assessment in the following EXACT format:

TRIAGE LEVEL: [0, 1, 2, or 3 - where 3 is highest emergency]
CONFIDENCE: [0-100]
PRIORITY: [High/Medium/Low]

IMMEDIATE ACTIONS:
• [First action based on NDMA protocol]
• [Second action] 
• [Third action]

RESPONSIBLE DEPARTMENT: [Department name]
URGENCY: [Immediate/Very Urgent/Urgent/Routine]

REASONING:
[Brief explanation based on NDMA guidelines]

RISK FACTORS:
[List key risk factors]

FOLLOW-UP:
[Recommended timeline]

ASSESSMENT:
"""
        return prompt
    
    def parse_response(self, response_text):
        """Parse the LLM response into structured format"""
        
        assessment = {
            'triage_level': '0',
            'confidence': 75,
            'priority': 'Medium',
            'immediate_actions': [],
            'department': 'General Ward',
            'urgency': 'Routine',
            'reasoning': '',
            'risk_factors': '',
            'follow_up': '',
            'timestamp': None
        }
        
        try:
            # Extract triage level
            level_match = re.search(r'TRIAGE LEVEL:\s*(\d+)', response_text)
            if level_match:
                level = level_match.group(1)
                assessment['triage_level'] = level if level in ['0','1','2','3'] else '1'
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response_text)
            if conf_match:
                assessment['confidence'] = min(100, int(conf_match.group(1)))
            
            # Extract priority
            priority_match = re.search(r'PRIORITY:\s*(\w+)', response_text)
            if priority_match:
                assessment['priority'] = priority_match.group(1)
            
            # Extract actions
            actions = re.findall(r'•\s*(.*?)(?=\n•|\n\n|\Z)', response_text, re.DOTALL)
            assessment['immediate_actions'] = [a.strip() for a in actions if a.strip()][:3]
            
            # Extract department
            dept_match = re.search(r'RESPONSIBLE DEPARTMENT:\s*(.*?)(?=\n)', response_text)
            if dept_match:
                assessment['department'] = dept_match.group(1).strip()
            
            # Extract urgency
            urgency_match = re.search(r'URGENCY:\s*(.*?)(?=\n)', response_text)
            if urgency_match:
                assessment['urgency'] = urgency_match.group(1).strip()
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=\n\n|\Z)', response_text, re.DOTALL)
            if reasoning_match:
                assessment['reasoning'] = reasoning_match.group(1).strip()
            
            # Extract risk factors
            risk_match = re.search(r'RISK FACTORS:\s*(.*?)(?=\n\n|\Z)', response_text, re.DOTALL)
            if risk_match:
                assessment['risk_factors'] = risk_match.group(1).strip()
            
            # Extract follow-up
            follow_match = re.search(r'FOLLOW-UP:\s*(.*?)(?=\n\n|\Z)', response_text, re.DOTALL)
            if follow_match:
                assessment['follow_up'] = follow_match.group(1).strip()
            
        except Exception as e:
            print(f"Error parsing response: {e}")
        
        return assessment
    
    def assess(self, query, include_vitals=None):
        """
        Main assessment function with emergency pre-check
        """
        
        # EMERGENCY PRE-CHECK
        query_lower = query.lower()
        emergency_phrases = [
            'heart attack', 'chest pain', 'chest pressure', 'crushing chest',
            'not breathing', 'can\'t breathe', 'unconscious', 'cardiac arrest',
            'severe bleeding', 'stroke', 'radiating pain', 'myocardial'
        ]
        
        is_emergency = any(phrase in query_lower for phrase in emergency_phrases)
        
        if is_emergency:
            print("⚠️ EMERGENCY DETECTED! Returning immediate Level 3 triage.")
            emergency_assessment = {
                'triage_level': '3',
                'confidence': 95,
                'priority': 'CRITICAL',
                'immediate_actions': [
                    'ACTIVATE EMERGENCY RESPONSE TEAM',
                    'IMMEDIATE transport to Emergency Department',
                    'Administer oxygen if available',
                    'Start cardiac monitoring',
                    'Prepare for potential resuscitation'
                ],
                'department': 'EMERGENCY - STAT',
                'urgency': 'IMMEDIATE - LIFE THREATENING',
                'reasoning': f'EMERGENCY: Critical symptoms detected - {query[:100]}',
                'risk_factors': 'High risk - requires immediate intervention',
                'follow_up': 'IMMEDIATE - Do not delay. Call emergency services if not already in hospital.'
            }
            return emergency_assessment, [], "Emergency override"
        
        # Retrieve relevant documents from PDF
        retrieved = self.retrieve_relevant_chunks(query)

        if not retrieved:
            print("No relevant chunks retrieved, using fallback")
            return self.fallback_assessment(query), [], None

        # Create prompt
        prompt = self.create_prompt(query, retrieved, include_vitals)

        # Generate response
        if self.generator:
            try:
                response = self.generator(prompt)
                full_response = response[0]['generated_text']
                new_text = full_response[len(prompt):].strip()
                assessment = self.parse_response(new_text)
                
                # Double-check parsed assessment for emergencies
                if assessment.get('triage_level') == '0' and is_emergency:
                    print("⚠️ Parser returned Level 0 but emergency detected - overriding!")
                    assessment['triage_level'] = '3'
                    assessment['priority'] = 'CRITICAL'
                    assessment['urgency'] = 'IMMEDIATE'
                    assessment['department'] = 'EMERGENCY'
                    assessment['confidence'] = 95
                
                return assessment, retrieved, new_text
                
            except Exception as e:
                print(f"Error generating response: {e}")
                return self.fallback_assessment(query), retrieved, None
        else:
            return self.fallback_assessment(query), retrieved, None
    
    def fallback_assessment(self, query):
        """Provide fallback assessment if LLM fails"""
        query_lower = query.lower()
        
        # Emergency detection
        emergency_keywords = [
            'heart attack', 'chest pain', 'chest pressure', 'crushing chest',
            'not breathing', 'unconscious', 'cardiac arrest', 'severe bleeding',
            'stroke', 'radiating pain', 'difficulty breathing', 'gasping'
        ]
        
        is_emergency = any(keyword in query_lower for keyword in emergency_keywords)
        
        if is_emergency:
            return {
                'triage_level': '3',
                'confidence': 90,
                'priority': 'CRITICAL',
                'immediate_actions': [
                    'ACTIVATE EMERGENCY RESPONSE',
                    'IMMEDIATE transport to ER',
                    'Administer oxygen',
                    'Cardiac monitoring'
                ],
                'department': 'EMERGENCY',
                'urgency': 'IMMEDIATE',
                'reasoning': f'EMERGENCY: Critical symptoms - {query[:100]}',
                'risk_factors': 'Life-threatening condition',
                'follow_up': 'Immediate intervention required'
            }
        elif any(word in query_lower for word in ['chest discomfort', 'breathing difficulty', 'severe pain']):
            level = '2'
            priority = 'High'
            urgency = 'Very Urgent'
            dept = 'Emergency'
        elif any(word in query_lower for word in ['fever', 'broken', 'fracture', 'moderate pain']):
            level = '1'
            priority = 'Medium'
            urgency = 'Urgent'
            dept = 'General Ward'
        else:
            level = '0'
            priority = 'Low'
            urgency = 'Routine'
            dept = 'OPD'
        
        return {
            'triage_level': level,
            'confidence': 70,
            'priority': priority,
            'immediate_actions': ['Assess patient', 'Check vital signs', 'Document symptoms'],
            'department': dept,
            'urgency': urgency,
            'reasoning': f'Based on keyword analysis: {query[:100]}',
            'risk_factors': 'Based on reported symptoms',
            'follow_up': 'As per department guidelines'
        }


# Test function
if __name__ == "__main__":
    print("Testing TriageAssistant with PDF extraction...")
    
    # Initialize with your NDMA.pdf
    assistant = TriageAssistant(pdf_path='NDMA.pdf')
    
    # Test with cardiac symptoms
    test_query = "55-year-old male with crushing chest pain radiating to left arm"
    print(f"\nTesting: {test_query}")
    assessment, _, _ = assistant.assess(test_query)
    print(f"Triage Level: {assessment['triage_level']}")
    print(f"Department: {assessment['department']}")
    print(f"Urgency: {assessment['urgency']}")