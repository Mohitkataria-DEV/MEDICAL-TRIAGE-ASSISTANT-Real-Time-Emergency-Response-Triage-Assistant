from model.triage_model import TriageAssistant
import os

# Check if NDMA.pdf exists
if not os.path.exists('NDMA.pdf'):
    print("ERROR: NDMA.pdf not found in current directory!")
    print(f"Current directory: {os.getcwd()}")
    print("Files in directory:")
    for file in os.listdir('.'):
        print(f"  - {file}")
else:
    print(f"Found NDMA.pdf ({os.path.getsize('NDMA.pdf')} bytes)")
    
    # Initialize assistant (this will extract from PDF)
    assistant = TriageAssistant(pdf_path='NDMA.pdf')
    
    # Test queries
    test_queries = [
        "55-year-old male with crushing chest pain radiating to left arm",
        "Patient with difficulty breathing and chest pressure",
        "Mild fever and cough for 2 days",
        "Unconscious patient found on the ground"
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        print(f"Query: {query}")
        assessment, retrieved, _ = assistant.assess(query)
        print(f"Triage Level: {assessment['triage_level']}")
        print(f"Department: {assessment['department']}")
        print(f"Urgency: {assessment['urgency']}")
        
        # Show what was retrieved from PDF
        if retrieved:
            print(f"\nRetrieved from PDF (score: {retrieved[0]['score']:.2f}):")
            print(f"  {retrieved[0]['text'][:200]}...")