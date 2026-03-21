# test_models.py
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

# Import your classes
from data_processor import DataProcessor
from triage_model import TriageAssistant

def main():
    print("=" * 50)
    print("Testing DataProcessor...")
    print("=" * 50)
    
    # Test DataProcessor
    try:
        processor = DataProcessor(model_path='data_processor.pkl')
        stats = processor.get_statistics()
        print(f"\nStatistics:")
        print(f"Total patients: {stats['total_patients']}")
        print(f"Triage distribution: {stats['triage_distribution']}")
        
        # Get sample queries
        samples = processor.get_sample_queries(2)
        print(f"\nSample queries:")
        for i, sample in enumerate(samples):
            print(f"{i+1}. {sample['query']}")
            print(f"   Actual triage: {sample['actual_triage']}")
            
    except Exception as e:
        print(f"Error with DataProcessor: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Testing TriageAssistant...")
    print("=" * 50)
    
    # Test TriageAssistant
    try:
        assistant = TriageAssistant(model_path='triage_assistant.pkl')
        
        # Test with a sample query
        test_query = "Patient has chest pain and difficulty breathing"
        print(f"\nTesting query: {test_query}")
        
        assessment, retrieved, response = assistant.assess(test_query)
        
        if assessment:
            print(f"\nAssessment result:")
            print(f"Triage Level: {assessment['triage_level']}")
            print(f"Confidence: {assessment['confidence']}%")
            print(f"Priority: {assessment['priority']}")
            print(f"Department: {assessment['department']}")
            print(f"Urgency: {assessment['urgency']}")
            print(f"\nImmediate Actions:")
            for action in assessment['immediate_actions']:
                print(f"  • {action}")
            
    except Exception as e:
        print(f"Error with TriageAssistant: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()