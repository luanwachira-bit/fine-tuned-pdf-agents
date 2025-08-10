import os
import time

# Import all the agent classes you've created
from agents.ml_agent import MachineLearningAgent
from agents.ds_agent import DataScienceAgent
from agents.cybersec_agent import CybersecurityAgent

def main():
    """
    Main function to dynamically run agents on all documents in the data directory.
    """
    print("🤖 AI Document Intelligence System Initializing...")
    print("="*50)

    data_directory = "data/"
    
    # Check if the data directory exists
    if not os.path.isdir(data_directory):
        print(f"❌ Error: Data directory '{data_directory}' not found.")
        print("Please make sure you have a 'data' folder with your PDFs inside.")
        return

    # A simple mapping to select the right agent and question based on filename keywords
    agent_mapping = {
        "machine learning": (MachineLearningAgent, "What are the primary machine learning concepts discussed?"),
        "deep learning": (MachineLearningAgent, "What is the key deep learning architecture or algorithm described?"),
        "data science": (DataScienceAgent, "What is the main topic of data analysis or data processing covered?"),
        "designing data": (DataScienceAgent, "What are the key principles for designing data-intensive applications mentioned?"),
        "cybersec": (CybersecurityAgent, "What is the primary security threat or defense mechanism identified?"),
    }
    
    # Using a more powerful, fine-tuned model for better results
    # 'deepset/deberta-v3-large-squad2' is a strong choice
    default_model = 'deepset/deberta-v3-large-squad2'

    # Get all files from the data directory
    pdf_files = [f for f in os.listdir(data_directory) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the 'data' directory. Nothing to process.")
        return

    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"\n📄 Processing Document: {pdf_file}")
        
        selected_agent_class = None
        question = "What is the overall purpose of this document?" # A default question
        
        # Find the right agent for the current file
        for keyword, (agent_class, specific_question) in agent_mapping.items():
            if keyword in pdf_file.lower():
                selected_agent_class = agent_class
                question = specific_question
                break
        
        if selected_agent_class:
            # 1. Initialize the correct agent
            print(f"✅ Agent Selected: {selected_agent_class.__name__}")
            agent = selected_agent_class(model_name=default_model)
            
            # 2. Load the document
            pdf_path = os.path.join(data_directory, pdf_file)
            document_context = agent.load_document(pdf_path)

            if document_context:
                # 3. Query the agent with the specific question
                print(f"\n❓ Asking: {question}")
                start_time = time.time()
                answer = agent.query(context=document_context, question=question)
                end_time = time.time()
                
                # 4. Display the answer
                print("\n✅ Agent's Answer:")
                print(f"> {answer}")
                print(f"(Query took {end_time - start_time:.2f} seconds)")
            else:
                print("❌ Failed to load document, exiting.")
        else:
            print(f"⚠️ No specific agent found for '{pdf_file}'. Skipping.")
        
        print("-" * 50)

if __name__ == "__main__":
    main()