#!/usr/bin/env python3
"""
Customer Support Triage System CLI
Multi-agent system using LangGraph for intelligent customer support automation
"""

import os
import sys
from dotenv import load_dotenv
from langgraph_triage import CustomerSupportTriageSystem

def main():
    """Main CLI interface for the customer support triage system."""
    
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Please set your OPENAI_API_KEY in a .env file")
        print("Copy .env.example to .env and add your OpenAI API key")
        sys.exit(1)
    
    print("🤖 Customer Support Triage System")
    print("=" * 50)
    print("This system will:")
    print("1. Classify your query")
    print("2. Search relevant knowledge base")
    print("3. Generate an appropriate response")
    print("4. Determine if human escalation is needed")
    print("\nType 'quit' to exit, 'test' to run test queries")
    print("=" * 50)
    
    # Initialize the system
    try:
        print("\n🔧 Initializing triage system...")
        system = CustomerSupportTriageSystem()
        print("✅ System ready!")
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")
        sys.exit(1)
    
    while True:
        try:
            print("\n" + "-" * 40)
            user_input = input("\n💬 Enter your customer query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Customer Support Triage System!")
                break
            
            if user_input.lower() == 'test':
                run_test_queries(system)
                continue
            
            if not user_input:
                print("Please enter a valid query.")
                continue
            
            print(f"\n🔄 Processing query: '{user_input}'")
            print("⏳ Please wait...")
            
            # Process the query
            result = system.process_query(user_input)
            
            # Display results
            print("\n" + "=" * 60)
            print("📊 TRIAGE RESULTS")
            print("=" * 60)
            print(f"📋 Classification: {result['classification'].upper()}")
            print(f"🚨 Escalated: {'YES' if result['escalated'] else 'NO'}")
            if result['escalated']:
                print(f"⚡ Priority: {result['priority'].upper()}")
            print("\n" + result['output'])
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
            print("Please try again or contact support.")

def run_test_queries(system):
    """Run a set of predefined test queries."""
    test_queries = [
        "I forgot my password, how do I reset it?",
        "When will my order arrive and how much does shipping cost?",
        "I want to speak to a manager immediately! This service is terrible!",
        "What features are included in the premium plan?",
        "I want to return this item, what's the process?",
        "My account shows as suspended, what does this mean?",
        "How do I cancel my subscription and get a refund?",
        "The app keeps crashing when I try to login"
    ]
    
    print("\n🧪 Running test queries...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}/{len(test_queries)}: {query}")
        print("-" * 40)
        
        try:
            result = system.process_query(query)
            print(f"Classification: {result['classification']}")
            print(f"Escalated: {'YES' if result['escalated'] else 'NO'}")
            if result['escalated']:
                print(f"Priority: {result['priority']}")
            print(f"Response Preview: {result['output'][:100]}...")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n✅ Test completed! Processed {len(test_queries)} queries.")

if __name__ == "__main__":
    main()