import sys
import os
import chromadb
from chromadb.config import Settings

# Assuming your GravitationalWaveDocumentationSearch class is in database.py
# You might need to adjust this import based on your file structure
try:
    from database import GravitationalWaveDocumentationSearch
except ImportError as e:
    print(f"Error: Required modules not found. Please ensure 'database.py' and 'chromadb' are in your environment.")
    print(f"Details: {e}")
    sys.exit(1)

def run_database_tests(db_path: str):
    """
    Initializes a search client and runs a series of tests against the ChromaDB database.
    """
    print(f"Attempting to connect to database at: {db_path}\n")
    try:
        # Create the ChromaDB client instance first
        client = chromadb.PersistentClient(path=db_path)
        
        # Now pass the client object to the searcher class
        searcher = GravitationalWaveDocumentationSearch(client)
        
        print("Test 1: Search across all documentation for 'signal processing'")
        results_all = searcher.search('signal processing')
        print(f"Found {len(results_all['results'])} results.\n")
        
        print("Test 2: Search specifically for 'timeseries' within GWpy docs")
        results_gwpy = searcher.search_by_source('timeseries', 'GWpy')
        print(f"Found {len(results_gwpy['results'])} results from GWpy.\n")
        
        print("Test 3: Search for API functions related to 'FFT'")
        results_api = searcher.search_api_docs('FFT functions')
        print(f"Found {len(results_api['results'])} results from API docs.\n")

        # Display the found results for the last search (API docs)
        if results_api['results']:
            print("Displaying top results from API docs search:")
            for i, result in enumerate(results_api['results']):
                print(f"--- Result {i + 1} ---")
                print(f"Source: {result['metadata'].get('source', 'N/A')}")
                print(f"Title: {result['metadata'].get('title', 'N/A')}")
                print(f"URL: {result['metadata'].get('source_url', 'N/A')}")
                print(f"Snippet: {result['text'][:200]}...")
                print()
        else:
            print("No results found for API docs search.")

        # Test 4: Search specifically for 'API database download' from GWOSC
        print("Test 4: Search for 'API database download' from GWOSC")
        results_gwosc = searcher.search_by_source('API database download', 'GWOSC')
        print(f"Found {len(results_gwosc['results'])} results from GWOSC.\n")

        if results_gwosc['results']:
            print("Displaying top results from GWOSC download search:")
            for i, result in enumerate(results_gwosc['results']):
                print(f"--- Result {i + 1} ---")
                print(f"Source: {result['metadata'].get('source', 'N/A')}")
                print(f"Title: {result['metadata'].get('title', 'N/A')}")
                print(f"URL: {result['metadata'].get('source_url', 'N/A')}")
                print(f"Snippet: {result['text'][:200]}...")
                print()
        else:
            print("No results found for 'API database download' from GWOSC.")

    except Exception as e:
        print(f"\n--- A fatal error occurred during the test! ---")
        print(f"Error: {e}")
        print("Please ensure the database path is correct and the files are not in use by another process.")
        sys.exit(1)

    print("\nAll tests completed successfully! The database appears to be functional.")


if __name__ == "__main__":
    database_directory = "/home/sr/Desktop/code/gravagents/database/code_documentation"
    run_database_tests(database_directory)
