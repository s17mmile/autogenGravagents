import chromadb

# ------------------------------
# Setup: Connect to your database
# ------------------------------
db_path = "/home/sr/Desktop/code/gravagents/database/memory"
client = chromadb.PersistentClient(path=db_path)

# List collections
collections = [col.name for col in client.list_collections()]
print("Available collections:", collections)

collection_name = input("Enter the collection name you want to manage: ").strip()
collection = client.get_collection(collection_name)

# ------------------------------
# Helper function: Show only query and execution status
# ------------------------------
def show_queries(limit=20):
    results = collection.get(include=["documents", "metadatas"], limit=limit)
    if not results['documents']:
        print("No entries found.")
        return []

    print("\nSession Queries and Execution Status:")
    for i, doc in enumerate(results['documents']):
        status = results['metadatas'][i].get("status", "Unknown")
        print(f"{i+1}. ID: {results['ids'][i]}")
        print(f"   Query: {doc}")
        print(f"   Status: {status}")
    return results['ids']

# ------------------------------
# Main loop
# ------------------------------
while True:
    print("\n--- ChromaDB Session Viewer ---")
    print("1. Show queries")
    print("2. Delete a query")
    print("3. Exit")
    choice = input("Choose an option: ").strip()

    if choice == "1":
        show_queries()
    elif choice == "2":
        ids = show_queries()
        if not ids:
            continue
        delete_id = input("Enter the ID of the query to delete: ").strip()
        if delete_id in ids:
            try:
                collection.delete(ids=[delete_id])
                print(f"Query {delete_id} deleted successfully.")
            except Exception as e:
                print(f"Error deleting query {delete_id}: {e}")
        else:
            print("Invalid ID.")
    elif choice == "3":
        print("Exiting...")
        break
    else:
        print("Invalid option. Please choose 1, 2, or 3.")
