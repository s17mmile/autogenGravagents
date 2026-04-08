import chromadb, os
client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "flexibleAgents", "chromaDatabase"))
print(client.list_collections())