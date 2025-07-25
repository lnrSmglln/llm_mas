from langgraph.graph import END, StateGraph
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

# Initialize Ollama (e.g., llama3, mistral)
llm = Ollama(model="deepseek-r1:8b")

# Connect to Neo4j
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)

# Define Graph State
class GraphState:
    def __init__(self, documents=None, chunks=None, graph_data=None):
        self.documents = documents or []
        self.chunks = chunks or []
        self.graph_data = graph_data or []

# Step 1: Load and chunk files
def load_files(state: GraphState):
    loaders = {
        ".docx": UnstructuredFileLoader,
        ".pdf": UnstructuredFileLoader,
        ".xlsx": lambda path: pd.read_excel(path).to_dict(orient="records")
    }
    
    all_docs = []
    for file_path in ["spec.docx", "report.pdf", "data.xlsx"]:
        ext = file_path.split(".")[-1]
        if ext == "xlsx":
            # Process Excel as tables
            table_data = loaders[ext](file_path)
            for row in table_data:
                all_docs.append(f"Table row: {str(row)}")
        else:
            # Process DOCX/PDF
            loader = loaders[ext](file_path)
            all_docs.extend(loader.load())
    
    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    state.chunks = splitter.split_documents(all_docs)
    return state

# Step 2: Extract entities/relationships using Ollama
def extract_graph_data(state: GraphState):
    schema = """
    [INST] Extract engineering entities and relationships in JSON format:
    {
        "nodes": [{"id": "...", "type": "Equipment|Parameter|Material", "properties": {...}}],
        "relationships": [{"source": "...", "target": "...", "type": "HAS_PART|DEPENDS_ON"}]
    }
    [/INST]
    """
    
    graph_data = {"nodes": [], "relationships": []}
    for chunk in state.chunks:
        response = llm.invoke(schema + chunk.page_content)
        try:
            data = json.loads(response)
            graph_data["nodes"].extend(data.get("nodes", []))
            graph_data["relationships"].extend(data.get("relationships", []))
        except:
            continue
    
    state.graph_data = graph_data
    return state

# Step 3: Populate Neo4j
def build_graph(state: GraphState):
    for node in state.graph_data["nodes"]:
        graph.query(
            f"MERGE (n:{node['type']} {{id: $id}} SET n += $props",
            {"id": node["id"], "props": node.get("properties", {})}
        )
    
    for rel in state.graph_data["relationships"]:
        graph.query(
            """MATCH (a), (b) 
            WHERE a.id = $source AND b.id = $target
            MERGE (a)-[r:{type}]->(b)""".format(type=rel["type"]),
            {"source": rel["source"], "target": rel["target"]}
        )
    return state

# Step 4: GraphRAG Query
def graph_rag(state: GraphState, question: str):
    # Retrieve relevant subgraph
    entities = llm.invoke(f"List key entities in: {question}").split(",")
    query = """
    MATCH path = (e)-[r*1..2]-(neighbor)
    WHERE e.id IN $entities
    RETURN path LIMIT 5
    """
    paths = graph.query(query, {"entities": entities})
    
    # Generate answer
    context = "\n".join([str(p) for p in paths])
    prompt = f"""
    [CONTEXT]
    {context}
    
    [QUESTION]
    {question}
    
    Answer as an engineering expert:
    """
    return llm.invoke(prompt)


# Initialize workflow
workflow = StateGraph(GraphState)
workflow.add_node("load_files", load_files)
workflow.add_node("extract_data", extract_graph_data)
workflow.add_node("build_graph", build_graph)

# Define edges
workflow.set_entry_point("load_files")
workflow.add_edge("load_files", "extract_data")
workflow.add_edge("extract_data", "build_graph")
workflow.add_edge("build_graph", END)

# Run with files
app = workflow.compile()
app.invoke({"file_paths": ["design_spec.docx", "test_results.pdf", "materials.xlsx"]})

# Query the graph
graph_rag("What's the safety factor for valve V-101 under pressure surge conditions?")