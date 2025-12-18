"""Parse raw MED documents, queries, and relevance judgments."""


def parse_documents(raw_text: str) -> dict[int, str]:
    """Parse MED.ALL contents into a structured list.
    
    Args:
        raw_text: The full text content of MED.ALL file
        
    Returns:
        dict[int, str]: Mapping of doc_id to document_text
    """
    documents = {}
    current_doc_id = None
    current_text = []
    
    for line in raw_text.split('\n'):
        line = line.strip()
        
        if line.startswith('.I '):
            # Save previous document if exists
            if current_doc_id is not None:
                documents[current_doc_id] = ' '.join(current_text).strip()
            
            # Start new document
            current_doc_id = int(line[3:].strip())
            current_text = []
            
        elif line.startswith('.W'):
            # Start of document text content
            continue
            
        elif line and not line.startswith('.'):
            # Accumulate document text
            current_text.append(line)
    
    # Save last document
    if current_doc_id is not None:
        documents[current_doc_id] = ' '.join(current_text).strip()
    
    return documents


def parse_queries(raw_text: str) -> dict[int, str]:
    """Parse MED.QRY contents into a structured list.
    
    Args:
        raw_text: The full text content of MED.QRY file
        
    Returns:
        dict[int, str]: Mapping of query_id to query_text
    """
    queries = {}
    current_query_id = None
    current_text = []
    
    for line in raw_text.split('\n'):
        line = line.strip()
        
        if line.startswith('.I '):
            # Save previous query if exists
            if current_query_id is not None:
                queries[current_query_id] = ' '.join(current_text).strip()
            
            # Start new query
            current_query_id = int(line[3:].strip())
            current_text = []
            
        elif line.startswith('.W'):
            # Start of query text content
            continue
            
        elif line and not line.startswith('.'):
            # Accumulate query text
            current_text.append(line)
    
    # Save last query
    if current_query_id is not None:
        queries[current_query_id] = ' '.join(current_text).strip()
    
    return queries


def parse_qrels(raw_text: str) -> dict[int, set[int]]:
    """Parse MED.REL contents into relevance judgments.
    
    Args:
        raw_text: The full text content of MED.REL file
        
    Returns:
        dict[int, set[int]]: Mapping of query_id to set of relevant doc_ids
    """
    qrels = {}
    
    for line in raw_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 4:
            query_id = int(parts[0])
            doc_id = int(parts[2])
            relevance = int(parts[3])
            
            # Only store relevant documents (relevance = 1)
            if relevance == 1:
                if query_id not in qrels:
                    qrels[query_id] = set()
                qrels[query_id].add(doc_id)
    
    return qrels
