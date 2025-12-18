"""Parse raw MED documents, queries, and relevance judgments."""

def parse_documents(raw_text: str):
    """Parse MED.ALL contents into a structured list."""
    raise NotImplementedError


def parse_queries(raw_text: str):
    """Parse MED.QRY contents into a structured list."""
    raise NotImplementedError


def parse_qrels(raw_text: str):
    """Parse MED.REL contents into relevance judgments."""
    raise NotImplementedError
