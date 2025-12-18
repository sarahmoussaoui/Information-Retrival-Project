"""Base retrieval model interface."""

class RetrievalModel:
    def fit(self, corpus):
        """Prepare the model with the given corpus."""
        raise NotImplementedError

    def rank(self, query):
        """Return ranked document ids for the query."""
        raise NotImplementedError
