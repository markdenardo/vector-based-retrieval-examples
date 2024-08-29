# Example of creating an index with dense vectors
PUT /my-index
{
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "dense_vector",
        "dims": 128
      }
    }
  }
}

# Example of searching using cosine similarity
GET /my-index/_search
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "cosineSimilarity(params.query_vector, 'my_vector') + 1.0",
        "params": {
          "query_vector": [0.1, 0.2, ..., 0.128]
        }
      }
    }
  }
}
