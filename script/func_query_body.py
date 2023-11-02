def to_list(data):
    if type(data) != list:
        return [data]
    return data

def query_tag(tag_name):
    return {
        "query": 
        {
            "term": {"tag": tag_name}
        },
        "_source": {"excludes": ["features"]}
    }

def query_tag_count(tag_name):
    return {
        "query": 
        {
            "term": {"tag": tag_name}
        }
    }

def query_cosine(unit_vector, tag_name_compare="train_split", top_n=5, collapse=True):
    tag_name_compare = to_list(tag_name_compare)
    query = {
        "query": 
        {
            "function_score": 
            {
                "query":
                {
                    "terms": {"tag": tag_name_compare}
                },
            
                "functions": 
                [
                    {
                        "script_score": 
                        {
                            "script": 
                            {
                                "source": "cosineSimilarity(params.query_vector, 'features')/2+0.5",
                                "params": {"query_vector": unit_vector}  # Replace with your query vector
                            }
                        }
                    }
                ], 
                "boost_mode": "replace"
            }
        },
        
        "_source": {"excludes": ["features"]},
        "size": top_n,
        "sort": 
        [
            {"_score": "desc"}
        ],
    }
    if collapse:
        query['collapse'] = {"field": "labels"}
    return query