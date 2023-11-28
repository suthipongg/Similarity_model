def to_list(data, add_one_img=False):
    if type(data) != list:
        return [data, "one_img"] if add_one_img else [data]
    return data + ["one_img"] if "one_img" not in data and add_one_img else data

def query_tag(tag_name):
    query = {}
    query["query"] = {"term": {"tag": tag_name}}
    query["_source"] = {"excludes": ["features"]}
    return query

def query_tag_count(tag_name):
    query = {}
    query["query"] = {"term": {"tag": tag_name}}
    return query

def query_cosine(unit_vector, tag_name_compare="train_split", top_n=5, collapse=True, 
                 filter_all=None, use_production=False):
    tag_name_compare = to_list(tag_name_compare, add_one_img=True)
    query = {}
    query["query"] = {}
    query["query"]["function_score"] = {}
    query["query"]["function_score"]["query"] = {}
    query["query"]["function_score"]["query"]["bool"] = {}
    query_filter = [
        {"terms": {"tag": tag_name_compare}}
    ]

    if filter_all is not None:
        for key in filter_all:
            query_filter += [{"terms": {key: filter_all[key]}}]
            
    if use_production:
        query_filter.append({"term": {"action": 1}})
    query["query"]["function_score"]["query"]["bool"]["must"] = query_filter
    query_function = []
    script_score = {}
    script_score["script_score"] = {}
    script_score["script_score"]["script"] = {
        "source": "cosineSimilarity(params.query_vector, 'features')/2+0.5",
        "params": {"query_vector": unit_vector}  # Replace with your query vector
    }
    query_function.append(script_score)
    query["query"]["function_score"]["functions"] = query_function
    query["query"]["function_score"]["boost_mode"] = "replace"
    query["_source"] = {"excludes": ["features"]}
    query["size"] = top_n
    query["sort"] = [{"_score": "desc"}]
    if collapse:
        query['collapse'] = {"field": "labels"}
    return query