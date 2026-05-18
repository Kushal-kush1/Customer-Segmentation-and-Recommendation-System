import ast
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 1. Load and parse the CSV OUTSIDE the function (Runs only once on startup)
rules_df = pd.read_csv('Recommendation_system/apriori_bestrules.csv')

def parse_frozenset(string_val):
    inner_string = string_val.replace('frozenset(', '').replace(')', '')
    return frozenset(ast.literal_eval(inner_string))

rules_df['antecedents'] = rules_df['antecedents'].apply(parse_frozenset)
rules_df['consequents'] = rules_df['consequents'].apply(parse_frozenset)

# 2. Removed rules_df from the parameters
def get_recommendations(product, top_n=3): 
    """
    Takes a product name and returns the top N recommended products based on Lift.
    """
    matching_rules = rules_df[rules_df['antecedents'].apply(lambda x: product in x)]

    sorted_rules = matching_rules.sort_values(['lift', 'confidence'], ascending=[False, False])

    recommendations = []
    for idx, row in sorted_rules.head(top_n).iterrows():
        recommended_item = list(row['consequents'])[0]
        recommendations.append({
            "product": recommended_item,
            "lift": round(row['lift'], 2),
            "confidence_percent": round(row['confidence'] * 100, 1)
        })
    if len(recommendations)==0:
        return None

    return {
            "status": "Success",
            "method": "Apriori",
            "data": recommendations
             }


user_item_matrix = pd.read_parquet(
    "Recommendation_system/useritem_matrix.parquet"
)


# Generate item-item similarity dynamically becz it is of huge size deployment fails due to memory constraints. This way we can compute similarity on the fly without storing the huge matrix.
item_similarity = cosine_similarity(user_item_matrix.T)

# Convert to dataframe
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)
def recommendation_cf(product,top_n=3):

    if product not in user_item_matrix.columns:
        top_products=user_item_matrix.sum(axis=0).sort_values(ascending=False).head(top_n)
        return {
                "status":"Fallback", 
                "method":"Popularity-based", 
                "data":[{"product": rec, "similarityscore": None} for rec in top_products.index.tolist() ]
                }    
    
    recommendations=item_similarity_df[product].sort_values(ascending=False).drop(product).head(top_n)
   
    return  {
             "status":"Success", 
             "method":"Collaborative Filtering",
             "data":[{"product": rec, "similarityscore": round(val, 2)} for rec, val in recommendations.items()]
             }
             


def hybrid_recommendations(product,top_n=3):
    apriori_recs=get_recommendations(product, top_n)
    cosine_recs=recommendation_cf(product,top_n)

    if apriori_recs is None:
        return cosine_recs   
    
    elif len(apriori_recs["data"]) >=top_n:
        return apriori_recs
    
    else:
        apriori_data = [{ **item, "source": "Apriori"} for item in apriori_recs["data"]]
        cf_recs = [
            {**item, "source": "CF"}
                   for item in cosine_recs["data"]
                   if item["product"] not in apriori_data]

    combined_recs = apriori_data + cf_recs
    return {"status": "Success",
                "method":"Hybrid (Apriori + Collaborative Filtering)",
                "data": combined_recs[:top_n]   }