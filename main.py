from fastapi import FastAPI
from Recommendation_system.recommendation_model import hybrid_recommendations, user_item_matrix
from customer_segmentation.model_loader import predict_customer_segment
from customer_segmentation.schema import CustomerData

from typing import List


app=FastAPI(title="Customer Segmentation & Product Recommendation API",
            description="Hybrid Recommendation System using Apriori and Collaborative Filtering",
            version="1.0"
            )


@app.get('/',tags=["Home"])
def home():
    return {"message":"Welcome to the Customer Segmentation & Product Recommendation API"}


@app.post('/predict',tags=["Customer Segmentation"])
def predict(payload:List[CustomerData]):
    data=[item.model_dump(by_alias=True) for item in payload]
    result=predict_customer_segment(data)
    return {"status": "success", "result": result}

@app.get('/products',tags=["Product Recommendation"])
def get_products():
        products=user_item_matrix.columns.tolist()
        return {"products": products}
   

@app.get('/recommend',tags=["Product Recommendation"])
def recommend(product_name: str, top_n: int = 3):
    return hybrid_recommendations(product_name, top_n)