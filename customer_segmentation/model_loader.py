import joblib
import sys
from customer_segmentation import ETL_transformer

sys.modules['ETL_transformer'] = ETL_transformer


model=joblib.load('customer_segmentation/customer_segmentation.pkl')
persona_map={0:'Loyal',1:'At Risk',2:'VIP',3:'Wholesaler'}

def predict_customer_segment(data):

    valid_data=[]
    invalid_data=[]
    for item in data:
        if (item['Quantity'] >0 and item['Price']>0):
            valid_data.append(item)
        else :
            invalid_data.append(item)

    final_lst=[]
    if len(valid_data)>0:

        rfm_features = ETL_transformer.extract_rfm(valid_data)
        unique_customer_ids = rfm_features.index
        
        pred_cluster=model.predict(valid_data)

        for i, cid in enumerate(unique_customer_ids):
            current_id=cid
            current_persona=persona_map[pred_cluster[i]]
            final_lst.append({'Customer ID':current_id, 'Persona':current_persona})

    if len(invalid_data)>0:
        unique_invalid_cids=set(item['Customer ID'] for item in invalid_data)
        for cid in unique_invalid_cids:
            cid = cid 
            final_lst.append({'Customer ID': cid, 'Persona': 'Reutrned/Invalid (Quantity/Price issue)'})

    return final_lst




