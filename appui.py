import streamlit as st
import requests  # This allows Streamlit to 'talk' to our FastAPI server
from datetime import date
import pandas as pd
import plotly.express as px

from customer_segmentation import ETL_transformer

# 1. Page Setup

st.sidebar.markdown("Navigate between Machine Learning engines:")
page = st.sidebar.radio("Select a Module:", [
    "Customer Persona Prediction",
    "🛒 Smart Recommendations"
])

if page == "Customer Persona Prediction":
    st.set_page_config(page_title="Customer Segemntaion",page_icon="🛒", layout="centered")
    st.title("🛒 E-Commerce Persona Predictor")
    st.markdown("Classify customer purchasing behavior instantly.")
    st.write("Enter the customer's transaction details below:")

    # B. Define where your FastAPI server is listening
    def fetch_predictions(payload):
        API_URL = "https://customer-segmentation-product-recommendation.up.railway.app/predict"  # Update with your actual API endpoint
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                return response.json()["result"] # Returns the clean list of predictions
            else:
                st.error(f"API Error {response.status_code}: Backend issue.")
                return None
        except requests.exceptions.ConnectionError:
            st.error("Could not connect. Check does backend running")
            return None
        
    def create_3d_cluster_plot(new_data_df):
        """
        Takes a dataframe of new predictions, merges it with historical data,
        and returns a formatted Plotly 3D Figure.
        """
        # Load Historical Data
        try:
            historical_df = pd.read_csv("customer_segmentation/Customer_Segmentation_Final.csv")
        except FileNotFoundError:
            st.error("Historical data file not found!")
            return None
            
        historical_df['Point Type'] = 'Historical Data'
        historical_df['Marker Size'] = 3  
        
        # Format the New Data
        # We use .copy() to ensure we don't accidentally modify the original dataframe
        plot_df = new_data_df.copy()
        plot_df['Point Type'] = 'NEW PREDICTION'
        plot_df['Marker Size'] = 15 
        plot_df['Persona'] = "NEW: " + plot_df['Persona'].astype(str)
        # Combine Them
        combined_df = pd.concat([historical_df, plot_df], ignore_index=True)
        
        # Generate the Plotly Figure
        fig = px.scatter_3d(
            combined_df, 
            x='Recency', 
            y='Frequency', 
            z='Monetary',
            color='Persona', 
            symbol='Point Type',        
            size='Marker Size',         
            size_max=15,                
            opacity=0.7,
            hover_name='Customer ID'    
        )
        
        # Clean up the visual layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Recency',
                yaxis_title='Frequency',
                zaxis_title='Monetary',
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        return fig
    # 2. Created Tabs for Single vs. Batch Processing 
    tab1, tab2 = st.tabs(["👤 Single Lookup", "📁 Bulk CSV Upload"])

    # TAB 1: SINGLE CUSTOMER LOOKUP
    # ==========================================
    with tab1:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2) 
            with col1:
                customer_id = st.number_input("Customer ID", value=12345, step=1)
                invoice_no = st.text_input("Invoice Number", value="A001")
            with col2:
                invoice_date = st.date_input("Invoice Date", value=date.today())
                quantity = st.number_input("Quantity", value=1, step=1)
                price = st.number_input("Price (£)", value=50.0, step=0.5)
                
            if st.form_submit_button("Predict Persona"):

                #FRONTEND FORM VALIDATION ---
                #if the Invoice is empty (or just spaces) OR if Price is 0
                if invoice_no.strip() == "" or price <= 0:
                    st.error("Please enter a valid Invoice Number and ensure the Price is greater than £0.00.")
                
                # If the data is good, proceed to talk to the API!
                else:
                    payload = [{"Invoice": invoice_no, "Quantity": quantity, "InvoiceDate": str(invoice_date), "Price": price, "Customer ID": customer_id}]
                
                    with st.spinner("Analyzing..."):
                        results = fetch_predictions(payload) # CALLING THE HELPER FUNCTION
                    
                    if results:
                        predicted_persona = results[0]["Persona"]
                        if "Returned" in predicted_persona:
                            st.warning(f"Transaction Flagged: {predicted_persona}")
                        elif predicted_persona == "VIP":
                            st.success(f"Prediction Complete: This customer is a **{predicted_persona}**! 🌟")
                        else:
                            st.info(f"Prediction Complete: This customer is classified as **{predicted_persona}**.")

                        with st.expander("📍 View Customer Location in 3D Space"):

                            #DataFrame for the new prediction to plot
                            single_api_result = pd.DataFrame([{"Customer ID": str(customer_id), "Persona": predicted_persona}])

                            # 2. Package into a 1-row DataFrame that looks like an invoice
                            single_invoice_df = pd.DataFrame([{
                                "Customer ID": str(customer_id),
                                "Quantity": quantity,
                                "Price": price,
                                "InvoiceDate": str(invoice_date),
                                "Invoice": invoice_no
                            }])

                            # We use your transformer to ensure the calculation matches the training data
                            single_rfm_coords = ETL_transformer.extract_rfm(single_invoice_df).reset_index()
                            # MERGE: Attach the label to the coordinates
                            single_plot_ready = pd.merge(
                                single_rfm_coords, 
                                single_api_result[['Customer ID', 'Persona']], 
                                on="Customer ID", 
                                how="left"
                            )

                            st.subheader("🎯Visualization of Customer on Clusters ")
                            with st.spinner("Plotting customer position..."):
                                fig_single = create_3d_cluster_plot(single_plot_ready)
                                
                                if fig_single:
                                    st.plotly_chart(fig_single, use_container_width=True)

    # ==========================================
    # TAB 2: BATCH PROCESSING (CSV UPLOAD)

    with tab2:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(3))
            
            if st.button("Predict Entire Batch"):

            #PANDAS DATA VALIDATION ---
                initial_records_count = len(df)
            
                #  Force Invoice to be a string so we can check for blanks
                df['Invoice'] = df['Invoice'].astype(str).copy()
                
                # 2. Filtered the DataFrame: Kept only rows where Price > 0 AND Invoice is not blank/nan
                df_clean = df[(df['Price'] > 0) & (df['Invoice'].str.strip() != "") & (df['Invoice'] != "nan")]
                
                # 3. Calculate if we dropped any garbage data
                dropped_rows = initial_records_count - len(df_clean)
                
                if dropped_rows > 0:
                    st.warning(f"Removed {dropped_rows} invalid rows (Which have Missing Invoices or Price £0/Negative).")
                
                # 4. Only proceed if there is actually valid data left!
                if len(df_clean) == 0:
                    st.error("All uploaded rows contained invalid data. Prediction aborted.")
                else:
                    with st.spinner("Processing massive dataset..."):
                        
                        df_clean['InvoiceDate'] = df_clean['InvoiceDate'].astype(str).copy()
                        
                        # Convert the CLEANED dataframe to a dictionary payload
                        payload = df_clean.to_dict(orient="records")
                        
                        results = fetch_predictions(payload) 
                        
                        if results:
                            
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df.drop_duplicates(subset=["Customer ID"]).reset_index(drop=True))
                            
                            with st.expander("📍 View Customer Location in 3D Space"):
                        
                    
                                # Generate the RFM coordinates from your cleaned data
                                # This is the step you were missing!
                                rfm_coords_df = ETL_transformer.extract_rfm(df_clean).reset_index()
                                
                                # Merge them together so we have the coordinates and the predicted labels in one dataframe
                                plot_ready_df = pd.merge(rfm_coords_df, results_df, on="Customer ID", how="left")
                                
                                # 4. Clean up: Remove any 'Returned/Invalid' rows if they exist
                                plot_ready_df = plot_ready_df[plot_ready_df['Persona'] != 'Returned/Invalid (Quantity/Price issue)']

                                st.subheader("🎯Visualization of Customers on Clusters ")
                                fig = create_3d_cluster_plot(plot_ready_df)
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)


elif page == "🛒 Smart Recommendations":

    st.set_page_config(page_title="Smart Recommendations", page_icon="🛒")

    st.title("🛒 Smart Product Recommendation Engine")
    st.markdown("Hybrid Recommendation System using Apriori + Collaborative Filtering")

    #Load products dynamically from API
    #@st.cache_data
    def load_products():
        try:
            res = requests.get("https://customer-segmentation-product-recommendation.up.railway.app/products")
            return res.json()["products"]
        except:
            return []

    products = load_products()

    if not products:
        st.error("Could not load products. Ensure backend is running.")
        st.stop()

    #UI Layout
    col1, col2 = st.columns([3, 1])

    with col1:
        product = st.selectbox("Select Product", products)

    with col2:
        top_n = st.number_input("Number of Recommendations", min_value=1, max_value=10, value=3)

    # Get Recommendations Button
    if st.button("Get Recommendations", use_container_width=True):

        with st.spinner("Finding best products for you..."):

            try:
                res = requests.get(
                    "https://customer-segmentation-product-recommendation.up.railway.app/recommend",
                    params={"product_name": product, "top_n": top_n}
                )

                if res.status_code == 200:
                    data = res.json()

                    # 🔹 Title
                    st.subheader(f"🎯 Recommendations for {product}")

                    method = data["method"]

                    # Method indicator (global)
                    if "Hybrid" in method:
                        st.warning("Hybrid Recommendations (Rule-based + Behavioral)")
                    elif "Apriori" in method:
                        st.success("Rule-based Recommendations (Frequently Bought Together)")
                    elif "Collaborative" in method:
                        st.info("Personalized Recommendations (User Behavior Based)")

                    st.markdown("---")

                    # Display cards
                    cols = st.columns(top_n)

                    for idx, rec in enumerate(data["data"]):

                        with cols[idx]:

                            #Smart keyword extraction for image
                            words = rec['product'].lower().split()
                            filtered_words = [w for w in words if not w.isdigit()]
                            keyword = " ".join(filtered_words[:4])

                            img_url = f"https://placehold.co/200x200/?text={keyword}"

                            st.image(img_url, use_container_width=True)

                            # 🔹 Product Name
                            st.markdown(f"**{rec['product']}**")

                                                        
                            source = rec.get("source")

                            if not source:
                                source = "Apriori" if "lift" in rec else "CF"

                            if source == "Apriori":
                                st.success("📊 Apriori (Rule-Based)")
                                st.markdown(f"🔥 Lift: {rec['lift']}x")
                                st.markdown(f"🎯 Confidence: {rec['confidence_percent']}%")
                                st.caption("💡 Strong association: Frequently bought together")
                                st.caption(f"💡 If a customer buys the selected product, this item is also purchased {rec['confidence_percent']}% of the time.")

                            elif source == "CF":
                                st.info("🤖 Collaborative Filtering")
                                st.markdown(f"🤖 Similarity: {rec['similarityscore']}")
                                st.caption("💡 Based on similar customer purchase behavior")

                else:
                    st.error(f"API Error: {res.status_code}")

            except:
                st.error("⚠️ Could not connect to backend. Make sure FastAPI is running.")

    #Explanation Section
    with st.expander("ℹ️ How recommendations work"):
        st.markdown("""
        **🔹 Apriori (Rule-Based)**
        - Identifies products frequently bought together  
        - Uses **Lift** and **Confidence**

        **🔹 Collaborative Filtering**
        - Recommends based on user purchase behavior patterns  
        - Uses **Similarity Score**
        - Similarity score ranges from 0 to 1. Higher values indicate stronger similarity based on customer behavior.")

        **🔹 Hybrid Model**
        - Combines both methods for better coverage and accuracy  

        ⚠️ *Images are illustrative and may not exactly match the product.*
        """)