import streamlit as st
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import os

# --- 1. Initialization ---
st.set_page_config(page_title="Zap AI Buying Advisor", layout="wide")
st.title("Zap AI Buying Advisor")

# --- 2. Loading Simulated Data ---
@st.cache_data
def load_data():
    """Load the product data from the CSV files."""
    df_tv = pd.read_csv("tv_data.csv")
    df_mobile = pd.read_csv("mobile_data.csv")
    return df_tv, df_mobile

df_tv, df_mobile = load_data()

# --- 3. AI (LangChain) Configuration ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDqvrRZ8kMCANELBPxS0aoovg7IO5S45t4"

prompt_template = PromptTemplate(
    input_variables=["user_input", "tv_data", "mobile_data"],
    template="""
    You are an expert AI Shopping Assistant named "ZapBot" for the Israeli e-commerce platform Zap.co.il. Your primary role is to help users find the best products by asking them questions about their needs and then recommending suitable options from the provided lists. You can assist with either Televisions or Mobile Phones.

    **Your Guiding Principles:**
    1.  **Determine the Product Type:** First, you must determine if the user is looking for a TV or a mobile phone based on their request.
    2.  **Be Data-Driven:** Base all your recommendations ONLY on the provided product data for the correct category. DO NOT invent information or recommend products not in the list.
    3.  **Synthesize, Don't Just List:** Instead of a long list, synthesize the key information. For each recommended product, provide a brief summary of why it's a good fit for the user's request.
    4.  **Prioritize User Needs:** Always address the specific points mentioned by the user, such as budget, use case (e.g., gaming, photo quality), and in-store pickup options.
    5.  **Assign a Compatibility Score:** For each product you recommend, you MUST assign a "Compatibility Score" out of 100. This score reflects how well the product's features and reviews align with the user's request. A higher score indicates a better match.
    6.  **Maintain Professionalism:** Your responses should be clear, well-structured, and easy to read. Use bullet points or a clear format to present recommendations.
    7.  **Handle Ambiguity Gracefully:** If the user's request is unclear, ask clarifying questions. If no products match, tell the user politely and offer to help with a different request.

    **Your Instructions for Each Interaction:**
    * You will receive the user's request and two CSV strings: one for TV data and one for mobile phone data.
    * Based on the user's request, you will only use the relevant dataset (either `tv_data` or `mobile_data`) for your recommendations.
    * For each matching product, provide:
        * **Product Name:** [e.g., iPhone 16 Pro Max]
        * **Compatibility Score:** [A score out of 100]
        * **Why it's a good fit:** [A brief sentence explaining how it meets the user's needs]
        * **Price & Sellers:** [List the price and seller(s) for comparison]
        * **Key Insights from Reviews:** [Summarize the most common pros and cons from the "reviews" column]

    **If no products are a perfect match**, suggest the closest alternatives or offer to adjust the search criteria.

    TV Data (in CSV string format):
    {tv_data}

    Mobile Phone Data (in CSV string format):
    {mobile_data}

    User's Request:
    {user_input}

    Your Response:
    """
)

llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# --- 4. Streamlit Conversation Loop ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What are you looking for?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Finding the best products for you..."):
            # Prepare both datasets as strings for the LLM
            tv_data_string = df_tv.to_csv(index=False)
            mobile_data_string = df_mobile.to_csv(index=False)
            
            # Pass both data strings to the LLM and let it decide
            response_content = llm_chain.run(user_input=prompt, tv_data=tv_data_string, mobile_data=mobile_data_string)
        
        st.markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.session_state.last_response_type = "products" # Set a flag for displaying the table

# --- 5. Adding the Dynamic Comparison Table ---
st.write("---")
st.subheader("Product Comparison")
st.write("Click the button below to see a detailed comparison of all products.")

# Define the function to display the table based on the last user request
def display_comparison_table(last_prompt):
    # Determine which table to show based on keywords in the last prompt
    if "phone" in last_prompt.lower() or "mobile" in last_prompt.lower():
        st.subheader("Mobile Phone Comparison")
        columns_to_show = ["product_name", "screen_size_inches", "processor", "ram_gb", "storage_gb", "rear_camera_mp", "battery_mah", "price_shekels"]
        st.dataframe(df_mobile[columns_to_show].set_index('product_name'))
    elif "tv" in last_prompt.lower() or "television" in last_prompt.lower() or "monitor" in last_prompt.lower():
        st.subheader("TV Comparison")
        columns_to_show = ["product_name", "screen_size_inches", "resolution", "panel_type", "refresh_rate_hz", "price_shekels"]
        st.dataframe(df_tv[columns_to_show].set_index('product_name'))
    else:
        st.write("Please specify 'TV' or 'mobile phone' in your request to see the comparison table.")

# Create a button to show the table
if st.button("Show Full Product Comparison Table"):
    if st.session_state.messages:
        last_prompt = st.session_state.messages[-2]['content']
        display_comparison_table(last_prompt)
    else:
        st.write("Hello there! To help you find the perfect product, please start by chatting with Zapbot. Once you've made a request, a comparison table tailored to your needs will become available here.")
