import streamlit as st
import google.generativeai as genai
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import tempfile
import base64

st.set_page_config(page_title="Plagiarism Detector", page_icon="📝", layout="centered")

with st.sidebar:
    st.title("🔑 API Keys Required")
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("[Get Gemini API Key](https://makersuite.google.com/app/apikey)")
    google_api_key = st.text_input("Enter Google Search API Key", type="password")
    google_cse_id = st.text_input("Enter Google CSE ID", type="password")
    st.markdown("[Get Google API Key & CSE ID](https://developers.google.com/custom-search/v1/introduction)")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    st.warning("⚠️ Please enter your Gemini API key to proceed.")

if not (google_api_key and google_cse_id):
    st.warning("⚠️ Please enter your Google Search API Key and CSE ID for full internet search.")

st.markdown("<h1 style='text-align: center;'>📝 AI-Powered Plagiarism Detector</h1>", unsafe_allow_html=True)
st.write("🔍 **Detect plagiarism** across the internet using AI and NLP with instant similarity scoring and source detection.")

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""
if "plagiarism_report" not in st.session_state:
    st.session_state["plagiarism_report"] = ""

def search_web(query, api_key, cse_id):
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": api_key, "cx": cse_id, "q": query[:200]}  
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json().get("items", [])
        return []
    except Exception as e:
        return [{"link": f"Error in search: {str(e)}", "snippet": ""}]

def check_plagiarism(input_text, gemini_api_key, google_api_key, google_cse_id):
    prompt = (
        "Analyze the following text for plagiarism and return a structured JSON response.\n"
        "Find real matching sources and return a JSON object with 'plagiarism_score' and 'sources'.\n"
        "ONLY return real sources. If no sources are found, return an empty list.\n\n"
        "Ensure response is in **valid JSON format** inside ```json ... ``` markdown blocks.\n\n"
        f"Text:\n{input_text}\n\n"
        "Expected JSON format:\n"
        "```json\n"
        "{\n"
        '  "plagiarism_score": "XX%",\n'
        '  "sources": ["https://example.com", "https://validsource.com"]\n'
        "}\n"
        "```"
    )

    ai_json = {"plagiarism_score": "0%", "sources": []}
    if gemini_api_key:
        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                response_text = response.text.strip()
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                ai_json = json.loads(response_text)
        except Exception as e:
            ai_json["sources"] = [f"Gemini Error: {str(e)}"]

    web_sources = []
    if google_api_key and google_cse_id:
        web_results = search_web(input_text, google_api_key, google_cse_id)
        for result in web_results[:5]:  
            snippet = result.get("snippet", "")
            similarity = compute_similarity(input_text, snippet)
            if similarity > 20:  
                web_sources.append(result["link"])

    ai_score = float(ai_json.get("plagiarism_score", "0%").replace("%", ""))
    web_score = max([0] + [compute_similarity(input_text, r.get("snippet", "")) for r in web_results[:5] if google_api_key and google_cse_id])
    combined_score = max(ai_score, web_score)
    combined_sources = list(set(ai_json.get("sources", []) + web_sources))

    return {
        "plagiarism_score": f"{combined_score:.1f}",
        "sources": combined_sources if combined_sources else ["⚠️ No valid sources found"]
    }

def verify_source_exists(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def compute_similarity(original, generated):
    if not original.strip() or not generated.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original, generated])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0] * 100

text_to_check = st.text_area("\U0001F4CC **Paste Your Text Below:**", value=st.session_state["text_input"], height=200, 
                            placeholder="Enter your text here...", key="text_input")

col1, col2, col3 = st.columns(3)

if col3.button("🛑 Stop", use_container_width=True):
    st.warning("Process stopped!")

if col2.button("❌ Clear Text", use_container_width=True):
    del st.session_state["text_input"]
    del st.session_state["plagiarism_report"]
    st.rerun()

if col1.button("🔎 Check Plagiarism", use_container_width=True):
    if not (gemini_api_key or (google_api_key and google_cse_id)):
        st.warning("⚠️ Please enter at least one API key to proceed.")
    elif not text_to_check.strip():
        st.warning("⚠️ Please enter valid text to check for plagiarism.")
    else:
        with st.spinner("🔍 Analyzing text for plagiarism across the internet..."):
            result = check_plagiarism(text_to_check, gemini_api_key, google_api_key, google_cse_id)

            report = (
                f"Plagiarism Score: {result['plagiarism_score']}%\n"
                f"Matching Sources:\n" + "\n".join([f"- {src}" for src in result['sources']])
            )
            st.session_state["plagiarism_report"] = report
            
            st.success("✅ **Analysis Complete!**")
            st.subheader("🔍 Plagiarism Report")
            st.metric(label="🚗 Plagiarism Score", value=f"{result['plagiarism_score']}%", delta="Higher means more plagiarism")
            st.write("📌 **Matching Sources:**")
            for source in result["sources"]:
                st.markdown(f"🔗 [{source}]({source})")
            
            score = float(result["plagiarism_score"])
            if score > 50:
                st.error("⚠️ **High plagiarism detected!** Consider rewriting the content.")
            elif 20 <= score <= 50:
                st.warning("⚠️ **Moderate plagiarism detected.** Some text may be similar.")
            else:
                st.success("✅ **No significant plagiarism detected!**")

def generate_pdf(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

if st.button("🌐 Print Report as PDF", use_container_width=True):
    if not st.session_state["plagiarism_report"]:
        st.warning("⚠️ No report available. Check plagiarism first.")
    else:
        pdf_path = generate_pdf(st.session_state["plagiarism_report"])
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode()
        pdf_link = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="plagiarism_report.pdf">📅 Download PDF</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
.stTextArea textarea {font-size: 18px; padding: 12px;}
.stMetric {text-align: center;}
.stButton>button {border-radius: 8px; font-size: 16px; background-color: #4CAF50; color: white;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)