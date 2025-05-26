import streamlit as st
import pdfplumber
import re
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import altair as alt

def clean_semester_bracket(line):
    return re.sub(r'\((1|2)학기\)', '', line).strip()

def clean_hours_bracket(line):
    return re.sub(r'\(\d+\s*시간\)', '', line).strip()

def extract_and_clean_pdf(pdf_path):
    remove_startswith = ["충남삼성고등학교"]
    remove_patterns = [
        r'구\s*분.*명칭.*번호.*취득연월일.*발급기관',
        r'학년.*학기.*세분류.*이수시간.*원점수.*성취도.*비고',
        r'일자.*장소.*활동내용.*시간.*누계시간'
    ]

    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages[1:]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    cleaned_lines = []
    for line in text.splitlines():
        stripped_line = line.strip()
        if len(stripped_line) <= 20:
            continue
        if any(stripped_line.startswith(word) for word in remove_startswith):
            continue
        if any(re.search(pattern, stripped_line) for pattern in remove_patterns):
            continue
        if re.search(r'\d+/\d+', stripped_line) or re.search(r'[ABC]\(\d+(\.\d+)?\)', stripped_line):
            continue
        if re.match(r'^\d', stripped_line):
            continue

        cleaned_line = clean_semester_bracket(stripped_line)
        cleaned_line = clean_hours_bracket(cleaned_line)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)

def load_keyword_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

def 분석_전체(pdf_path, categories):
    cleaned_text = extract_and_clean_pdf(pdf_path)

    with st.spinner("SBERT 모델 로딩 중..."):
        model = load_sbert_model()

    sentences = re.split(r'[.!?]\s*|\n', cleaned_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    sentence_embeddings = model.encode(sentences)

    chart_data_all = {}

    for category, lines in categories.items():
        documents = [line.split(':', 1)[1] if ':' in line else line for line in lines]
        labels = [line.split(':', 1)[0] if ':' in line else line[:10] for line in lines]

        keyword_embeddings = model.encode(documents)
        sim_matrix = util.cos_sim(keyword_embeddings, sentence_embeddings).cpu().numpy()

        keyword_evidence = {}
        for i, label in enumerate(labels):
            sentence_scores = list(zip(sentences, sim_matrix[i]))
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:3]
            keyword_evidence[label] = top_sentences

        main_embedding = model.encode([cleaned_text])
        sbert_sims = util.cos_sim(main_embedding, keyword_embeddings)[0].cpu().numpy().tolist()

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_text] + documents)
        tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        hybrid_scores = [tfidf_score * (1 + sbert_sim) for tfidf_score, sbert_sim in zip(tfidf_scores, sbert_sims)]
        df = pd.DataFrame({
            '키워드': labels,
            'TF-IDF': tfidf_scores,
            'SBERT': sbert_sims,
            '의미 가중 점수': hybrid_scores,
            '근거 문장': [keyword_evidence[label] for label in labels]
        })

        df_for_chart = df.drop(columns=['근거 문장'])
        top_df = df_for_chart.sort_values('의미 가중 점수', ascending=False).head(5)

        chart_data_all[category] = {
            'chart_df': top_df,
            'evidence': keyword_evidence
        }


    return cleaned_text, chart_data_all

st.set_page_config(page_title="생기부 키워드 분석 TOOL", layout="wide")

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

st.title("🚀 생기부 키워드 분석 TOOL")

uploaded_file = st.file_uploader("생기부 PDF 업로드", type=["pdf"])

if uploaded_file:
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(uploaded_file.read())
    temp_pdf.close()

    try:
        categories = {
            '역량': load_keyword_lines('역량.txt'),
            '태도': load_keyword_lines('태도.txt'),
            '관심사': load_keyword_lines('관심사.txt')
        }
    except Exception:
        st.error("⚠️ 키워드 파일이 없습니다.")
        os.unlink(temp_pdf.name)
        st.stop()

    if not st.session_state.analyzed and not st.session_state.analyze_clicked:
        if st.button("🔍 분석 시작"):
            st.session_state.analyze_clicked = True
            st.rerun()

    if st.session_state.analyze_clicked and not st.session_state.analyzed:
        with st.spinner("분석 중..."):
            cleaned_text, chart_data_all = 분석_전체(temp_pdf.name, categories)
            st.session_state.cleaned_text = cleaned_text
            st.session_state.chart_data_all = chart_data_all
            st.session_state.analyzed = True
            st.session_state.analyze_clicked = False
            st.success("✅ 분석 완료.")

    if st.session_state.analyzed:
        if st.button("📊 결과 확인하기"):
            st.session_state.page = 'result'

    os.unlink(temp_pdf.name)

if st.session_state.get('page') == 'result':
    st.title("📊 분석 결과")

    chart_data_all = st.session_state.chart_data_all
    cleaned_text = st.session_state.cleaned_text

    tab1, tab2 = st.tabs(["상위 키워드 요약", "의미 가중 점수 그래프"])

    with tab1:
        st.write("### 🏷️ 상위 키워드 + 근거 문장 보기")
        cols = st.columns(3)
        for i, (category, data) in enumerate(chart_data_all.items()):
            df = data['chart_df']
            evidence_dict = data['evidence']

            with cols[i]:
                st.markdown(f"<h3 style='color:#FF4B4B'>{category}</h3>", unsafe_allow_html=True)
                for _, row in df.iterrows():
                    with st.expander(f"🔍 {row['키워드']}"):
                        for sent, score in evidence_dict[row['키워드']]:
                            st.markdown(f"- {sent}  \n<span style='color:gray'>유사도: {score:.3f}</span>", unsafe_allow_html=True)

    with tab2:
        st.write("### 📈 의미 가중 점수 그래프")
        category_colors = {
            '역량': '#FF4B4B',
            '태도': '#4B7BEC',
            '관심사': '#FFA41B'
        }

        cols = st.columns(3)
        for i, (category, data) in enumerate(chart_data_all.items()):
            df = data['chart_df']

            with cols[i]:
                st.markdown(f"##### {category}")
                chart = alt.Chart(df).mark_bar(
                    cornerRadiusTopLeft=10,
                    cornerRadiusTopRight=10,
                    color=category_colors.get(category, '#888888')
                ).encode(
                    x=alt.X('키워드', sort='-y'),
                    y=alt.Y('의미 가중 점수'),
                    tooltip=['키워드', '의미 가중 점수', 'TF-IDF', 'SBERT']
                ).properties(width=300, height=300)
                st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.write("### 📥 생기부 텍스트 추출본 다운로드")
    st.download_button(
        label="생기부_추출본.txt 다운로드",
        data=cleaned_text,
        file_name="생기부_추출본.txt",
        mime="text/plain"
    )

    st.divider()
    if st.button("🏠 다시 시작하기"):
        st.session_state.page = 'home'
        st.session_state.analyzed = False