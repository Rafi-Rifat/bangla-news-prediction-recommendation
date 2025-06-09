import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Bangla News Prediction & Recommendation",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.prediction-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.recommendation-box {
    background-color: #f9f9f9;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ff7f0e;
    margin: 0.5rem 0;
}
.metric-container {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📰 বাংলা নিউজ ক্যাটেগরি প্রেডিকশন ও রেকমেন্ডেশন সিস্টেম</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #666;">Bangla News Category Prediction & Recommendation System</h2>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ System Information")

@st.cache_data
def load_models_and_data():
    """Load trained models and preprocessed data"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        df = pd.read_csv('cleaned_bangla_news.csv')
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, vectorizer, df, model_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def clean_bangla_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u0980-\u09FF\s।,;:!?]', ' ', text)
    return text.strip()

class NewsRecommendationSystem:
    def __init__(self, vectorizer, model, df):
        self.vectorizer = vectorizer
        self.model = model
        self.df = df
        self.tfidf_matrix = self.vectorizer.transform(df['text_combined'])

    def predict_category(self, text):
        clean_text = clean_bangla_text(text)
        if not clean_text.strip():
            return None, None
        tfidf_input = self.vectorizer.transform([clean_text])
        pred = self.model.predict(tfidf_input)[0]
        prob_dict = None
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(tfidf_input)[0]
            prob_dict = dict(zip(self.model.classes_, probs))
        return pred, prob_dict

    def recommend_similar_news(self, text, top_k=5):
        clean_text = clean_bangla_text(text)
        if not clean_text.strip():
            return []
        tfidf_input = self.vectorizer.transform([clean_text])
        similarities = cosine_similarity(tfidf_input, self.tfidf_matrix).flatten()
        similar_indices = similarities.argsort()[::-1]
        recs = []
        for idx in similar_indices:
            if len(recs) >= top_k:
                break
            if 0.1 < similarities[idx] < 0.99:
                row = self.df.iloc[idx]
                recs.append({
                    'title': row['title'],
                    'category': row['category'],
                    'similarity': similarities[idx],
                    'content': row['content'][:200] + "..." if len(row['content']) > 200 else row['content'],
                    'url': row['url'] if 'url' in row else None
                })
        return recs

# Load resources
model, vectorizer, df, model_info = load_models_and_data()

if model is not None:
    recommender = NewsRecommendationSystem(vectorizer, model, df)

    st.sidebar.success("✅ Models loaded successfully!")
    st.sidebar.write(f"**Model:** {model_info['best_model_name']}")
    st.sidebar.write(f"**Accuracy:** {model_info['accuracy']:.4f}")
    st.sidebar.write(f"**Categories:** {len(model_info['categories'])}")
    # st.sidebar.write(f"**Features:** {model_info['feature_count']:,}")

    st.sidebar.markdown("### 📊 Dataset Stats")
    st.sidebar.write(f"**Total News:** {len(df):,}")
    st.sidebar.write(f"**Categories:** {df['category'].nunique()}")

    for cat, count in df['category'].value_counts().head().items():
        st.sidebar.write(f"• {cat}: {count}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📰 Sample News", "📈 Analytics"])

    with tab1:
        st.markdown('<h2 class="sub-header">নিউজ ক্যাটেগরি প্রেডিকশন</h2>', unsafe_allow_html=True)

        input_method = st.radio("Input Method:", ["✍️ Type your own news", "📝 Select sample news"], horizontal=True)

        if input_method == "✍️ Type your own news":
            user_input = st.text_area("Enter Bengali news text:", height=150,
                                      placeholder="এখানে বাংলা নিউজ লিখুন...")
        else:
            sample_index = st.selectbox("Select a sample:", options=range(min(20, len(df))),
                                        format_func=lambda i: df.iloc[i]['title'][:80] + "...")
            user_input = df.iloc[sample_index]['text_combined']
            st.info(f"**Selected News:** {df.iloc[sample_index]['title']}")

        if st.button("🔍 Predict Category & Recommend", type="primary"):
            if user_input.strip():
                with st.spinner("Processing..."):
                    category, probs = recommender.predict_category(user_input)
                    if category:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.markdown(f"### 🎯 Predicted Category")
                            st.markdown(f"<h2 style='text-align:center; color:#1f77b4'>{category}</h2>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with col2:
                            if probs:
                                st.markdown("### 📊 Probabilities")
                                for c, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
                                    st.progress(p, text=f"{c}: {p:.3f}")

                        recs = recommender.recommend_similar_news(user_input)
                        if recs:
                            st.markdown('<h2 class="sub-header">📚 Recommendations</h2>', unsafe_allow_html=True)
                            for i, r in enumerate(recs, 1):
                                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**{i}. {r['title']}**")
                                    st.write(f"{r['content']}")
                                    if r['url']:
                                        st.markdown(f"🔗 [Full Article]({r['url']})")
                                with col2:
                                    st.write(f"**Category:** {r['category']}")
                                    st.write(f"**Similarity:** {r['similarity']:.3f}")
                                    st.progress(r['similarity'], text=f"{r['similarity'] * 100:.1f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("No similar news found.")
                    else:
                        st.error("Prediction failed. Please enter valid Bangla news.")
            else:
                st.warning("Please input some text.")

    with tab2:
        st.markdown('<h2 class="sub-header">📰 Sample News Articles</h2>', unsafe_allow_html=True)

        cat_filter = st.selectbox("Filter by category:", ['All'] + sorted(df['category'].unique()))
        display_df = df if cat_filter == 'All' else df[df['category'] == cat_filter]

        for _, row in display_df.head(10).iterrows():
            with st.expander(row['title']):
                st.write(f"**Content:** {row['content'][:300]}...")
                st.write(f"**Category:** {row['category']}")
                if 'url' in row and pd.notna(row['url']):
                    st.markdown(f"[Read Full Article]({row['url']})")

    with tab3:
        st.markdown('<h2 class="sub-header">📈 Dataset Analytics</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Category Distribution")
            st.bar_chart(df['category'].value_counts())
        with col2:
            if 'text_length' in df:
                st.markdown("### Text Length Distribution")
                # st.histogram_chart(df['text_length'], bins=30)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Articles", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Categories", df['category'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            if 'text_length' in df:
                avg_len = df['text_length'].mean()
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Avg Length", f"{avg_len:.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            if 'word_count' in df:
                avg_words = df['word_count'].mean()
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Avg Words", f"{avg_words:.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("❌ Models not loaded. Please ensure all `.pkl` and `.csv` files are in the directory.")
