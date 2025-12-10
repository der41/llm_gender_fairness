import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Configuration ---
st.set_page_config(layout="wide", page_title="LLM Gender Fairness Explorer")

# --- Helper Functions ---
@st.cache_data
def load_main_data():
    """Loads the combined embeddings and paragraphs file."""
    # Adjust path if necessary based on where you run the app
    file_path = os.path.join("results", "data", "embeddings", "all_professions_with_gender_scores.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def load_specific_data(profession, file_type):
    """Loads specific csv files for a profession (judge scores, adj freq, etc)."""
    # file_type options: 'judge_scores', 'adjectives_freq', 'gender_freq'
    file_path = os.path.join("results", "data", f"{profession}_{file_type}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=60)
st.sidebar.title("fairness-lens")
st.sidebar.markdown("**Pipeline Explorer**")

# Load Main Data
df = load_main_data()

if df is not None:
    # Profession Selector
    all_professions = df['profession'].unique().tolist()
    selected_profession = st.sidebar.selectbox("Select Profession", sorted(all_professions))
    
    # Filter data for selected profession
    prof_df = df[df['profession'] == selected_profession]
else:
    st.error("Could not find 'all_professions_with_gender_scores.csv'. Please check the file path.")
    st.stop()

# --- Navigation ---
page = st.sidebar.radio("Go to", ["Project Overview", "Bias Analysis (LLM Judge)", "Language Patterns", "Embedding Distances", "Raw Data Explorer"])

# --- Pages ---

# 1. Project Overview
if page == "Project Overview":
    st.title(" 🕵️‍♂️ How AI Imagines Professions")
    st.markdown("### Bias & Stereotype Analysis Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("This project evaluates gender bias and stereotypical framing in Large Language Models (specifically Qwen 1.7B) using NLP-based explainability methods.")
        
        st.markdown("""
        #### 📦 The Methodology
        Based on the project proposal, the pipeline follows these steps:
        1.  **Text Generation**: 1,000 descriptions per job generated using Qwen 1.7B and ChatGPT-5 prompt variations.
        2.  **Adjective Extraction**: Identifying dominant traits (e.g., *Analytical* for Accountants, *Creative* for Artists).
        3.  **Gender Detection**: Measuring frequency of male/female/neutral pronouns.
        4.  **LLM-as-a-Judge**: Using **Gemini 2.5** to score and explain bias in the generated text.
        
        #### 🎯 Goal
        To provide a measurable, visual, and explainable way to understand how AI models might reinforce or reshape occupational stereotypes.
        """)

    with col2:
        st.markdown("### 📊 Summary Stats")
        total_samples = len(df)
        unique_jobs = df['profession'].nunique()
        st.metric("Total Samples Generated", total_samples)
        st.metric("Professions Analyzed", unique_jobs)
        st.metric("Target Model", "Qwen 1.7B")
        st.metric("Judge Model", "Gemini 2.5")

# 2. Bias Analysis (LLM Judge)
elif page == "Bias Analysis (LLM Judge)":
    st.title(f"⚖️ Gemini 2.5 Judge: {selected_profession.title()}")
    
    judge_df = load_specific_data(selected_profession, "judge_scores")
    
    if judge_df is not None:
        # Calculate average scores
        avg_male = judge_df['male_bias_score'].mean()
        avg_female = judge_df['female_bias_score'].mean()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Male Bias Score", f"{avg_male:.2f}")
        c2.metric("Avg Female Bias Score", f"{avg_female:.2f}")
        
        # Determine Bias Direction based on scores
        if avg_male > avg_female + 5:
            bias_status = "Leans Masculine"
            color = "red"
        elif avg_female > avg_male + 5:
            bias_status = "Leans Feminine"
            color = "blue"
        else:
            bias_status = "Neutral / Balanced"
            color = "green"
            
        c3.markdown(f"**Verdict:** :{color}[{bias_status}]")
        
        st.subheader("Judge Explanations")
        st.write("Below are specific samples where Gemini 2.5 analyzed the text for bias:")
        
        for index, row in judge_df.head(5).iterrows():
            with st.expander(f"Sample Group {row.get('group_id', index)} Analysis"):
                st.markdown(f"**Explanation:** {row['explanation']}")
                st.markdown(f"**Scores:** Male: `{row['male_bias_score']}` | Female: `{row['female_bias_score']}`")
    else:
        st.warning(f"No Judge Scores file found for {selected_profession} (checked for `{selected_profession}_judge_scores.csv`).")

# 3. Language Patterns
elif page == "Language Patterns":
    st.title(f"🗣️ How is a {selected_profession.title()} described?")
    
    # 3.1 Gender Frequency from Main DF
    st.subheader("Gender Representation in Text")
    gender_counts = prof_df['gender_label'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    
    fig_gender = px.pie(gender_counts, values='Count', names='Gender', 
                        title=f'Gender Frequency in {selected_profession.title()} Descriptions',
                        color='Gender',
                        color_discrete_map={'male':'#636EFA', 'female':'#EF553B', 'non-gender':'#00CC96'})
    st.plotly_chart(fig_gender, use_container_width=True)

    # 3.2 Adjective Frequency
    st.subheader("Dominant Adjectives")
    adj_df = load_specific_data(selected_profession, "adjectives_freq")
    
    col1, col2 = st.columns([1, 1])
    
    if adj_df is not None:
        with col1:
            # Bar Chart
            top_adj = adj_df.sort_values(by='count', ascending=False).head(15)
            fig_adj = px.bar(top_adj, x='count', y='adjective', orientation='h', 
                             title="Top 15 Adjectives Used",
                             color='count', color_continuous_scale='Viridis')
            fig_adj.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_adj, use_container_width=True)
            
        with col2:
            # Word Cloud
            st.markdown("##### Word Cloud Visualization")
            text_dict = dict(zip(adj_df['adjective'], adj_df['count']))
            if text_dict:
                wc = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(text_dict)
                plt.figure(figsize=(8, 8))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("Not enough data for word cloud.")
    else:
        st.warning("Adjective frequency data not found.")

# 4. Embedding Distances
elif page == "Embedding Distances":
    st.title("📏 Semantic Distance & Stereotypes")
    st.markdown("""
    This section visualizes the **Gender Score**, which represents the semantic distance between the generated descriptions and gendered concepts in the embedding space.
    * **Higher Score**: Indicates strong semantic associations with gendered attributes.
    * **Lower Score**: Indicates more neutral descriptions.
    """)
    
    # Histogram of gender scores
    fig_hist = px.histogram(prof_df, x="gender_score", color="gender_label", 
                            marginal="box", 
                            title=f"Distribution of Gender Scores for {selected_profession.title()}",
                            nbins=30,
                            color_discrete_map={'male':'#636EFA', 'female':'#EF553B', 'non-gender':'#00CC96'})
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Scatter plot vs Adjective Count (if available)
    # Approximate adjective count by counting commas in the 'adjectives' column
    prof_df['adj_count'] = prof_df['adjectives'].apply(lambda x: str(x).count(',') + 1 if pd.notnull(x) else 0)
    
    fig_scatter = px.scatter(prof_df, x="gender_score", y="adj_count", 
                             color="gender_label", hover_data=["paragraph"],
                             title="Gender Score vs. Complexity (Adjective Count)",
                             labels={"adj_count": "Number of Adjectives used"})
    st.plotly_chart(fig_scatter, use_container_width=True)

# 5. Raw Data Explorer
elif page == "Raw Data Explorer":
    st.title("📂 Data Explorer")
    
    st.markdown("Filter and view the raw text generations produced by Qwen 1.7B.")
    
    # Filters
    filter_gender = st.multiselect("Filter by Gender Label", prof_df['gender_label'].unique(), default=prof_df['gender_label'].unique())
    
    filtered_df = prof_df[prof_df['gender_label'].isin(filter_gender)]
    
    st.dataframe(filtered_df[['gender_label', 'question', 'paragraph', 'gender_score', 'adjectives']], use_container_width=True)
    
    st.markdown("### Detailed View")
    # Selection for detailed view
    sample_id = st.number_input("Enter Sample ID (index) to view details:", min_value=0, max_value=len(filtered_df)-1, value=0, step=1)
    
    if len(filtered_df) > 0:
        record = filtered_df.iloc[sample_id]
        st.markdown(f"**Prompt:** {record['question']}")
        st.info(f"**Generated Text:** {record['paragraph']}")
        st.json({
            "Gender Label": record['gender_label'],
            "Bias/Gender Score": record['gender_score'],
            "Adjectives Extracted": record['adjectives']
        })

# --- Footer ---
st.markdown("---")
st.markdown("*LLM Gender Fairness Pipeline | Based on the proposal 'How AI Imagine Professions?'*")