"""
Streamlit Web Interface for Document Classification
Run with: streamlit run app.py --server.port 3000
"""

import streamlit as st
import pandas as pd
from azure_openai_classifier import LLMClassifier
import time
from io import StringIO

# Page config
st.set_page_config(
    page_title="NassaQ Document Classifier",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .confidence-high { background-color: #d4edda; color: #155724; }
    .confidence-medium { background-color: #fff3cd; color: #856404; }
    .confidence-low { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📄 NassaQ Document Classifier</h1>', unsafe_allow_html=True)
st.markdown("**Powered by GPT-4o mini on Azure OpenAI**")
st.markdown("---")

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.info("**Provider:** Azure OpenAI")
    st.success("**Model:** GPT-4o mini")
    
    if st.button("🔄 Initialize Classifier"):
        with st.spinner("Initializing classifier..."):
            try:
                st.session_state.classifier = LLMClassifier(provider="azure")
                st.success(f"✅ Initialized: {st.session_state.classifier.deployment_name}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    st.header("📊 Categories")
    st.markdown("""
    - 🎨 **Culture**: Arts, heritage, literature
    - 💰 **Finance**: Economy, stocks, banking
    - 🏥 **Medical**: Health, medicine, hospitals
    - 🏛️ **Politics**: Government, diplomacy
    - 📿 **Religion**: Religious topics
    - ⚽ **Sports**: Games, tournaments
    - 💻 **Technology**: AI, software, innovation
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Single Document", "📚 Batch Upload", "📈 Results"])

# Tab 1: Single Document Classification
with tab1:
    st.header("Classify Single Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter document text (Arabic or English):",
            height=200,
            placeholder="مثال: ارتفعت أسعار النفط في الأسواق العالمية...\nExample: The Federal Reserve announced..."
        )
    
    with col2:
        st.markdown("**Examples:**")
        if st.button("📰 Finance Example"):
            st.session_state.example_text = "ارتفعت أسعار النفط في الأسواق العالمية بنسبة 4٪ بعد اجتماع أوبك الأخير"
        if st.button("⚽ Sports Example"):
            st.session_state.example_text = "المنتخب السعودي يفوز على نظيره الإماراتي بثلاثة أهداف مقابل هدف واحد"
        if st.button("🏥 Medical Example"):
            st.session_state.example_text = "The new AI model can diagnose skin cancer with 95% accuracy using smartphone cameras"
    
    # Use example text if set
    if 'example_text' in st.session_state and st.session_state.example_text:
        text_input = st.text_area(
            "Enter document text (Arabic or English):",
            value=st.session_state.example_text,
            height=200,
            key="text_area_with_example"
        )
        st.session_state.example_text = None  # Clear after use
    
    if st.button("🚀 Classify", type="primary", disabled=(st.session_state.classifier is None)):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to classify")
        else:
            with st.spinner("Classifying..."):
                result = st.session_state.classifier.classify(text_input)
                
                if result.error:
                    st.error(f"❌ Error: {result.error}")
                else:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Category", result.category)
                    with col2:
                        confidence_pct = result.confidence * 100
                        st.metric("Confidence", f"{confidence_pct:.1f}%")
                    with col3:
                        st.metric("Cost", f"${result.cost_usd:.4f}")
                    
                    st.markdown("---")
                    st.subheader("Reasoning")
                    st.info(result.reasoning)
                    
                    # Confidence indicator
                    if result.confidence >= 0.8:
                        st.success("✅ High confidence classification")
                    elif result.confidence >= 0.6:
                        st.warning("⚠️ Medium confidence - review recommended")
                    else:
                        st.error("❌ Low confidence - manual review required")

# Tab 2: Batch Upload
with tab2:
    st.header("Batch Classification")
    
    st.markdown("""
    Upload a CSV file with a column containing document texts.
    The classifier will process all documents and return results.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        
        # Column selection
        text_column = st.selectbox(
            "Select text column:",
            df.columns.tolist()
        )
        
        max_docs = st.slider("Maximum documents to process:", 1, min(len(df), 100), 10)
        
        if st.button("🚀 Process Batch", type="primary", disabled=(st.session_state.classifier is None)):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            texts = df[text_column].head(max_docs).tolist()
            
            for i, text in enumerate(texts):
                status_text.text(f"Processing document {i+1}/{len(texts)}...")
                result = st.session_state.classifier.classify(str(text))
                
                results.append({
                    'text': text[:100] + "..." if len(str(text)) > 100 else text,
                    'category': result.category,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'tokens_used': result.tokens_used,
                    'cost_usd': result.cost_usd
                })
                
                progress_bar.progress((i + 1) / len(texts))
                time.sleep(0.5)  # Rate limiting
            
            st.session_state.results = pd.DataFrame(results)
            status_text.text("✅ Processing complete!")
            st.success(f"Processed {len(results)} documents")
            
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Docs", len(results))
            with col2:
                st.metric("Total Tokens", f"{st.session_state.results['tokens_used'].sum():,}")
            with col3:
                st.metric("Total Cost", f"${st.session_state.results['cost_usd'].sum():.4f}")

# Tab 3: Results
with tab3:
    st.header("Classification Results")
    
    if st.session_state.results is not None:
        df_results = st.session_state.results
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", len(df_results))
        with col2:
            st.metric("Avg Confidence", f"{df_results['confidence'].mean():.1%}")
        with col3:
            st.metric("Total Tokens", f"{df_results['tokens_used'].sum():,}")
        with col4:
            st.metric("Total Cost", f"${df_results['cost_usd'].sum():.4f}")
        
        st.markdown("---")
        
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Distribution")
            category_counts = df_results['category'].value_counts()
            st.bar_chart(category_counts)
        
        with col2:
            st.subheader("Confidence Distribution")
            st.bar_chart(df_results['confidence'])
        
        st.markdown("---")
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        # Add confidence color coding
        def confidence_color(val):
            if val >= 0.8:
                return 'background-color: #d4edda'
            elif val >= 0.6:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        
        styled_df = df_results.style.applymap(
            confidence_color, 
            subset=['confidence']
        ).format({
            'confidence': '{:.1%}',
            'cost_usd': '${:.4f}',
            'tokens_used': '{:,}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download button
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="classification_results.csv",
            mime="text/csv"
        )
        
        # Filter options
        st.markdown("---")
        st.subheader("🔍 Filter Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_categories = st.multiselect(
                "Filter by category:",
                df_results['category'].unique().tolist(),
                default=df_results['category'].unique().tolist()
            )
        
        with col2:
            min_confidence = st.slider(
                "Minimum confidence:",
                0.0, 1.0, 0.0, 0.1
            )
        
        # Apply filters
        filtered_df = df_results[
            (df_results['category'].isin(selected_categories)) &
            (df_results['confidence'] >= min_confidence)
        ]
        
        st.write(f"Showing {len(filtered_df)} of {len(df_results)} documents")
        st.dataframe(filtered_df, use_container_width=True)
        
    else:
        st.info("👆 Process some documents in the 'Batch Upload' tab to see results here")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>NassaQ Document Classification System</strong></p>
    <p>Graduation Project 2026 | Powered by LLM Few-Shot Learning</p>
</div>
""", unsafe_allow_html=True)
