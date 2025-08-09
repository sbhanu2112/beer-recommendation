import streamlit as st
import pandas as pd
from beer_recommender import BeerRecommender
import base64

# Page config
st.set_page_config(
    page_title="Beer Buddy",
    page_icon="🍺",
    layout="centered"
)

def get_base64_image(image_path):
    """Convert local image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Load background image if it exists
bg_image = get_base64_image("background.jpg")

# Clean and simple styling with background
if bg_image:
    background_css = f"""
        background-image: 
            linear-gradient(
                rgba(255, 255, 255, 0.75),
                rgba(255, 255, 255, 0.70)
            ),
            url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    """
else:
    background_css = "background-color: #ffffff;"

st.markdown(f"""
<style>
    /* Background with image */
    .stApp {{
        {background_css}
    }}
    
    /* Terminal output style - more opaque for better readability */
    .terminal-output {{
        background-color: rgba(248, 249, 250, 0.98);
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 14px;
        line-height: 1.6;
        white-space: pre-wrap;
        color: #212529;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Simple button styling */
    .stButton > button {{
        background-color: #007bff;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .stButton > button:hover {{
        background-color: #0056b3;
    }}
    
    /* Clean input styling with better visibility */
    .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #ced4da;
        border-radius: 4px;
    }}
    
    /* Title styling with background for readability */
    h1 {{
        color: #212529;
        font-weight: 600;
        border-bottom: 2px solid #dee2e6;
        padding: 15px;
        margin-bottom: 20px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Make expander more visible */
    .streamlit-expanderHeader {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 4px;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Load and initialize the beer recommender system"""
    recommender = BeerRecommender()
    recommender.load_and_preprocess_data()
    recommender.train_regression_model()
    return recommender

def format_terminal_output(prompt, predicted_rating, recommendations, alt_recommendations=None):
    """Format output exactly like beer_expected.ipynb"""
    
    output = f"User Prompt = {prompt}\n"
    
    if alt_recommendations is not None:
        # Low rating warning
        output += "━" * 60 + "\n"
        output += f"⚠️  Warning: This flavor combination typically rates {predicted_rating:.2f}/5\n"
        output += "━" * 60 + "\n\n"
        
        output += "📍 Here's what matches your exact request:\n"
        if recommendations:
            for i, beer in enumerate(recommendations[:2], 1):
                output += f"{i}. {beer['name']} ({beer['rating']:.2f}★ - {int(beer['num_reviews'])} reviews)\n"
                output += f"   Distance: {beer['distance']:.3f}\n"
        else:
            output += "   No exact matches found in our database.\n"
        
        output += "\n💡 Suggested Alternatives (similar but better rated):\n"
        if alt_recommendations:
            for i, beer in enumerate(alt_recommendations[:2], 1):
                output += f"{i}. {beer['name']} ({beer['rating']:.2f}★ - {int(beer['num_reviews'])} reviews)\n"
                output += f"   Distance: {beer['distance']:.3f}\n"
        else:
            output += "   No high-rated alternatives found with your criteria.\n"
        
        output += "\n💭 Tip: The flavor combination you requested is uncommon. The alternatives above\n"
        output += "   maintain similar characteristics but with proven appeal to beer enthusiasts.\n"
        
    else:
        # Good rating
        output += "━" * 60 + "\n"
        output += f"✅ Great choice! Predicted rating: {predicted_rating:.2f}/5\n"
        output += "━" * 60 + "\n\n"
        
        output += "🍺 Top Recommendations:\n"
        for i, beer in enumerate(recommendations[:2], 1):
            output += f"\n{i}. {beer['name']}\n"
            output += f"   Rating: {beer['rating']:.2f}/5 ({int(beer['num_reviews'])} reviews)\n"
            output += f"   Distance: {beer['distance']:.3f}\n"
            
            desc = beer.get('description', '')
            if desc:
                desc = desc[:120] + "..." if len(desc) > 120 else desc
                output += f"   Notes: Notes:{desc}\n"
            else:
                output += "   Notes: Notes:...\n"
    
    output += "\n" + "─" * 60 + "\n"
    output += "=" * 133
    
    return output

def main():
    # st.title("🍺 Beer Buddy 🍺")
    st.markdown(
    """
    <h1 style='text-align: center;'>🍺 Beer Buddy 🍺</h1>
    """,
    unsafe_allow_html=True
)
    
    # Load recommender
    with st.spinner("Loading beer database..."):
        recommender = load_recommender()
    
    # Description
    # st.markdown("Enter your beer preference to get personalized recommendations.")
    
    # Initialize session state for selected query
    if 'selected_query' not in st.session_state:
        st.session_state.selected_query = ""
    
    # Input section - now connected to session state
    user_input = st.text_input(
        "**Tell us what you\'re craving and we\'ll find your perfect beer**",
        value=st.session_state.selected_query,  # Use session state value
        placeholder="e.g., I want a hoppy IPA with tropical notes",
        help="Describe the type of beer you're looking for",
        key="beer_input"
    )
    
    # Update session state when user types
    if user_input != st.session_state.selected_query:
        st.session_state.selected_query = user_input
    
    # Example queries
    st.markdown("**Can't make your mind? Try these:**")
    
    examples = [
        "I want a light 🍊 citrusy beer",
        "Give me a 🌺 hoppy IPA with tropical notes",
        "I want a sessionable pilsner",
        "Something 🍋 sour and funky with brett character",
        "Just a Bad beer"
    ]
    
    # Create buttons in columns
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.selected_query = example  # Update session state
                st.rerun()
    
    # Get recommendations button
    if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
        if user_input:
            with st.spinner("Analyzing your request..."):
                try:
                    # Get recommendations
                    results = recommender.get_recommendations(user_input)
                    
                    # Format output
                    terminal_output = format_terminal_output(
                        user_input,
                        results['predicted_rating'],
                        results['recommendations'],
                        results.get('alt_recommendations')
                    )
                    
                    # Display in terminal style
                    st.markdown(
                        f'<div class="terminal-output">{terminal_output}</div>',
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please check if the GROQ_API_KEY is set in your .env file")
        else:
            st.warning("Please enter a beer preference!")
    
    # Info section
    with st.expander("About this system"):
        st.markdown("""
        This recommendation system uses:
        - **Natural Language Processing** via LLM to understand your request
        - **Gradient Boosting** to predict beer ratings
        - **K-Nearest Neighbors** to find similar beers
        - **Quality Scoring** based on ratings and review counts
        
        The system analyzes **3,197 craft beers** and will warn you if your requested 
        combination might not taste good (predicted rating < 3.0/5).
        """)

if __name__ == "__main__":
    main()
