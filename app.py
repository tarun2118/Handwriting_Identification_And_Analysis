import streamlit as st
from PIL import Image
import tempfile
from pipeline import process_image


# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Handwritten Notes Processor",
    page_icon="🧠",
    layout="wide"
)


# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------

st.markdown("""
<style>

/* Background */

.stApp {
background: radial-gradient(circle at top,#0f2027,#090d13,#000000);
color:white;
}

/* Floating shapes */

.shape1 {
position: fixed;
top: 10%;
left: 5%;
width:120px;
height:120px;
background: linear-gradient(45deg,#00dbde,#fc00ff);
border-radius: 30%;
opacity:0.2;
animation: float 8s infinite;
}

.shape2 {
position: fixed;
bottom: 10%;
right: 10%;
width:150px;
height:150px;
background: linear-gradient(45deg,#ff512f,#dd2476);
border-radius: 50%;
opacity:0.15;
animation: float 10s infinite;
}

.shape3 {
position: fixed;
top:60%;
left:40%;
width:90px;
height:90px;
background: linear-gradient(45deg,#36d1dc,#5b86e5);
border-radius: 20%;
opacity:0.15;
animation: float 12s infinite;
}

@keyframes float {
0%{transform:translateY(0px)}
50%{transform:translateY(-20px)}
100%{transform:translateY(0px)}
}


/* Animated Title */

.title {
font-size:55px;
font-weight:900;
text-align:center;
background: linear-gradient(90deg,#00dbde,#fc00ff,#00dbde);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
animation: shimmer 5s infinite linear;
}

@keyframes shimmer {
0%{background-position:0%}
100%{background-position:200%}
}


/* Glass cards */

.card {
padding:30px;
border-radius:20px;
background: rgba(255,255,255,0.05);
backdrop-filter: blur(20px);
border:1px solid rgba(255,255,255,0.1);
box-shadow:0 0 20px rgba(0,0,0,0.5);
margin-top:10px;
transition:0.4s;
}

.card:hover{
transform:scale(1.02);
box-shadow:0 0 25px rgba(255,0,200,0.5);
}


/* Upload box */

[data-testid="stFileUploader"]{
background:rgba(255,255,255,0.04);
border:2px dashed #888;
border-radius:20px;
padding:25px;
}


/* Image style */

img{
border-radius:20px;
box-shadow:0 0 25px rgba(0,0,0,0.6);
transition:0.4s;
}

img:hover{
transform:scale(1.03);
}


/* Section titles */

.section {
font-size:30px;
font-weight:700;
margin-top:30px;
color:#00dbde;
}


/* Download button */

.stDownloadButton button{
background: linear-gradient(45deg,#ff512f,#dd2476);
border:none;
border-radius:30px;
padding:12px 28px;
font-size:16px;
font-weight:bold;
color:white;
transition:0.3s;
}

.stDownloadButton button:hover{
transform:scale(1.08);
box-shadow:0 0 20px #ff4b8a;
}

</style>

<div class="shape1"></div>
<div class="shape2"></div>
<div class="shape3"></div>

""", unsafe_allow_html=True)



# -----------------------------------------------------
# HEADER
# -----------------------------------------------------

st.markdown("<div class='title'>🧠 Writer Identification and HandWritten Text Processing using Deep Neural Networks</div>", unsafe_allow_html=True)

st.markdown(
"""
<center>

Upload a handwritten notes image  

✔ Identify the writer  
✔ Extract text using OCR  
✔ Generate enhanced structured notes  

</center>
""",
unsafe_allow_html=True
)

# -----------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------

uploaded_file = st.file_uploader(
"Upload handwritten page",
type=["png","jpg","jpeg"]
)



# -----------------------------------------------------
# MAIN PROCESS
# -----------------------------------------------------

if uploaded_file is not None:

    col1,col2 = st.columns(2)

    with col1:

        st.markdown("<div class='section'>Uploaded Image</div>", unsafe_allow_html=True)

        image = Image.open(uploaded_file)

        st.image(image,use_column_width=True)

        temp = tempfile.NamedTemporaryFile(delete=False,suffix=".png")
        image.save(temp.name)

        image_path = temp.name



    with col2:

        st.markdown("<div class='section'>Processing</div>", unsafe_allow_html=True)

        progress = st.progress(0)

        for i in range(100):
            progress.progress(i+1)

        with st.spinner("Running AI pipeline..."):

            result = process_image(image_path)

        st.success("Processing completed!")

        st.markdown(
        f"""
        <div class="card">

        <h3>Writer Identification</h3>

        Writer: <b>{result['writer']}</b> <br>
        Confidence: <b>{result['confidence']}%</b>

        </div>
        """,
        unsafe_allow_html=True
        )


    st.divider()



    # OCR TEXT

    st.markdown("<div class='section'>Extracted OCR Text</div>", unsafe_allow_html=True)

    st.markdown(
    f"""
    <div class="card">
    <pre>{result["ocr_text"]}</pre>
    </div>
    """,
    unsafe_allow_html=True
    )



    # ENHANCED NOTES

    st.markdown("<div class='section'>Enhanced Study Notes</div>", unsafe_allow_html=True)

    st.markdown(
    f"""
    <div class="card">
    {result["enhanced_notes"]}
    </div>
    """,
    unsafe_allow_html=True
    )



    # DOWNLOAD BUTTON

    st.download_button(
        label="📥 Download Notes",
        data=result["enhanced_notes"],
        file_name="enhanced_notes.txt",
        mime="text/plain"
    )


else:

    st.info("Upload a handwritten page to start.")