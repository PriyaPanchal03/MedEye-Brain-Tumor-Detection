# ------------------ Imports ------------------
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import cv2
#added 
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import textwrap
from datetime import datetime
from chatbot_ui import render_chatbot

#---Added
st.set_page_config(
    page_title="MedEye - Brain Tumor Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ UI THEME ------------------
def set_professional_theme():
    st.markdown("""
    <style>

    /* ===== Remove default padding & width ===== */
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 100%;
    }

    /* ===== Full app background ===== */
    .stApp, body {
        background-color: #eef5ff;
    }

    /* ===== FIX TOP HEADER ===== */
    header[data-testid="stHeader"],
    header[data-testid="stHeader"] * {
        background-color: #eef5ff;
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background-color: #23408e;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* ===== Selectbox container ===== */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 10px;
        border: 2px solid #60a5fa;
        min-height: 45px;
        display: flex;
        align-items: center;
    }

    /* Selectbox text ("Home") */
    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #0b1f44 !important;
        font-size: 15px;
        font-weight: 500;
    }

    /* Dropdown arrow */
    section[data-testid="stSidebar"] svg {
        fill: #0b1f44 !important;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: #93c5fd;
    }
    /* ===== INPUT FIELDS (FIX FADED LOOK) ===== */
    input,
    textarea {
        background-color: #ffffff ;
        color: #0b1f44 ;
        border: 2px solid #c7d2fe ;
        border-radius: 10px ;
        font-size: 15px;
    }

    /* Number input buttons (+ / -) */
    button[aria-label="Increment"],
    button[aria-label="Decrement"] {
        background-color: #e0e7ff;
        color: #1e3a8a;
        border-radius: 6px;
    }

    /* Date input */
    div[data-baseweb="input"] > div {
        background-color: #ffffff;
        border-radius: 10px;
        border: 2px solid #c7d2fe;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #ffffff;
        border-radius: 10px;
        border: 2px solid #c7d2fe;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 2px dashed #93c5fd;
        padding: 1rem;
    }

    /* Hover + focus effect */
    # input:focus,
    # textarea:focus,
    # div[data-baseweb="select"] > div:focus-within {
    #     border-color: #2563eb;
    #     box-shadow: 0 0 0 1px #2563eb;
    # }

    </style>
    """, unsafe_allow_html=True)

# ------------------ Constants ------------------
IMG_SIZE = (224, 224)
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
CONFIDENCE_THRESHOLD = 70  # minimum confidence %

recommendations = {
    'glioma_tumor': """🧠 **Recommended Actions:**\n
    - Consult a neurologist or neuro-oncologist immediately.
    - Request a contrast-enhanced MRI for confirmation.
    - Discuss treatment options such as surgery, radiotherapy, or chemotherapy depending on grade.
    - Maintain a record of symptoms and report any worsening headache, seizures, or vision issues.""",

    'meningioma_tumor': """🧠 **Recommended Actions:**\n
    - Schedule an appointment with a neurosurgeon.
    - Consider regular MRI scans every 6–12 months to monitor growth.
    - Most meningiomas are benign, but surgical removal might be advised if the tumor is large or symptomatic.
    - Maintain a healthy diet and lifestyle to reduce recurrence risks.""",

    'pituitary_tumor': """🧠 **Recommended Actions:** \n
    - Visit an endocrinologist to check for hormonal imbalance.
    - Get a pituitary MRI with contrast for detailed analysis.
    - Observe for symptoms like vision issues, fatigue, or mood changes.
    - Medication or surgery might be required based on the tumor’s hormonal activity.""",

    'no_tumor': """✅ **Great News:** \n
    - No tumor detected in this MRI image.
    - Continue regular check-ups if you have symptoms.
    - Maintain a balanced lifestyle and avoid stress to support brain health."""
}


# ------------------ Load Model ------------------
@st.cache_resource
def load_trained_model():
    model = load_model(r"D:\MedEye\trial_model.h5")  # update path
    return model

model = load_trained_model()

# ------------------ Grad-CAM Function ------------------
def get_gradcam(img_array, model, last_conv_layer_name='Conv_1'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)  # convert to numpy safe
    heatmap = cv2.resize(heatmap, (IMG_SIZE[0], IMG_SIZE[1]))       # remove .numpy()
    return heatmap

# --- NEW ADDITION: Tumor spatial location estimation ---
def estimate_location(heatmap):
    h, w = heatmap.shape
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    vertical = "Upper" if y < h/2 else "Lower"
    horizontal = "Left" if x < w/2 else "Right"

    return f"{vertical}-{horizontal} region"

# --- NEW ADDITION: Tumor size estimation ---
def estimate_tumor_size(heatmap, threshold=0.5):
    binary = heatmap > threshold
    tumor_pixels = np.sum(binary)
    total_pixels = heatmap.size
    return round((tumor_pixels / total_pixels) * 100, 2)

# --- NEW ADDITON: Risk assessment ---
def risk_level(predicted_class, confidence, size):
    if predicted_class == "no_tumor":
        return "Low Risk -- No Tumor"

    if confidence > 85 and size > 8:
        return "High Risk -- Consult to specialist immediately for further diagnostic tests and personalized care planning."

    return "Moderate -- Consult with a healthcare professional"



# ------------------ Prediction Function ------------------
def model_prediction(image_data):
    img = Image.open(image_data).convert('RGB')
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized) / 255.0  #normalizing img (0-1)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    confidence = round(100 * np.max(predictions[0]), 3)
    predicted_class = labels[np.argmax(predictions[0])]

    gradcam_heatmap = get_gradcam(img_array, model)
    heatmap_img = cv2.applyColorMap(np.uint8(255 *(1-gradcam_heatmap)), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_img, 0.4, 0)
    superimposed_img = Image.fromarray(superimposed_img)

    location= estimate_location(gradcam_heatmap) #new
    size= estimate_tumor_size(gradcam_heatmap)#new


    return predicted_class, confidence, superimposed_img, location, size

#-- pdf generater 
def generate_pdf_report(predicted_class, confidence, gradcam_path, original_image_path,
                        patient_name, patient_id, age, gender, scan_date, location,size,risk):
    pdf_path = "MedEye_Brain_Tumor_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    left_margin   = 50
    right_margin  = 50
    top_margin    = 50
    bottom_margin = 50
    main_region= width-left_margin-right_margin
    y = height - top_margin  # starting Y position for content

    # ------------------ TITLE ------------------
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString((main_region / 2)+ left_margin, y, "MedEye - Brain Tumor Detection Report")
    y-=30
    # ------------------ DATE ------------------
    c.setFont("Helvetica", 10)
    c.drawString(main_region/2 + 120, y, f"Report Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    y-=25

    # ------------------ PATIENT INFORMATION ------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Patient Information:")
    y -= 20

    c.setFont("Helvetica", 11)
    c.drawString(left_margin+20, y, f"Name: {patient_name}")
    y -= 18
    c.drawString(left_margin+20, y, f"Patient ID: {patient_id}")
    y -= 18
    c.drawString(left_margin+20, y, f"Age: {age}")
    y -= 18
    c.drawString(left_margin+20, y, f"Gender: {gender}")
    y -= 18
    c.drawString(left_margin+20, y, f"Scan Date: {scan_date}")
    y -= 30

    # ------------------ PREDICTION SUMMARY ------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Prediction Summary:")
    y -= 20

    c.setFont("Helvetica", 11)
    c.drawString(left_margin+20, y, f"Tumor Type: {predicted_class}")
    y -= 18
    c.drawString(left_margin+20, y, f"Confidence Level: {confidence:.3f}%")
    y -= 18
    if predicted_class=="no_tumor":
        c.drawString(left_margin+20, y, f"Predicted Location: NIL") #
        y -= 18
        c.drawString(left_margin+20, y, f"Relative Size of Tumor: NIL") #
    else:        
        c.drawString(left_margin+20, y, f"Predicted Location: {location}") #
        y -= 18
        c.drawString(left_margin+20, y, f"Relative Size of Tumor: {size}% of scan area") #
        
    y -= 18
    text = c.beginText(left_margin + 20, y)
    text.setFont("Helvetica", 11)
    text.textLine("Clinical Risk: ")
    for line in textwrap.wrap(risk, 90):
        text.textLine(line)
    c.drawText(text)
    y = text.getY() - 20
 
    # ------------------ MEDICAL RECOMMENDATION ------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Medical Recommendation:")
    y -= 20

    c.setFont("Helvetica", 10)
    text = c.beginText(left_margin+20, y)
    for line in recommendations[predicted_class].split("\n"):
        text.textLine(line)
    c.drawText(text)

    # Update y-position after recommendation text
    y = text.getY() - 30

    # ------------------ DIAGNOSTIC VISUAL EVIDENCE ------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Diagnostic Visual Evidence:")
    y -= 20

    c.setFont("Helvetica-Bold", 10)
    c.drawString(left_margin, y, "Original MRI Scan")
    c.drawString(300, y, "Grad-CAM Heatmap (AI Explanation)")
    y -= 210  # space for images

    # Draw Images
    c.drawImage(
        original_image_path,
        left_margin,
        y,
        width=220,
        height=200,
        preserveAspectRatio=True,
        mask='auto'
    )

    c.drawImage(
        gradcam_path,
        300,
        y,
        width=220,
        height=200,
        preserveAspectRatio=True,
        mask='auto'
    )

    # ------------------ DISCLAIMER ------------------
    text = c.beginText(left_margin, bottom_margin)
    text.setFont("Helvetica", 9)
    for line in textwrap.wrap(
        "Disclaimer: This AI-generated analysis and heatmap represents regions influencing the model's prediction and should not be interpreted as exact tumor boundaries, and it does not replace professional medical advice.",
        100):
        text.textLine(line)
    c.drawText(text)


    c.save()
    return pdf_path

#  APPLY THEME 
set_professional_theme()

# ------------------ Sidebar ------------------
st.sidebar.title('MedEye-Brain Tumor Detection')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About Dataset', 'Disease Recognition','MedBot'])

# ------------------ Home Page ------------------
if app_mode == 'Home':
    st.header('MedEye-Seeing Beyond the Surface')
    st.markdown("""
    **MedEye** combines medical imaging and artificial intelligence to provide preliminary assessment of brain tumor with visual explainability.  
        
    Brain tumors are abnormal growths of cells within the brain, categorized into two types: malignant and benign. Malignant tumors can be life-threatening due to their location and growth rate, making timely and accurate detection crucial. 
    
    This system classifies brain MRI images into four categories:
    1. Glioma Tumor
    2. Meningioma Tumor
    3. Pituitary Tumor
    4. No Tumor
                
    This project aims to distinguish between three types of brain tumors and normal cases (i.e., no tumor) based on their location.
    
    ### How It Works
    1. **Upload Image:** Navigate to the **Disease Recognition** page and upload an image of the suspected tumor.
    2. **Analysis:** The system processes the image using advanced algorithms to classify the tumor.
    3. **Results:** View the classification results and recommendations for further action.
    """)

# ------------------ About Dataset ------------------
elif app_mode == 'About Dataset':
    st.header("About the Brain Tumor MRI Dataset")
    st.markdown("""
    The dataset used for this model is sourced from the Brain Tumor MRI Dataset available on Kaggle.
    The dataset contains 7,000+ MRI scans, divided into four classes:
    - Glioma Tumor
    - Meningioma Tumor
    - Pituitary Tumor
    - No Tumor
    
    """)

# ------------------ Disease Recognition ------------------
elif app_mode == 'Disease Recognition':
    st.header("MedEye - Brain Tumor Recognition")
    # new added
    st.subheader("👤 Patient Details")

    patient_name = st.text_input("Patient Name")
    patient_id = st.text_input(
        "Patient ID",
        value=f"MED-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    scan_date = st.date_input("MRI Scan Date", datetime.today())

    #old one--
    uploaded_image = st.file_uploader("Upload Brain MRI Image:", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded MRI Image', width=300)
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                #new added
                 # ------------------ Save Original MRI ------------------
                original_img = Image.open(uploaded_image).convert("RGB")
                original_img_path = "uploaded_mri.png"
                original_img.save(original_img_path)

                # ------------------ Model Prediction ------------------ old one
                predicted_class, confidence, gradcam_img, location, size = model_prediction(uploaded_image)
                 # Store in session state --- new added
                st.session_state.predicted_class = predicted_class
                st.session_state.confidence = confidence
                st.session_state.gradcam_img = gradcam_img
                st.session_state.original_img_path = original_img_path
                st.session_state.prediction_done = True
                st.session_state.predicted_location= location
                st.session_state.predicted_size=size
    
    #-------New Added 27/12-----
        if st.session_state.prediction_done:

                confidence = st.session_state.confidence
                predicted_class = st.session_state.predicted_class
                gradcam_img = st.session_state.gradcam_img
                location= st.session_state.predicted_location
                size=st.session_state.predicted_size
                risk=risk_level(predicted_class, confidence, size)
                #----old one not to delete ----
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f"Prediction uncertain ({round(confidence,3)}%). Consult a medical expert.")
                else:
                    st.success(f"Predicted: {predicted_class} with {confidence:.3f}% confidence")
                    # 🩺 Show the recommendation
                    st.image(gradcam_img, caption='Grad-CAM Heatmap', width=300)
                    if predicted_class=="no_tumor":
                        st.info(f"Estimated Tumor Location: NIL")
                        st.info(f"Model detection for tumor: {size}% of scan area")
                        st.warning(f"Clinical Concern Level : {risk}")
                    else:    
                        st.info(f"Estimated Tumor Location: {location}")
                        st.info(f"Estimated Tumor Coverage: {size}% of scan area")
                        st.warning(f"Clinical Concern Level : ⚠️ {risk}")
                    st.markdown(recommendations[predicted_class])
                    
                    #new added 
                    # ------------------ Save Grad-CAM Image ------------------
                    gradcam_path = "MedEye_GradCAM_Result.png"
                    gradcam_img.save(gradcam_path)

                    col1,col2,col3=st.columns([1,2,1]) #ratio of division of page into columns
                    with col2:
                        b1,b2=st.columns(2) #placing 2buttons side by side
                        with b1:    
                            # ------------------ Download Button for Grad-CAM ------------------
                            with open(gradcam_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Your Heatmap Image",
                                    data=file,
                                    file_name="MedEye_GradCAM_Result.png",
                                    mime="image/png"
                                )

                        with b2:
                            # ------------------ Download Button for Report-pdf ------------------
                            original_img_path = st.session_state.original_img_path
                            pdf_path = generate_pdf_report(
                                predicted_class,
                                confidence,
                                gradcam_path,
                                original_img_path,
                                patient_name,
                                patient_id,
                                age,
                                gender,
                                scan_date, location, size, risk
                            )
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    label="📄 Download Medical PDF Report",
                                    data=pdf_file,
                                    file_name="MedEye_Brain_Tumor_Report.pdf",
                                    mime="application/pdf"
                                )
                # render_chatbot()
else :
    render_chatbot()

