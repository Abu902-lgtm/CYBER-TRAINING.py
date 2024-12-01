import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import plotly.express as px
from transformers import pipeline
from huggingface_hub import login
import streamlit as st

# Authenticate with Hugging Face
login(token="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")  # Use your Hugging Face token here

# Create Example Dataset
data = {
    'employee_id': [1, 2, 3, 4, 5],
    'completed_training': [True, False, True, False, True],
    'quiz_score': [85, 60, 92, 45, 88],
    'training_topic_scores': [{'phishing': 90, 'malware': 80}, {'phishing': 60, 'malware': 50}, 
                               {'phishing': 85, 'malware': 95}, {'phishing': 40, 'malware': 50}, 
                               {'phishing': 92, 'malware': 85}],
    'quiz_passed': [True, False, True, False, True]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Assign RAG Status based on Training and Quiz Scores
def assign_RAG_status(row):
    if not row['completed_training'] or not row['quiz_passed']:
        return 'Red'
    elif row['quiz_score'] < 70:
        return 'Amber'
    else:
        return 'Green'

df['RAG_status'] = df.apply(assign_RAG_status, axis=1)

# Streamlit UI Elements
st.title('Employee Cybersecurity Awareness Tracker')
st.write("### Employee RAG Status Overview")
st.dataframe(df[['employee_id', 'RAG_status']])

# Initialize Hugging Face pipeline with GPT-2 (text generation)
generator = pipeline('text-generation', model='gpt2')

def generate_quiz(question_topic):
    prompt = f"Generate a multiple-choice quiz question on the topic of {question_topic}."
    return generator(prompt, max_length=100)[0]['generated_text']

# Generate Quiz Example
quiz_question = generate_quiz('Phishing')

st.write("### Generated Quiz Question")
st.write(quiz_question)

# Plotting Quiz Scores Using Plotly
st.write("### Employee Quiz Scores")
fig = px.bar(df, x='employee_id', y='quiz_score', title='Employee Quiz Scores')
st.plotly_chart(fig)

# Analyze Weak Points using Sentiment Analysis
sentiment_analyzer = pipeline('sentiment-analysis')

def analyze_quiz_feedback(feedback):
    return sentiment_analyzer(feedback)

# Example Feedback Analysis
feedback = "I was confused about how phishing emails look like."
result = analyze_quiz_feedback(feedback)

st.write("### Sentiment Analysis of Feedback")
st.write(result)

# Visualizing RAG Status and Scores
st.write("### Distribution of Employee RAG Status")
fig2 = px.pie(df, names='RAG_status', title='Distribution of Employee RAG Status')
st.plotly_chart(fig2)

# Optionally, track progress and weak points by analyzing training topic scores
def analyze_training_weaknesses(row):
    weaknesses = {}
    for topic, score in row['training_topic_scores'].items():
        if score < 70:
            weaknesses[topic] = score
    return weaknesses

df['training_weaknesses'] = df.apply(analyze_training_weaknesses, axis=1)

st.write("### Employee Training Weaknesses")
st.write(df[['employee_id', 'training_weaknesses']])

# Send Email Reminder for Employees Who Haven't Completed Training
def send_reminder(email, name):
    sender_email = "your_email@gmail.com"  # Replace with your email
    sender_password = "your_app_password"  # Replace with your Gmail App Password (not regular password)
    
    # Email setup
    subject = "Reminder: Complete Your Cybersecurity Awareness Training"
    body = f"Dear {name},\n\nYou have not completed your cybersecurity awareness training. Please complete it to stay compliant and secure.\n\nBest regards,\nThe Security Team"
    
    # Create email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    # Send email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Upgrade the connection to secure TLS
        server.login(sender_email, sender_password)  # Login using your email and the app password
        text = msg.as_string()
        server.sendmail(sender_email, email, text)  # Send the email
        server.quit()  # Quit the SMTP server
        print(f"Reminder sent to {name}!")
    except smtplib.SMTPAuthenticationError:
        print("SMTP Authentication error: Incorrect username/password or security block.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Send reminder to employees who haven't completed training
for idx, row in df.iterrows():
    if not row['completed_training']:
        send_reminder("employee_email@example.com", f"Employee {row['employee_id']}")
