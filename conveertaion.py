import requests
import time
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import re
import google.generativeai as genai
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import pytz

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/convai/conversations"

# Headers for ElevenLabs API requests
headers = {
    "xi-api-key": ELEVEN_LABS_API_KEY,
    "Content-Type": "application/json"
}

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def validate_email(email):
    """Validate email address format."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def extract_email_from_gemini_response(gemini_response):
    """Extract email address from Gemini response."""
    if not gemini_response:
        return None
    try:
        for line in gemini_response.split('\n'):
            if line.startswith("Email:"):
                email = line.split(":", 1)[1].strip()
                if email and email != "None" and validate_email(email):
                    return email.lower()
        return None
    except Exception as e:
        logging.error(f"Error extracting email from Gemini response: {e}")
        return None

def lowercase_gemini_response(gemini_response):
    """Lowercase the email address in the Gemini response."""
    if not gemini_response:
        return gemini_response
    try:
        lines = gemini_response.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("Email:"):
                email = line.split(":", 1)[1].strip()
                if email and email != "None" and validate_email(email):
                    lines[i] = f"Email: {email.lower()}"
        return '\n'.join(lines)
    except Exception as e:
        logging.error(f"Error lowercasing Gemini response: {e}")
        return gemini_response

def test_smtp_credentials():
    """Test SMTP credentials."""
    print("Debug: Testing SMTP credentials...")
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
        server.quit()
        logging.info("SMTP credentials validated successfully")
        print("Debug: SMTP credentials validated successfully")
        return True
    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP authentication failed")
        print("Error: SMTP authentication failed. Check SMTP_EMAIL or SMTP_APP_PASSWORD.")
        return False
    except Exception as e:
        logging.error(f"SMTP connection failed: {e}")
        print(f"Error: SMTP connection failed: {e}")
        return False

def fetch_conversations(cursor=None, page_size=30):
    """Fetch conversations with pagination."""
    params = {"page_size": page_size}
    if cursor:
        params["cursor"] = cursor
    try:
        response = requests.get(ELEVEN_LABS_BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        logging.info("Successfully fetched conversations")
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching conversations: {e}")
        return None

def fetch_conversation_details(conversation_id):
    """Fetch details of a specific conversation."""
    url = f"{ELEVEN_LABS_BASE_URL}/{conversation_id}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logging.info(f"Successfully fetched details for conversation {conversation_id}")
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching conversation {conversation_id}: {e}")
        return None

def extract_customer_email(conversation_data):
    """Extract customer email from conversation data."""
    try:
        client_data = conversation_data.get("conversation_initiation_client_data", {})
        email = client_data.get("email")
        if email and validate_email(email):
            logging.info("Email found in conversation_initiation_client_data")
            return email.lower()
        transcript = conversation_data.get("transcript", [])
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        for entry in transcript:
            message = entry.get("message", "")
            match = email_pattern.search(message)
            if match and validate_email(match.group(0)):
                logging.info("Email found in transcript")
                return match.group(0).lower()
        logging.warning("No valid email found")
        return None
    except Exception as e:
        logging.error(f"Error extracting email: {e}")
        return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.info(f"Retrying Gemini API call (attempt {retry_state.attempt_number})")
)
def send_to_gemini(transcript):
    """Send conversation transcript to Gemini API."""
    try:
        if not transcript:
            logging.warning("Transcript is empty, skipping Gemini API call")
            print("Debug: Transcript is empty, skipping Gemini API call")
            return None
        
        transcript_text = "\n".join(
            f"[{entry.get('time_in_call_secs', 0)}s] {entry.get('role', 'Unknown')}: {entry.get('message', '') or 'No message (e.g., tool call)'}"
            for entry in transcript
        )
        
        prompt = f"""Analyze the following conversation transcript and perform these tasks:
1. Extract the email address mentioned in the transcript, if any.
If it has a hyphen, remove that hyphen and give output in lowercase.
2. Provide a concise summary of the conversation.

Transcript:
{transcript_text}

Output format:
Email: [email address or "None"]
Summary: [summary of the conversation]
"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        if response.text:
            logging.info("Successfully received response from Gemini API")
            print("Debug: Gemini API response:", response.text)
            return lowercase_gemini_response(response.text)
        else:
            logging.warning("No response text from Gemini API")
            print("Debug: No response text from Gemini API")
            return None
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        print(f"Debug: Error calling Gemini API: {e}")
        return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((smtplib.SMTPException, ConnectionError)),
    before_sleep=lambda retry_state: logging.info(f"Retrying send_email (attempt {retry_state.attempt_number})")
)
def send_email(to_email, summary, conversation_id):
    """Send email with provided summary."""
    print(f"Debug: Entering send_email(to_email={to_email}, conversation_id={conversation_id})")
    server = None
    try:
        if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
            logging.error("Missing SMTP credentials")
            return False, "Missing SMTP credentials"
        if not validate_email(to_email):
            logging.error(f"Invalid email address: {to_email}")
            print(f"Debug: Invalid email address: {to_email}")
            return False, "Invalid email address"
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'Conversation Summary (ID: {conversation_id})'
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email

        text_part = MIMEText(summary or "No summary available.", 'plain')
        html_message = f"<h2>Conversation Summary</h2><p>{summary or 'No summary available.'}</p>"
        html_message += f"<p>Conversation ID: {conversation_id}</p><p>Best regards,<br>The Hotel Team</p>"
        html_part = MIMEText(html_message, 'html')

        msg.attach(text_part)
        msg.attach(html_part)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
        server.send_message(msg)
        
        logging.info(f"Email sent successfully to {to_email}")
        print(f"Debug: Email sent successfully to {to_email}")
        return True, "Success"
    except smtplib.SMTPAuthenticationError:
        logging.error(f"SMTP authentication failed for {to_email}")
        print(f"Debug: SMTP authentication failed for {to_email}")
        return False, "Authentication error"
    except Exception as e:
        logging.error(f"Error sending email to {to_email}: {e}")
        print(f"Debug: Error sending email to {to_email}: {e}")
        return False, str(e)
    finally:
        if server:
            try:
                server.quit()
            except:
                pass

def display_conversation_details(conversation_data, conversation_id):
    """Display and return conversation details."""
    try:
        transcript = conversation_data.get("transcript", [])
        status = conversation_data.get("status", "Unknown")
        start_time_unix = conversation_data.get("metadata", {}).get("start_time_unix_secs", 0)
        call_duration = conversation_data.get("metadata", {}).get("call_duration_secs", 0)

        # Convert UTC timestamp to IST using timezone-aware datetime
        if start_time_unix:
            utc_time = datetime.fromtimestamp(start_time_unix, tz=pytz.UTC)
            ist_timezone = pytz.timezone('Asia/Kolkata')
            ist_time = utc_time.astimezone(ist_timezone)
            start_time = ist_time.strftime('%Y-%m-%d %H:%M:%S IST')
        else:
            start_time = "Unknown"

        conversation_text = (
            f"Conversation Details (ID: {conversation_id}):\nStatus: {status}\nStart Time: {start_time}\n"
            f"Call Duration: {call_duration} seconds\nTranscript:\n"
        )

        if not transcript:
            conversation_text += "No transcript available.\n"
        else:
            for entry in transcript:
                role = entry.get("role", "Unknown")
                message = entry.get("message", "") or "No message (e.g., tool call)"
                time_in_call = entry.get("time_in_call_secs", 0)
                conversation_text += f"[{time_in_call}s] {role}: {message}\n"
        
        logging.info(f"Displayed conversation details for {conversation_id}")
        print(conversation_text)
        return conversation_text, transcript
    except Exception as e:
        logging.error(f"Error displaying conversation details: {e}")
        return f"Error displaying conversation details: {e}", []

def get_last_conversation_id():
    """Get the ID of the most recent completed conversation without processing it."""
    conversations_data = fetch_conversations(page_size=10)
    if not conversations_data or "conversations" not in conversations_data or not conversations_data["conversations"]:
        logging.info("No conversations found for last ID")
        print("No conversations found for last ID.")
        return None
    
    for conversation in conversations_data["conversations"]:
        conversation_id = conversation["conversation_id"]
        conversation_details = fetch_conversation_details(conversation_id)
        if conversation_details and conversation_details.get("status") == "done":
            logging.info(f"Found last completed conversation: {conversation_id}")
            print(f"Found last completed conversation: {conversation_id}")
            return conversation_id
    
    logging.info("No completed conversations found for last ID")
    print("No completed conversations found for last ID.")
    return None

def process_conversation(conversation_details, conversation_id):
    """Process a conversation: get Gemini response, log details."""
    if conversation_id == "conv_01jxvvk0kvfspb2p808vwxwb0m":
        print("Debug: Conversation details:", json.dumps(conversation_details, indent=2))
        print("Debug: Status:", conversation_details.get("status", "Unknown"))
        print("Debug: Transcript:", conversation_details.get("transcript", []))
    
    transcript = conversation_details.get("transcript", [])
    gemini_response = send_to_gemini(transcript) if transcript else None
    
    email = extract_customer_email(conversation_details)
    if email:
        logging.info(f"Extracted Email: {email}")
        print(f"Extracted Email: {email}")
    else:
        logging.warning(f"No email found for {conversation_id}")
        print(f"No email found for {conversation_id}")
    
    conversation_text, _ = display_conversation_details(conversation_details, conversation_id)
    if gemini_response:
        logging.info(f"Gemini Summary for {conversation_id}:\n{gemini_response}")
        print(f"Gemini Summary for {conversation_id}:\n{gemini_response}")
    else:
        logging.warning(f"No Gemini summary generated for {conversation_id}")
        print(f"No Gemini summary generated for {conversation_id}")
    
    return conversation_text, email, gemini_response

def monitor_new_conversations(poll_interval=60):
    """Monitor for new completed conversations and send summary to ashimlugun09@gmail.com."""
    last_conversation_id = get_last_conversation_id()
    if last_conversation_id:
        logging.info(f"Monitoring after ID: {last_conversation_id}")
        print(f"Monitoring after ID: {last_conversation_id}")
    else:
        logging.info("No prior completed conversations found")
        print("No prior completed conversations found")
    
    with open("conversation_emails.log", "a") as log_file:
        while True:
            conversations_data = fetch_conversations(page_size=1)
            if not conversations_data or "conversations" not in conversations_data:
                logging.warning("No conversations found")
                print("No conversations found.")
                time.sleep(poll_interval)
                continue
            
            conversations = conversations_data["conversations"]
            if not conversations:
                logging.info("No new conversations")
                print("No new conversations.")
                time.sleep(poll_interval)
                continue
            
            latest_conversation = conversations[0]
            conversation_id = latest_conversation["conversation_id"]
            
            if conversation_id != last_conversation_id:
                conversation_details = fetch_conversation_details(conversation_id)
                if conversation_details and conversation_details.get("status") == "done":
                    logging.info(f"New conversation: {conversation_id}")
                    print(f"\nNew conversation (ID: {conversation_id}):")
                    last_conversation_id = conversation_id
                    
                    conversation_text, email, gemini_response = process_conversation(conversation_details, conversation_id)
                    
                    if gemini_response:
                        logging.info(f"Sending summary for conversation {conversation_id} to ashimlugun09@gmail.com")
                        print(f"Sending summary for conversation {conversation_id} to ashimlugun09@gmail.com")
                        summary_email_status, summary_email_error = send_email(
                            "ashimlugun09@gmail.com", gemini_response, conversation_id
                        )
                        log_file.write(f"Summary Email Status for {conversation_id} to ashimlugun09@gmail.com: {'Success' if summary_email_status else f'Failed - {summary_email_error}'}\n")
                        print(f"Summary Email Status to ashimlugun09@gmail.com: {'Success' if summary_email_status else f'Failed - {summary_email_error}'}")
                    else:
                        logging.warning(f"No Gemini summary to send for {conversation_id} to ashimlugun09@gmail.com")
                        print(f"No Gemini summary to send for {conversation_id} to ashimlugun09@gmail.com")
                    
                    if email:
                        log_file.write(f"Conversation {conversation_id}: {email}\n")
                        print(f"Logged email for {conversation_id}: {email}")
                    log_file.write(f"Conversation {conversation_id} Details:\n{conversation_text}\n")
                    print(f"Logged conversation details for {conversation_id}")
                    if gemini_response:
                        log_file.write(f"Gemini Summary for {conversation_id}:\n{gemini_response}\n")
                        print(f"Logged Gemini summary for {conversation_id}")
            
            time.sleep(poll_interval)

def main():
    print(f"Debug: SMTP_EMAIL={SMTP_EMAIL}")
    print(f"Debug: SMTP_APP_PASSWORD={'*' * len(SMTP_APP_PASSWORD) if SMTP_APP_PASSWORD else None}")
    print(f"Debug: GEMINI_API_KEY={'*' * len(GEMINI_API_KEY) if GEMINI_API_KEY else None}")
    if not ELEVEN_LABS_API_KEY or not GEMINI_API_KEY or not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        logging.error("Missing required environment variables")
        print("Error: Missing required environment variables")
        return
    
    if not test_smtp_credentials():
        logging.error("SMTP credential test failed")
        print("Debug: SMTP credential test failed")
        return
    
    logging.info("Starting conversation monitoring")
    try:
        monitor_new_conversations(poll_interval=60)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
        print("Script interrupted by user")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()