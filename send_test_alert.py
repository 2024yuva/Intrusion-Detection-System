"""
Simple script to send a test email alert from the IDS system.
Run this script to manually trigger a test email.
"""

from utils.alerts import send_email_alert
from datetime import datetime

def send_test_email():
    now = datetime.now()
    
    subject = "🔴 [Test Alert] Manual Security Alert - IDS"
    body = f"""Dear Administrator,

This is a MANUAL TEST ALERT from the Intrusion Detection System (IDS).

**Test Details:**
- **Type:** Manual Test
- **Time:** {now.strftime('%Y-%m-%d %H:%M:%S')}
- **Triggered By:** Manual Test Script

This alert was triggered manually for testing purposes.

Regards,
AI-IDS System"""
    
    print("📧 Sending test email alert...")
    send_email_alert(
        subject=subject,
        body=body,
        to_email="yuvarrunjithars.aiml2024@citchennai.net"
    )
    print("✅ Test email sent successfully!")

if __name__ == "__main__":
    send_test_email()
