import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

# ⚠️ Email Configuration - Use Gmail App Password
EMAIL_SENDER = "yuvarrunjithars@gmail.com"
EMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "yuva@1910")  # Use environment variable or fallback
EMAIL_RECIPIENT = "yuvarrunjithars.aiml2024@citchennai.net"

# Alert tracking
alert_history = {
    "total_count": 0,
    "last_alert": None,
    "recent_alerts": []
}

def send_email_alert(anomaly_count, total_packets, anomaly_rate):
    """
    Send a professional email alert for detected anomalies.
    
    Args:
        anomaly_count: Number of anomalies detected
        total_packets: Total packets analyzed
        anomaly_rate: Percentage of anomalies
    """
    global alert_history
    
    # Create professional email
    subject = f"🚨 IDS Alert: {anomaly_rate:.1f}% Anomaly Rate Detected"
    
    body = f"""
INTRUSION DETECTION SYSTEM - SECURITY ALERT

Incident Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

THREAT SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Anomaly Rate: {anomaly_rate:.2f}%
• Anomalies Detected: {anomaly_count}/{total_packets} packets
• Detection Models: Ensemble (SVM + Isolation Forest)

RECOMMENDED ACTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Review network traffic logs immediately
2. Check for suspicious IP addresses
3. Verify firewall rules and access controls
4. Monitor for potential DDoS or intrusion attempts

This is an automated alert from your AI-powered Intrusion Detection System.

Dashboard: http://localhost:5000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    try:
        # Create message
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECIPIENT
        msg.attach(MIMEText(body, "plain"))

        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        
        # Update tracking
        timestamp = datetime.now().strftime("%d/%m/%Y, %I:%M:%S %p")
        alert_history["total_count"] += 1
        alert_history["last_alert"] = timestamp
        alert_history["recent_alerts"].insert(0, {
            "time": timestamp,
            "anomalies": anomaly_count,
            "total": total_packets,
            "rate": round(anomaly_rate, 1)
        })
        
        # Keep only last 5 alerts
        alert_history["recent_alerts"] = alert_history["recent_alerts"][:5]
        
        print(f"✅ Email alert sent successfully to {EMAIL_RECIPIENT}")
        return True
        
    except Exception as e:
        print(f"❌ Error sending email alert: {e}")
        return False


def get_alert_status():
    """
    Get current alert status and history.
    """
    return {
        "alert_count": alert_history["total_count"],
        "last_alert": alert_history["last_alert"],
        "recent_alerts": alert_history["recent_alerts"]
    }
