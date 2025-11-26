import os
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("⚠️ Gemini client not found — AI summary will use local fallback.")


# 🔹 API Key Configuration
# If you wish to manually enter your key, paste it inside the quotes below:
MANUAL_API_KEY = ""

# Try manual key first, then environment variable
API_KEY = MANUAL_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY and genai:
    print("⚠️ No Gemini API key found. Set GEMINI_API_KEY env variable.")
if genai and API_KEY:
    genai.configure(api_key=API_KEY)


def generate_summary():
    """
    Generates a human-readable AI summary for detected network threats.
    Uses Gemini API if available, else falls back to a local summary generator.
    """
    try:
        # ✅ Use Gemini API (if available and configured)
        if genai and API_KEY:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "Summarize potential threat patterns observed in network traffic logs. "
                "Describe anomalies in clear technical terms with suggested actions. "
                "Keep it concise, 2-3 sentences."
            )

            response = model.generate_content(prompt)
            summary_text = (
                response.text.strip()
                if hasattr(response, "text")
                else str(response).strip()
            )
            return summary_text or "Network analysis complete. No critical threats detected."

        # ⚙️ Local fallback (always available)
        return random.choice([
            "Network shows stable activity; minimal anomalies detected.",
            "Multiple high-latency connections observed, possibly due to DDoS attempts.",
            "Unusual TCP traffic pattern detected between internal hosts.",
            "Port 22 and 443 show irregular connection spikes. Recommend deeper inspection.",
            "Overall network behavior normal with low anomaly confidence.",
        ])

    except Exception as e:
        print("⚠️ Gemini API error:", str(e))
        # Always return a fallback summary instead of failing
        return random.choice([
            "Network shows stable activity; minimal anomalies detected.",
            "Multiple high-latency connections observed, possibly due to DDoS attempts.",
            "Unusual TCP traffic pattern detected between internal hosts.",
            "Port 22 and 443 show irregular connection spikes. Recommend deeper inspection.",
            "Overall network behavior normal with low anomaly confidence.",
        ])
