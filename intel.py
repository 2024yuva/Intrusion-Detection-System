"""
intel.py — Predictive Threat Intelligence Module
------------------------------------------------
Fetches external reputation data from:
- VirusTotal
- AbuseIPDB
and computes a combined threat score (0–100).
If API keys aren't provided, returns random mock data.
"""

import os
import random
import requests

# Insert your API keys here
VIRUSTOTAL_API_KEY = "f63f2bb631b93fa895a82941e8b5377a84e2f4b17180fcd5b6945b126abd6102"
ABUSEIPDB_API_KEY = "14cb37f71b2bdb12e31184efdeac7cfcee861f5e3b63d35e019b0a2a79c39275fe36703fbf935340"


def check_ip_threat(ip: str):
    result = {
        "ip": ip,
        "virustotal_malicious": 0,
        "abuse_confidence": 0,
        "threat_score": 0
    }

    # MOCK MODE — if keys missing, return random data
    if "your_virustotal_api_key_here" in VIRUSTOTAL_API_KEY or "your_abuseipdb_api_key_here" in ABUSEIPDB_API_KEY:
        result["virustotal_malicious"] = random.randint(0, 10)
        result["abuse_confidence"] = random.randint(0, 100)
        result["threat_score"] = 0.6 * result["virustotal_malicious"] + 0.4 * result["abuse_confidence"]
        return result

    # 🔹 VirusTotal
    try:
        headers = {"x-apikey": VIRUSTOTAL_API_KEY}
        vt_url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
        vt_resp = requests.get(vt_url, headers=headers, timeout=10)
        if vt_resp.status_code == 200:
            data = vt_resp.json()
            result["virustotal_malicious"] = data["data"]["attributes"]["last_analysis_stats"]["malicious"]
    except Exception as e:
        print("VirusTotal error:", e)

    # 🔹 AbuseIPDB
    try:
        headers = {"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"}
        abuse_url = f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}&maxAgeInDays=90"
        ab_resp = requests.get(abuse_url, headers=headers, timeout=10)
        if ab_resp.status_code == 200:
            data = ab_resp.json()
            result["abuse_confidence"] = data["data"]["abuseConfidenceScore"]
    except Exception as e:
        print("AbuseIPDB error:", e)

    # Combine
    result["threat_score"] = 0.6 * result["virustotal_malicious"] + 0.4 * result["abuse_confidence"]
    return result
