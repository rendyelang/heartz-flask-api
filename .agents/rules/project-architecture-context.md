---
trigger: always_on
---

# Project Architecture Context: Flask AI API (Tunarungu Speech Training)

## 1. Environment & Infrastructure

This project uses a dual-environment setup on Google Cloud Platform (GCP):

- **Production (Branch: main):** Deployed on GCP Cloud Run. Requires 100% uptime.
- **Development (Branch: dev):** Deployed on GCP Cloud Run. Used for continuous experiments.

## 2. LLM Inference Strategy & Fallback Logic

The API generates motivational text for users using LLMs. The application must strictly follow this fallback logic to balance cost and rate limits:

- **Primary AI (Local VM):** The primary endpoint for generating text is a local LLM (e.g., Gemma/Llama) hosted on a separate GCP Compute Engine VM. The Flask app must attempt to call this VM's IP address first.
- **Secondary AI (Fallback/API Key):** The VM will be frequently turned OFF (status: STOPPED) to save costs. If the Flask app detects a timeout, connection error, or 502 Bad Gateway from the VM, it MUST automatically fallback to using free public API keys (e.g., Gemini Flash via Google AI Studio) to ensure the service never goes down.

## 3. Coding Guidelines

- All external API calls and VM requests must be wrapped in clean `try-catch` (or `try-except` in Python) blocks.
- Environment variables (`.env`) must be used to store the VM IP address and all API Keys. Never hardcode credentials or IP addresses.
- Focus on clean, asynchronous, and non-blocking handling to ensure the fallback transition happens smoothly without massive latency for the user.
