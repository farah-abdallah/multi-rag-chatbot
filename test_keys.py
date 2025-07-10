import google.generativeai as genai

api_keys = [
"AIzaSyC3CI_Ti1qrM9xrs-23KLcvFPyUGhc0T-Y",
"AIzaSyB2P0MgAwtuHhiJDJustNP3KJeAHe1FJYg",
"AIzaSyArH1EbvOwy5NY1HzgJJ4xri_qPPLT3sHM",
"AIzaSyCQU0N0zrVwWrYeFyEGZ8Xl0CUI33ewLrs",
"AIzaSyCwtr3lEgKJeRBo5BwJSsFbuuZK60YVt0o",
"AIzaSyDi02ttwmYEWfhxgCd0Q7fqMpzQTdrKh_k"
]

for idx, key in enumerate(api_keys, 1):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Hello, world!")
        print(f"Key #{idx}: VALID ✅")
    except Exception as e:
        print(f"Key #{idx}: INVALID ❌ - {e}")