import re
import json
import asyncio
import tempfile
import urllib.parse
import os
import base64
from typing import Optional, List, Dict, Any, Tuple
import httpx
from playwright.async_api import async_playwright, Page
from dotenv import load_dotenv
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load environment
load_dotenv()

# AI PIPE Configuration
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY", "")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://api.aipipe.org/openai/v1")

# Initialize LLM client
llm_client = None
if AIPIPE_API_KEY:
    try:
        llm_client = AsyncOpenAI(
            api_key=AIPIPE_API_KEY,
            base_url=AIPIPE_BASE_URL,
            timeout=120.0
        )
    except Exception as e:
        print(f"Warning: Could not initialize AI PIPE client: {e}")

# Optional imports
try:
    from word2number import w2n
except:
    w2n = None

try:
    import whisper as whisper_pkg
except:
    whisper_pkg = None

# -------------------------
# HELPER FUNCTIONS (These were missing!)
# -------------------------
def extract_api_headers(text: str) -> Dict[str, str]:
    """Extract API headers from text."""
    headers = {}
    patterns = [
        (r'[Hh]eader[s]?\s*:\s*\{([^}]+)\}', 'json'),
        (r'[Aa]uthorization\s*:\s*([^\s<>"]+)', 'auth'),
        (r'API[- ]?[Kk]ey\s*:\s*([^\s<>"]+)', 'apikey'),
        (r'[Bb]earer\s+([^\s<>"]+)', 'bearer'),
        (r'[Tt]oken\s*:\s*([^\s<>"]+)', 'token'),
    ]
    for pattern, ptype in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if ptype == 'json':
                try: headers.update(json.loads('{' + match + '}'))
                except: pass
            elif ptype == 'auth':
                if 'Authorization' not in headers: headers['Authorization'] = match
            elif ptype in ['apikey', 'bearer', 'token']:
                if 'Authorization' not in headers:
                    headers['Authorization'] = f'Bearer {match}' if 'Bearer' not in match else match
    return headers

def parse_html_with_bs4(html: str) -> Dict[str, Any]:
    """Parse HTML with BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, 'lxml')
        result = {
            "text": soup.get_text(strip=True, separator=' '),
            "tables": [],
            "links": [],
            "hidden_data": [],
            "forms": []
        }
        for table in soup.find_all('table'):
            try:
                df = pd.read_html(str(table))[0]
                result["tables"].append(df.to_dict())
            except: pass
        
        for link in soup.find_all('a', href=True):
            result["links"].append({"text": link.get_text(strip=True), "href": link['href']})
            
        # Extract hidden data
        for attr in ['data-secret', 'data-code', 'data-answer', 'data-value']:
            for elem in soup.find_all(attrs={attr: True}):
                result["hidden_data"].append(elem.get(attr))
                
        return result
    except Exception as e:
        print(f"BS4 error: {e}")
        return {"text": "", "tables": [], "links": [], "hidden_data": []}

# -------------------------
# INTELLIGENT LLM FUNCTIONS
# -------------------------
async def llm_understand_question(question_text: str, context: str) -> Dict[str, Any]:
    """Use LLM to deeply understand what the question is asking."""
    if not llm_client: return {"understood": False}
    try:
        prompt = f"""Analyze this quiz question carefully and extract:
1. The specific column name to operate on (if data is tabular).
2. The filter condition (e.g., "where city is Paris").
3. The operation (sum, average, count, etc.).

Question: {question_text}
Context Preview: {context[:500]}

Return JSON: {{"target_column": "...", "filter": "...", "operation": "...", "answer_format": "..."}}"""

        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst helper. JSON response only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        if result.startswith('```json'): result = result.split('```json')[1].split('```')[0]
        if result.startswith('{'): return json.loads(result)
        return {"understood": True}
    except Exception as e:
        print(f"LLM understanding error: {e}")
        return {"understood": False}

async def llm_vision_analysis(image_path: str, question: str) -> Optional[str]:
    """Use GPT-4o Vision to analyze charts or screenshots."""
    if not llm_client: return None
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Answer this question based on the image. If it's a number, return just the number. Question: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Vision error: {e}")
        return None

async def llm_solve_with_reasoning(question: str, data: str, question_understanding: Dict) -> Dict[str, Any]:
    """Use LLM with chain-of-thought reasoning."""
    if not llm_client: return {"success": False}
    try:
        prompt = f"""Solve this data problem step-by-step.
Question: {question}
Data Excerpt: {data[:3000]}
MetaData: {json.dumps(question_understanding)}

Return strictly JSON:
{{
    "answer": <final_answer>,
    "confidence": <0.0-1.0>
}}"""
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise solver. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()
        if "```" in result: result = result.split("```json")[-1].split("```")[0]
        return {"success": True, "result": json.loads(result.strip())}
    except:
        return {"success": False}

# -------------------------
# DATA PROCESSING & UTILITIES
# -------------------------
def intelligent_column_match(df: pd.DataFrame, target_hint: str) -> Optional[str]:
    if not target_hint or target_hint.lower() == "none":
        nums = df.select_dtypes(include=[np.number]).columns
        return nums[-1] if len(nums) > 0 else None
    target_hint = target_hint.lower()
    for col in df.columns:
        if col.lower() == target_hint: return col
    for col in df.columns:
        if target_hint in col.lower(): return col
    return None

async def process_data_file(file_path: str, file_type: str) -> Optional[pd.DataFrame]:
    try:
        if file_type == "csv": return pd.read_csv(file_path)
        if file_type in ["xlsx", "xls"]: return pd.read_excel(file_path)
        if file_type == "json": return pd.read_json(file_path)
    except: return None
    return None

def normalize_url(candidate: str, page_url: str) -> Optional[str]:
    if not candidate: return None
    try:
        return urllib.parse.urljoin(page_url, candidate)
    except: return None

async def download_file(url: str, client: httpx.AsyncClient, dest_path: str, headers: Dict = None):
    try:
        r = await client.get(url, follow_redirects=True, timeout=60.0, headers=headers)
        if r.status_code == 200:
            with open(dest_path, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

def extract_and_sum_numbers(text: str) -> float:
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if not nums: return 0.0
    total = 0.0
    for n in nums:
        val = float(n)
        total += val
    return total

async def transcribe_audio_with_fallbacks(path):
    if llm_client:
        try:
            with open(path, "rb") as f:
                return (await llm_client.audio.transcriptions.create(
                    model="whisper-1", file=f, response_format="text"
                ))
        except Exception as e: print(f"OpenAI Audio Error: {e}")
    if whisper_pkg:
        try:
            model = whisper_pkg.load_model("base")
            return model.transcribe(path)["text"]
        except: pass
    return None

# -------------------------
# STRATEGIES
# -------------------------
class DataFileStrategy:
    async def try_solve(self, context: Dict) -> Tuple[Optional[Any], float]:
        data_files = context.get("data_files", [])
        if not data_files: return None, 0.0
        
        question_data = context.get("question_understanding", {})
        target_col_hint = question_data.get("target_column")
        operation = question_data.get("operation", "sum").lower()
        
        for item in data_files:
            df = item['dataframe']
            if df is None or df.empty: continue
            target_col = intelligent_column_match(df, target_col_hint)
            if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
                if "mean" in operation or "average" in operation: return float(df[target_col].mean()), 0.95
                elif "count" in operation: return len(df), 0.95
                else: return float(df[target_col].sum()), 0.95
            nums = df.select_dtypes(include=[np.number])
            if not nums.empty: return float(nums.iloc[:, -1].sum()), 0.6
        return None, 0.0

class ImageStrategy:
    async def try_solve(self, context: Dict) -> Tuple[Optional[Any], float]:
        images = context.get("images", [])
        text = context.get("text", "")
        if not images: return None, 0.0
        if not any(x in text.lower() for x in ['chart', 'graph', 'image', 'picture', 'plot']): return None, 0.0
        
        print("[ImageStrategy] Analyzing image...")
        result = await llm_vision_analysis(images[0], text)
        if result:
             clean_res = re.sub(r"[^\d\.-]", "", result)
             if clean_res: return float(clean_res), 0.9
             return result, 0.85
        return None, 0.0

class AudioStrategy:
    async def try_solve(self, context: Dict) -> Tuple[Optional[Any], float]:
        audio_files = context.get("audio_files", [])
        if not audio_files: return None, 0.0
        
        print("[AudioStrategy] Transcribing...")
        transcript = await transcribe_audio_with_fallbacks(audio_files[0])
        if transcript:
            print(f"[AudioStrategy] Transcript: {transcript[:50]}...")
            q_data = context.get("question_understanding", {})
            res = await llm_solve_with_reasoning(context.get("text", ""), transcript, q_data)
            if res['success']: return res['result']['answer'], 0.9
            return extract_and_sum_numbers(transcript), 0.7
        return None, 0.0

# -------------------------
# MAIN SOLVER
# -------------------------
async def solve_quiz(payload: dict) -> dict:
    quiz_url = payload.get("url")
    if not quiz_url: return {"error": "No URL"}

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
                "--no-sandbox",
            ]
        )
        context_browser = await browser.new_context()
        page = await context_browser.new_page()
        
        try:
            print(f"Navigating to {quiz_url}...")
            await page.goto(quiz_url, wait_until="networkidle", timeout=60000)
            await page.wait_for_timeout(2000)

            # 1. AUTH & COOKIES
            api_headers = {}
            cookies = await page.context.cookies()
            cookie_dict = {c['name']: c['value'] for c in cookies}

            try:
                local_storage = await page.evaluate("() => JSON.stringify(localStorage)")
                ls_data = json.loads(local_storage)
                for k, v in ls_data.items():
                    if any(x in k.lower() for x in ["token", "auth", "key", "jwt"]):
                        api_headers["Authorization"] = f"Bearer {v}"
            except: pass
            
            # 2. PARSE PAGE
            html = await page.content()
            text = await page.inner_text("body")
            parsed_data = parse_html_with_bs4(html)
            api_headers.update(extract_api_headers(text))

            context = {
                "quiz_url": quiz_url, "html": html, "text": text,
                "parsed_data": parsed_data, "api_headers": api_headers,
                "data_files": [], "audio_files": [], "images": []
            }

            # 3. ANALYZE & DOWNLOAD
            q_understanding = await llm_understand_question(text[:2000], html[:3000])
            context["question_understanding"] = q_understanding

            async with httpx.AsyncClient(cookies=cookie_dict, headers=api_headers) as client:
                # Data Files
                links = re.findall(r'href=["\']([^"\']+\.(csv|xlsx|xls|json))["\']', html, re.IGNORECASE)
                for link, ext in links:
                    url = normalize_url(link, quiz_url)
                    if url:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}").name
                        if await download_file(url, client, tmp, api_headers):
                            df = await process_data_file(tmp, ext)
                            if df is not None: context["data_files"].append({"dataframe": df, "path": tmp})

                # Audio
                audio_links = re.findall(r'(?:href|src)=["\']([^"\']+\.(mp3|wav|m4a))["\']', html, re.IGNORECASE)
                for link, ext in audio_links:
                    url = normalize_url(link, quiz_url)
                    if url:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}").name
                        if await download_file(url, client, tmp, api_headers):
                            context["audio_files"].append(tmp)

                # Images
                img_links = re.findall(r'src=["\']([^"\']+\.(png|jpg|jpeg))["\']', html, re.IGNORECASE)
                for link, ext in img_links:
                    url = normalize_url(link, quiz_url)
                    if url and "logo" not in url.lower():
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}").name
                        if await download_file(url, client, tmp, api_headers):
                            context["images"].append(tmp)

            # 4. EXECUTE STRATEGIES
            strategies = [DataFileStrategy(), AudioStrategy(), ImageStrategy()]
            for strat in strategies:
                ans, conf = await strat.try_solve(context)
                if ans is not None and conf > 0.8:
                    return await submit_answer(payload, ans, html, page)

            # 5. FALLBACKS
            data_preview = ""
            if context["data_files"]: data_preview = context["data_files"][0]["dataframe"].to_string()[:2000]
            llm_res = await llm_solve_with_reasoning(text, data_preview + "\n" + html[:2000], q_understanding)
            if llm_res["success"]: return await submit_answer(payload, llm_res["result"]["answer"], html, page)
            
            return await submit_answer(payload, extract_and_sum_numbers(text), html, page)

        except Exception as e:
            print(f"Solver Error: {e}")
            return {"error": str(e)}
        finally:
            await context_browser.close()
            await browser.close()

async def submit_answer(payload, answer, html, page):
    submit_url = None
    match = re.search(r'https?://[^\s"<>]+/submit', html)
    if match: submit_url = match.group(0)
    else: 
        parsed = urllib.parse.urlparse(payload['url'])
        submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"

    print(f"Submitting Answer: {answer} to {submit_url}")
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            submit_url, 
            json={
                "email": payload["email"],
                "secret": payload["secret"],
                "url": payload["url"],
                "answer": answer
            },
            timeout=30.0
        )
        return {"submit": resp.json(), "answer": answer}