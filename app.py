import asyncio
import json
import os
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from solver import solve_quiz

# Load environment variables
load_dotenv()



app = FastAPI(title="LLM Quiz Solver API")

# Load secrets from secrets.json (keeping for email) and .env
try:
    with open("secrets.json") as f:
        SECRETS = json.load(f)
    EXPECTED_SECRET = SECRETS.get("secret", os.getenv("SECRET_KEY", ""))
    EXPECTED_EMAIL = SECRETS.get("email", os.getenv("EMAIL", ""))
except FileNotFoundError:
    # Fallback to environment variables only
    EXPECTED_SECRET = os.getenv("SECRET_KEY", "")
    EXPECTED_EMAIL = os.getenv("EMAIL", "")

if not EXPECTED_SECRET:
    raise ValueError("SECRET_KEY not found in secrets.json or .env file!")

class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "LLM Quiz Solver API is running"}

@app.post("/")
async def receive_quiz(payload: QuizPayload, request: Request):
    """
    Main endpoint to receive quiz tasks.
    Returns HTTP 200 for valid requests, 403 for invalid secret, 400 for invalid JSON.
    """
    # Validate secret
    if payload.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    # Spawn background task to solve the quiz
    asyncio.create_task(background_process(payload.dict()))
    
    # Return immediate response as required
    return JSONResponse(
        status_code=200,
        content={
            "status": "accepted",
            "message": "Quiz task received and processing started",
            "url": payload.url
        }
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle invalid JSON payloads with HTTP 400"""
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid JSON payload", "details": str(exc)}
    )

async def background_process(payload: dict):
    """
    Run solver on the provided payload. If the solver's submit response returns
    a new 'url', follow it and solve again. Continue until no new url is returned
    or until 3 minutes have elapsed from start.
    """
    start_ts = time.time()
    max_seconds = 3 * 60  # 3 minutes
    
    current_payload = payload.copy()
    attempt = 0
    final_result = None
    
    print(f"[bg] Starting background process for {payload.get('url')}")
    
    while True:
        attempt += 1
        
        # Check elapsed time first
        elapsed = time.time() - start_ts
        if elapsed > max_seconds:
            print(f"[bg] Time limit exceeded ({elapsed:.1f}s > {max_seconds}s). Stopping.")
            break
        
        try:
            print(f"[bg] Attempt {attempt} for {current_payload.get('url')} (elapsed: {elapsed:.1f}s)")
            
            # Solve the quiz
            result = await solve_quiz(current_payload)
            print(f"[bg] Solver result: {json.dumps(result, default=str)[:200]}...")
            
            # Save result
            final_result = {
                "attempt": attempt,
                "payload": current_payload,
                "result": result,
                "timestamp": time.time(),
                "elapsed_seconds": elapsed
            }
            
            with open("last_solver_result.json", "w") as f:
                json.dump(final_result, f, indent=2, default=str)
            
            # Check if there's a next URL to follow
            next_url = None
            if isinstance(result, dict):
                # Look for next URL in common locations
                submit = result.get("submit") or result.get("submit_response") or {}
                if isinstance(submit, dict):
                    next_url = (
                        submit.get("url") or 
                        submit.get("next") or 
                        submit.get("data", {}).get("url")
                    )
                
                # Also check if correct=true and url is provided at top level
                if not next_url and result.get("correct"):
                    next_url = result.get("url")
            
            # Stop if no next URL
            if not next_url:
                print("[bg] No next URL received; quiz complete.")
                break
            
            # Prepare next iteration
            print(f"[bg] Following next URL: {next_url}")
            current_payload = {
                "email": payload.get("email"),
                "secret": payload.get("secret"),
                "url": next_url
            }
            
            # Small delay before next request
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"[bg] Exception in background_process: {type(e).__name__}: {str(e)}")
            error_result = {
                "error": str(e),
                "error_type": type(e).__name__,
                "attempt": attempt,
                "timestamp": time.time()
            }
            with open("last_solver_result.json", "w") as f:
                json.dump(error_result, f, indent=2)
            break
    
    # Final logging
    print(f"[bg] Background process complete. Total attempts: {attempt}, Time: {time.time() - start_ts:.1f}s")
    if final_result:
        with open("last_solver_result.json", "w") as f:
            json.dump(final_result, f, indent=2, default=str)

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi.exceptions import RequestValidationError
from starlette.requests import Request as StarletteRequest

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: StarletteRequest, exc: RequestValidationError):
    """
    Return HTTP 400 for invalid JSON / request validation errors (so the test expecting 400 passes).
    """
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid JSON payload", "details": str(exc)}
    )
