# llm_client.py
import os
import json
import asyncio
import time
from typing import Dict, Tuple, Any

class GlobalRateLimiter:
    """
    Forces a minimum delay between API calls.
    Fixed to lazily initialize the asyncio.Lock() inside the active event loop.
    """
    def __init__(self, requests_per_minute: int):
        self.min_delay = 60.0 / requests_per_minute
        self.lock = None  # Do NOT initialize the lock here
        self.last_call_time = 0.0

    async def wait_for_token(self, provider_name: str):
        if self.lock is None:
            self.lock = asyncio.Lock()
            
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed
                print(f"{provider_name} Limiter: Waiting {sleep_time:.1f}s to respect RPM limit...")
                await asyncio.sleep(sleep_time)
            
            self.last_call_time = time.time()

aws_rate_limiter = GlobalRateLimiter(50)
gcp_rate_limiter = GlobalRateLimiter(10)

class UniversalLLMClient:
    """
    Unified client for vLLM (OpenAI-compatible), Google Vertex AI, and Anthropic Claude.
    """
    def __init__(self, provider: str, model_name: str, base_url: str = None, api_key: str = None, project_id: str = None, location: str = "us-central1"):
        self.provider = provider.lower()
        self.model_name = model_name
        self.client = None
        
        print(f"Initializing LLM Client: {self.provider} / {self.model_name}")

        # --- vLLM / OPENAI ---
        if self.provider in ["vllm", "openai"]:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                base_url=base_url, 
                api_key=api_key or "EMPTY"
            )
        
        elif self.provider == "bedrock":
            import boto3
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-2")
            )

        # --- ANTHROPIC (CLAUDE) ---
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )

        # --- GOOGLE VERTEX AI (GEMINI) ---
        elif self.provider == "google_vertex":
            from google import genai
            
            # Setting vertexai=True routes the call securely through GCP.
            # It will automatically use the GOOGLE_APPLICATION_CREDENTIALS from your .env file
            self.client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=location
            )

    async def invoke(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> Tuple[str, Dict[str, int]]:
        usage = {'input': 0, 'output': 0, 'total': 0}
        content = ""
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                # --- vLLM / OPENAI LOGIC ---
                if self.provider in ["vllm", "openai"]:
                    combined_system = (
                    "You are a helpful assistant. You MUST respond ONLY with a valid JSON object. "
                    "Do not include any conversational preamble, markdown formatting, or explanations. "
                    "Your entire response must be ONLY the JSON object.\n"
                    "---SPECIFIC ROLE---\n"
                    f"{system_prompt}"
                )
                    
                    messages = [
                        {"role": "system", "content": combined_system},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # Check if model supports json_object
                    resp_format = {"type": "json_object"} if "gpt" in self.model_name else None

                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=resp_format
                    )
                    content = response.choices[0].message.content
                    if response.usage:
                        usage = {
                            'input': response.usage.prompt_tokens,
                            'output': response.usage.completion_tokens,
                            'total': response.usage.total_tokens
                        }

                # --- ANTHROPIC LOGIC ---
                elif self.provider == "anthropic":
                    # Claude supports system parameter directly
                    message = await self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt + " Respond only in JSON.",
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    content = message.content[0].text
                    usage = {
                        'input': message.usage.input_tokens,
                        'output': message.usage.output_tokens,
                        'total': message.usage.input_tokens + message.usage.output_tokens
                    }
                
                elif self.provider == "bedrock":
                    await aws_rate_limiter.wait_for_token("AWS Bedrock")

                    system_messages = [{"text": system_prompt + "\nIMPORTANT: You must output ONLY valid JSON."}]
                    user_messages = [{"role": "user", "content": [{"text": user_prompt}]}]
                    
                    response = await asyncio.to_thread(
                        self.client.converse,
                        modelId=self.model_name,
                        messages=user_messages,
                        system=system_messages,
                        inferenceConfig={"temperature": temperature, "maxTokens": max_tokens}
                    )
                    
                    content = response['output']['message']['content'][0]['text']
                    usage = {
                        'input': response['usage']['inputTokens'],
                        'output': response['usage']['outputTokens'],
                        'total': response['usage']['totalTokens']
                    }

                # --- GOOGLE VERTEX LOGIC ---
                elif self.provider == "google_vertex":
                    await gcp_rate_limiter.wait_for_token("Google Vertex")
                    from google.genai import types
                    config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                    )
                    
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=user_prompt,
                        config=config
                    )
                    
                    content = response.text
                    
                    # 3. Safely calculate token usage, including the hidden reasoning tokens
                    if response.usage_metadata:
                        input_toks = response.usage_metadata.prompt_token_count or 0
                        candidate_toks = response.usage_metadata.candidates_token_count or 0
                        # Use getattr in case you switch to a non-thinking model later
                        thought_toks = getattr(response.usage_metadata, 'thoughts_token_count', 0)
                        total_toks = response.usage_metadata.total_token_count or 0
                        
                        usage = {
                            'input': input_toks,
                            'output': candidate_toks + thought_toks, 
                            'total': total_toks
                        }
                        
                return content, usage

            except Exception as e:
                print(f"LLM Error ({self.provider}): {e}")
                return "{}", usage

        return {}, usage