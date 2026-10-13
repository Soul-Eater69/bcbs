import os
import jwt
import requests

SEC_TOKEN_VERIFY_URL = os.getenv(
    "SEC_TOKEN_VERIFY_URL",
    "https://t-stv-ig-uat-svc.test.hcscint.net/v6/sec-token-verify",
)

def verify_with_sec_token_verify(token: str) -> dict:
    token = token.removeprefix("Bearer ").strip()
    try:
        resp = requests.post(SEC_TOKEN_VERIFY_URL, data={"token": token}, timeout=5)
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise jwt.InvalidTokenError("Token verification service unavailable") from exc

    if resp.status_code != 200 or str(body.get("active")).lower() != "true":
        raise jwt.InvalidTokenError("Token is not active")

    # Service verified signature, alg, iss, exp. Read the claims DeliA needs.
    return jwt.decode(token, options={"verify_signature": False})
