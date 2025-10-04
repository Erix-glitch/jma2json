import httpx
from jma2json import decode_to_json

# fetches the latest EEW report from JMA
r = httpx.get("https://api.wolfx.jp/jma_eew.json").json()

json_payload = decode_to_json(r['OriginalText'], ensure_ascii=False, indent=2)
print(json_payload)