from j2jmodule import decode_to_json

raw = """37 03 00 251002090935 C11 251002090859 ND20251002090902 NCN906 JD////////////// JN/// 212 N399 E1409 010 36 03 RK44519 RT00/// RC0//// 9999="""

json_payload = decode_to_json(raw, ensure_ascii=False, indent=2)
print(json_payload)
