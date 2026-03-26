import json

with open("IDMT-19761_field_check_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

fields = data.get("fields", {})

for k, v in fields.items():
    if not k.startswith("customfield_"):
        continue
    if v is None or v == [] or v == "":
        continue

    print(f"\n{k}")
    try:
        print(json.dumps(v, indent=2, ensure_ascii=False)[:1000])
    except Exception:
        print(str(v)[:1000])