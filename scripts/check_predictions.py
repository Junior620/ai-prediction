from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv("e:/prediction/.env")
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
r = sb.table("predictions").select("created_at,horizon,predicted_price,model_version").order("created_at", desc=True).limit(50).execute()

print(f"Total rows: {len(r.data)}")
seen = set()
for d in r.data:
    key = d["created_at"][:16]
    if key not in seen:
        seen.add(key)
        h1 = [x for x in r.data if x["created_at"][:16] == key and x["horizon"] == 1]
        h7 = [x for x in r.data if x["created_at"][:16] == key and x["horizon"] == 7]
        h30 = [x for x in r.data if x["created_at"][:16] == key and x["horizon"] == 30]
        p1 = h1[0]["predicted_price"] if h1 else "?"
        p7 = h7[0]["predicted_price"] if h7 else "?"
        p30 = h30[0]["predicted_price"] if h30 else "?"
        model = d["model_version"]
        print(f"  {key} | model={model} | 1d=${p1} | 7d=${p7} | 30d=${p30}")
