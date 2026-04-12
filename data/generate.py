"""
PawsIQ — Synthetic Data Generator
===================================
Produces three CSVs that mirror the production PostgreSQL schema:
  - bookings.csv       ~2 years of walk/service bookings
  - gps_traces.csv     per-booking GPS pings along NJ routes
  - reviews.csv        client reviews with star ratings + text

Run:
    python generate.py

Outputs land in ./data/synthetic/
"""

import os
import random
import csv
from datetime import datetime, timedelta

import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Output dir ───────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "synthetic")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
START_DATE   = datetime(2023, 1, 1)
END_DATE     = datetime(2024, 12, 31)
N_CLIENTS    = 60
N_WALKERS    = 8
N_PETS       = 70

# ── NJ zip codes with approximate lat/lng centroids ──────────────────────────
NJ_ZIPS = [
    ("07017", 40.7651, -74.2099),
    ("07018", 40.7537, -74.2321),
    ("07103", 40.7282, -74.2107),
    ("07104", 40.7587, -74.1624),
    ("07106", 40.7432, -74.2365),
    ("07108", 40.7151, -74.2154),
    ("07201", 40.6634, -74.2107),
    ("07208", 40.6734, -74.1987),
    ("07302", 40.7178, -74.0431),
    ("07305", 40.6987, -74.0876),
    ("07042", 40.8098, -74.2026),
    ("07043", 40.8293, -74.1943),
    ("07052", 40.8051, -74.2543),
    ("07079", 40.7565, -74.2654),
    ("07083", 40.6998, -74.2612),
]

# ── Service types ─────────────────────────────────────────────────────────────
SERVICES = {
    "walk_30":   {"base_price": 16.00, "duration_min": 30,  "weight": 0.45},
    "walk_60":   {"base_price": 24.00, "duration_min": 60,  "weight": 0.30},
    "drop_in":   {"base_price": 14.00, "duration_min": 20,  "weight": 0.15},
    "overnight": {"base_price": 55.00, "duration_min": 720, "weight": 0.10},
}
SERVICE_NAMES   = list(SERVICES.keys())
SERVICE_WEIGHTS = [SERVICES[s]["weight"] for s in SERVICE_NAMES]

BREEDS = [
    ("Labrador Retriever", "high"), ("Golden Retriever", "high"),
    ("French Bulldog", "low"),      ("German Shepherd", "high"),
    ("Poodle", "medium"),           ("Shiba Inu", "high"),
    ("Chihuahua", "low"),           ("Beagle", "medium"),
    ("Dachshund", "low"),           ("Border Collie", "high"),
    ("Pug", "low"),                 ("Boxer", "high"),
    ("Maltese", "low"),             ("Siberian Husky", "high"),
    ("Corgi", "medium"),
]

POSITIVE_REVIEWS = [
    "Absolutely love {walker}! {pet} is always tired and happy after walks.",
    "{walker} is so reliable and sends great photo updates every time.",
    "Best walker we've ever had. {pet} gets so excited when {walker} arrives.",
    "{walker} goes above and beyond. Highly recommend to anyone in the area.",
    "5 stars every time. {pet} has improved so much with consistent walks.",
    "Always on time and so communicative. {walker} is a gem.",
    "{pet} absolutely adores {walker}. We feel so lucky to have found her.",
    "Professional, caring, and thorough. {walker} sends detailed reports after each visit.",
    "Our anxious dog warmed up to {walker} immediately. That says everything.",
    "{walker} treats {pet} like her own. We couldn't ask for more.",
]

NEUTRAL_REVIEWS = [
    "{walker} is good. {pet} seems comfortable with them.",
    "Solid service, no complaints. {pet} always comes back walked.",
    "Reliable and punctual. Does what's asked, communication is fine.",
    "Good experience overall. {pet} was well taken care of.",
    "Nothing to complain about. Would use again.",
    "{walker} is professional. Walks are on schedule.",
    "Decent service. {pet} seems happy enough after walks.",
    "Fine experience. Gets the job done.",
]

NEGATIVE_REVIEWS = [
    "Walk was cut short. {pet} came back with energy still to burn.",
    "{walker} arrived 20 minutes late with no heads up.",
    "Communication could be better. Didn't hear back for hours.",
    "Felt rushed. Would have appreciated more care with {pet}.",
    "Expected GPS updates but didn't receive any for this walk.",
    "Not a bad walker but not consistent. Some days great, some days not.",
]

FIRST_NAMES = [
    "Amara","Devon","Cleo","Jordan","Priya","Chris","Natalie","Marcus",
    "Sofia","Elijah","Zoe","Aiden","Maya","Owen","Layla","Ethan",
    "Imani","Tyler","Keisha","Ryan","Destiny","Alex","Brianna","Caleb",
    "Jasmine","Noah","Tiffany","Malik","Savannah","Kyle","Aaliyah","Sean",
    "Morgan","Jason","Unique","Derek","Camille","Justin","Monique","Travis",
    "Simone","Brandon","Whitney","Darius","Alexis","Kevin","Dominique","Eric",
    "Jade","Trevor","Kayla","Anthony","Kendra","Michael","Ciara","James",
    "Ashley","Robert","Diamond","William","Brittany","David",
]

LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson",
    "White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
    "Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell",
    "Carter","Roberts","Patel","Kim","Washington","Evans","Collins","Edwards",
]

PET_NAMES = [
    "Mochi","Bella","Max","Luna","Charlie","Daisy","Buddy","Molly","Rocky","Lola",
    "Cooper","Sadie","Bear","Maggie","Tucker","Sophie","Duke","Chloe","Zeus","Penny",
    "Biscuit","Ruby","Murphy","Rosie","Oscar","Lily","Jack","Zoe","Toby","Nala",
    "Gus","Coco","Sam","Gracie","Leo","Stella","Winston","Roxy","Bentley","Ellie",
    "Harley","Abby","Bruno","Lucy","Roscoe","Willow","Louie","Piper","Finn","Nova",
    "Bandit","Ginger","Ollie","Scout","Koda","Honey","Jax","Pepper","Axel","Mia",
    "Thor","Layla","Dexter","Xena","Simba","Angel","Shadow","Izzy","Rex","Leia",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rand_date_between(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def demand_weight(dt):
    hour  = dt.hour
    dow   = dt.weekday()
    month = dt.month

    if 7 <= hour <= 9:       h = 1.8
    elif 17 <= hour <= 19:   h = 1.5
    elif 11 <= hour <= 13:   h = 0.9
    elif 6 <= hour <= 20:    h = 0.7
    else:                    h = 0.05

    d = 1.2 if dow < 5 else 0.8

    if month in (3,4,5,9,10):   s = 1.3
    elif month in (6,7,8):      s = 1.0
    else:                        s = 0.75

    return h * d * s

def surge_multiplier(dt, zip_demand):
    base = demand_weight(dt) * zip_demand
    raw  = np.clip(base / 2.0, 0, 1)
    return round(0.85 + raw * 0.50, 4)

def jitter_coord(lat, lng, radius_km=0.5):
    dlat = (random.random() - 0.5) * 2 * (radius_km / 111.0)
    dlng = (random.random() - 0.5) * 2 * (radius_km / (111.0 * abs(np.cos(np.radians(lat)))))
    return round(lat + dlat, 6), round(lng + dlng, 6)

def sample_time_of_day():
    hours   = list(range(6, 21))
    weights = [demand_weight(datetime(2024, 6, 3, h)) for h in hours]
    total   = sum(weights)
    probs   = [w / total for w in weights]
    return int(np.random.choice(hours, p=probs))

def write_csv(path, rows):
    if not rows:
        print(f"  [SKIP] {path} — no rows")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):,} rows  →  {os.path.basename(path)}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Clients
# ─────────────────────────────────────────────────────────────────────────────
print("Generating users...")

clients = []
for i in range(N_CLIENTS):
    first   = random.choice(FIRST_NAMES)
    last    = random.choice(LAST_NAMES)
    zip_row = random.choice(NJ_ZIPS)
    joined  = rand_date_between(START_DATE, START_DATE + timedelta(days=180))
    clients.append({
        "user_id":   f"C{i+1:04d}",
        "name":      f"{first} {last}",
        "email":     f"{first.lower()}.{last.lower()}{random.randint(1,99)}@email.com",
        "role":      "client",
        "zip":       zip_row[0],
        "lat":       zip_row[1],
        "lng":       zip_row[2],
        "joined_at": joined.isoformat(),
    })

# ─────────────────────────────────────────────────────────────────────────────
# 2. Walkers
# ─────────────────────────────────────────────────────────────────────────────
walkers = []
walker_first = ["Amara","Devon","Cleo","Jordan","Priya","Chris","Natalie","Marcus"]
walker_last  = ["Stevens","Park","Mitchell","Rivera","Okafor","Laurent","Diaz","Chen"]
for i in range(N_WALKERS):
    zip_row = random.choice(NJ_ZIPS)
    walkers.append({
        "user_id":     f"W{i+1:04d}",
        "name":        f"{walker_first[i]} {walker_last[i]}",
        "email":       f"{walker_first[i].lower()}.{walker_last[i].lower()}@pawsiq.com",
        "role":        "walker",
        "zip":         zip_row[0],
        "lat":         zip_row[1],
        "lng":         zip_row[2],
        "joined_at":   START_DATE.isoformat(),
        "rating":      round(random.uniform(4.3, 5.0), 1),
        "total_walks": 0,
    })

# ─────────────────────────────────────────────────────────────────────────────
# 3. Pets
# ─────────────────────────────────────────────────────────────────────────────
print("Generating pets...")

pets = []
client_to_pets = {c["user_id"]: [] for c in clients}

for i in range(N_PETS):
    eligible = [c for c in clients if len(client_to_pets[c["user_id"]]) < 2]
    if not eligible:
        eligible = clients
    owner = random.choice(eligible)
    breed, energy = random.choice(BREEDS)
    weight_lbs = max(5, min(120, int(np.random.normal(35, 18))))
    pet = {
        "pet_id":    f"P{i+1:04d}",
        "owner_id":  owner["user_id"],
        "name":      random.choice(PET_NAMES),
        "breed":     breed,
        "weight_lb": weight_lbs,
        "energy":    energy,
        "age_years": round(random.uniform(0.5, 12.0), 1),
    }
    pets.append(pet)
    client_to_pets[owner["user_id"]].append(pet["pet_id"])

for c in clients:
    if not client_to_pets[c["user_id"]]:
        client_to_pets[c["user_id"]].append(random.choice(pets)["pet_id"])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Bookings
# ─────────────────────────────────────────────────────────────────────────────
print("Generating bookings...")

FREQ_WEEKLY = {"low": 0.4, "medium": 1.2, "high": 2.8}
client_freq = {c["user_id"]: random.choice(["low","medium","high"]) for c in clients}

bookings   = []
booking_id = 1
current    = START_DATE

while current <= END_DATE:
    for client in clients:
        avg  = FREQ_WEEKLY[client_freq[client["user_id"]]]
        n    = np.random.poisson(avg)
        for _ in range(n):
            day_offset   = random.randint(0, 6)
            book_date    = current + timedelta(days=day_offset)
            if book_date > END_DATE:
                continue
            hour         = sample_time_of_day()
            minute       = random.choice([0, 15, 30, 45])
            scheduled_at = book_date.replace(hour=hour, minute=minute, second=0)
            service      = np.random.choice(SERVICE_NAMES, p=SERVICE_WEIGHTS)
            svc          = SERVICES[service]
            same_zip     = [w for w in walkers if w["zip"] == client["zip"]]
            walker       = random.choice(same_zip) if same_zip else random.choice(walkers)
            pet_id       = random.choice(client_to_pets[client["user_id"]])
            zip_demand   = 0.8 + random.random() * 0.4
            surge        = surge_multiplier(scheduled_at, zip_demand)
            price        = round(svc["base_price"] * surge, 2)
            r            = random.random()
            status       = "completed" if r < 0.93 else ("cancelled" if r < 0.97 else "no_show")

            bookings.append({
                "booking_id":       f"B{booking_id:06d}",
                "client_id":        client["user_id"],
                "walker_id":        walker["user_id"],
                "pet_id":           pet_id,
                "service_type":     service,
                "scheduled_at":     scheduled_at.isoformat(),
                "duration_min":     svc["duration_min"],
                "zip":              client["zip"],
                "lat":              client["lat"],
                "lng":              client["lng"],
                "base_price":       svc["base_price"],
                "surge_multiplier": surge,
                "final_price":      price,
                "status":           status,
                "hour_of_day":      hour,
                "day_of_week":      scheduled_at.weekday(),
                "month":            scheduled_at.month,
                "is_peak_hour":     int(7 <= hour <= 9 or 17 <= hour <= 19),
                "is_weekend":       int(scheduled_at.weekday() >= 5),
            })

            if status == "completed":
                walker["total_walks"] += 1
            booking_id += 1

    current += timedelta(days=7)

print(f"  Generated {len(bookings):,} bookings")

# ─────────────────────────────────────────────────────────────────────────────
# 5. GPS traces
# ─────────────────────────────────────────────────────────────────────────────
print("Generating GPS traces...")

gps_rows = []
gps_id   = 1
walk_bookings = [b for b in bookings
                 if b["status"] == "completed"
                 and b["service_type"] in ("walk_30","walk_60")]

for b in walk_bookings:
    start_dt = datetime.fromisoformat(b["scheduled_at"])
    n_pings  = b["duration_min"] // 2
    lat, lng = jitter_coord(b["lat"], b["lng"], radius_km=0.1)
    heading  = random.uniform(0, 360)

    for ping in range(n_pings):
        ping_time = start_dt + timedelta(minutes=ping * 2)
        heading  += random.gauss(0, 15)
        rad       = np.radians(heading)
        lat      += 0.001 * np.cos(rad)
        lng      += 0.001 * np.sin(rad)
        if lat < 40.5 or lat > 41.0:  heading = (heading + 180) % 360
        if lng < -74.5 or lng > -73.8: heading = (heading + 180) % 360

        gps_rows.append({
            "gps_id":      f"G{gps_id:08d}",
            "booking_id":  b["booking_id"],
            "walker_id":   b["walker_id"],
            "lat":         round(lat, 6),
            "lng":         round(lng, 6),
            "speed_mph":   round(max(0.5, random.gauss(3.2, 0.5)), 2),
            "heading_deg": round(heading % 360, 1),
            "recorded_at": ping_time.isoformat(),
            "ping_index":  ping,
        })
        gps_id += 1

print(f"  Generated {len(gps_rows):,} GPS pings")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Reviews
# ─────────────────────────────────────────────────────────────────────────────
print("Generating reviews...")

completed   = [b for b in bookings if b["status"] == "completed"]
review_id   = 1
reviews     = []
walker_names = {w["user_id"]: w["name"].split()[0] for w in walkers}
pet_names    = {p["pet_id"]: p["name"] for p in pets}

for b in completed:
    if random.random() > 0.72:
        continue
    wname   = walker_names.get(b["walker_id"], "your walker")
    pname   = pet_names.get(b["pet_id"], "your dog")
    rating  = int(np.random.choice([1,2,3,4,5], p=[0.02,0.03,0.07,0.22,0.66]))

    if rating >= 4:
        sentiment = "positive"
        template  = random.choice(POSITIVE_REVIEWS)
    elif rating == 3:
        sentiment = "neutral"
        template  = random.choice(NEUTRAL_REVIEWS)
    else:
        sentiment = "negative"
        template  = random.choice(NEGATIVE_REVIEWS)

    body      = template.format(walker=wname, pet=pname)
    walk_dt   = datetime.fromisoformat(b["scheduled_at"])
    posted_at = walk_dt + timedelta(hours=random.randint(1, 72))

    reviews.append({
        "review_id":       f"R{review_id:06d}",
        "booking_id":      b["booking_id"],
        "client_id":       b["client_id"],
        "walker_id":       b["walker_id"],
        "rating":          rating,
        "body":            body,
        "sentiment_label": sentiment,
        "posted_at":       posted_at.isoformat(),
    })
    review_id += 1

print(f"  Generated {len(reviews):,} reviews")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Write CSVs
# ─────────────────────────────────────────────────────────────────────────────
print("\nWriting CSVs...")
write_csv(os.path.join(OUT_DIR, "bookings.csv"),   bookings)
write_csv(os.path.join(OUT_DIR, "gps_traces.csv"), gps_rows)
write_csv(os.path.join(OUT_DIR, "reviews.csv"),    reviews)
write_csv(os.path.join(OUT_DIR, "clients.csv"),    clients)
write_csv(os.path.join(OUT_DIR, "walkers.csv"),    walkers)
write_csv(os.path.join(OUT_DIR, "pets.csv"),       pets)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary
# ─────────────────────────────────────────────────────────────────────────────
completed_count = sum(1 for b in bookings if b["status"] == "completed")
total_revenue   = sum(b["final_price"] for b in bookings if b["status"] == "completed")
avg_surge       = float(np.mean([b["surge_multiplier"] for b in bookings]))
peak_bookings   = sum(1 for b in bookings if b["is_peak_hour"])

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PawsIQ Synthetic Dataset Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Date range:      {START_DATE.date()} → {END_DATE.date()}
 Clients:         {len(clients)}
 Walkers:         {len(walkers)}
 Pets:            {len(pets)}
 Total bookings:  {len(bookings):,}
   Completed:     {completed_count:,}  ({completed_count/len(bookings)*100:.1f}%)
   Peak-hour:     {peak_bookings:,}  ({peak_bookings/len(bookings)*100:.1f}%)
 Total revenue:   ${total_revenue:,.2f}
 Avg surge mult:  x{avg_surge:.3f}
 GPS pings:       {len(gps_rows):,}
 Reviews:         {len(reviews):,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
