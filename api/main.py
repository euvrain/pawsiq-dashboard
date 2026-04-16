from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import demand, pricing, walkers, bookings

app = FastAPI(
    title="PawsIQ API",
    description="ML-powered pet services platform backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(demand.router,   prefix="/predict", tags=["ML — Demand"])
app.include_router(pricing.router,  prefix="/predict", tags=["ML — Pricing"])
app.include_router(walkers.router,  prefix="/walkers", tags=["Walkers"])
app.include_router(bookings.router, prefix="/bookings",tags=["Bookings"])

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "pawsiq-api"}
