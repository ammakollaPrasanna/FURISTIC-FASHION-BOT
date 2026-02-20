import uvicorn
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Make sure this path is correct for your local machine!
app.mount("/static", StaticFiles(directory=r"C:\Users\ammak\Downloads"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class BodyRequest(BaseModel):
    body_type: str

@app.post("/color-analysis")
async def analyze_color():
    options = ["Cool Summer", "Warm Autumn", "Neutral Spring"]
    selected = random.choice(options)
    
    # Logic to check for keywords
    is_cool = "Cool" in selected
    is_warm = "Warm" in selected

    # Data mapping
    palettes = {
        "Cool": {"hex": ["#002366", "#50C878", "#C0C0C0"], "names": ["Royal Blue", "Emerald", "Silver"]},
        "Warm": {"hex": ["#FF8C00", "#8B4513", "#FFD700"], "names": ["Burnt Orange", "Saddle Brown", "Gold"]},
        "Neutral": {"hex": ["#E6E6FA", "#F5F5DC", "#708090"], "names": ["Lavender", "Beige", "Slate"]}
    }

    # Extract the correct palette based on the selection
    key = "Cool" if is_cool else "Warm" if is_warm else "Neutral"
    current_palette = palettes[key]

    return {
        "undertone": selected,
        "colors": current_palette["hex"],
        "color_names": current_palette["names"],
        "jewelry": {
            "metal": "Silver & Platinum" if is_cool else "Yellow Gold & Copper" if is_warm else "Rose Gold & Mixed Metals",
            "earrings": "Diamond Studs or Blue Sapphires" if is_cool else "Amber Drops or Gold Hoops" if is_warm else "Champagne Pearls",
            "chains": "Fine Herringbone Silver" if is_cool else "Heavy Cuban Gold Link" if is_warm else "Mixed Metal Layered Chains"
        }
    }

@app.post("/body-analysis")
async def get_body_advice(req: BodyRequest):
    base_url = "http://localhost:8000/static"
    profiles = {
        "Apple": {
            "description": "Weight is primarily carried in the midsection. Focus on deep necklines and empire waists.",
            "tops": "Empire waist, V-necks, and tunic styles.",
            "bottoms": "Straight leg or bootcut jeans.",
            "dresses": "Shift dresses and A-line silhouettes.",
            "best_fit_img": f"{base_url}/WhatsApp%20Image%202026-02-19%20at%201.18.04%20PM.jpeg"
        },
        "Hourglass": {
            "description": "Balanced bust and hips with a defined waist. Highlight your curves.",
            "tops": "Wrap tops, fitted shirts, and V-necklines.",
            "bottoms": "High-waisted pants and pencil skirts.",
            "dresses": "Bodycon and wrap dresses.",
            "best_fit_img": f"{base_url}/WhatsApp%20Image%202026-02-19%20at%201.43.53%20PM.jpeg"
        },
        "Pear": {
            "description": "Hips are wider than shoulders. Add volume to your top half.",
            "tops": "Boat necks and statement sleeves.",
            "bottoms": "A-line skirts and wide-leg trousers.",
            "dresses": "Fit and flare dresses.",
            "best_fit_img": f"{base_url}/WhatsApp%20Image%202026-02-19%20at%201.43.53%20PM%20(1).jpeg"
        },
        "Rectangle": {
            "description": "Straight silhouette. Create the illusion of curves with structure.",
            "tops": "Peplum tops and sweetheart necklines.",
            "bottoms": "Cargo pants or skirts with pockets.",
            "dresses": "Cut-out dresses and ruched styles.",
            "best_fit_img": f"{base_url}/WhatsApp%20Image%202026-02-19%20at%201.43.54%20PM%20(1).jpeg"
        },
        "Inverted Triangle": {
            "description": "Broad shoulders and narrow hips. Volume on the bottom is key.",
            "tops": "Halter necks and V-neck cardigans.",
            "bottoms": "Palazzo pants and pleated skirts.",
            "dresses": "Full skirts and A-line cuts.",
            "best_fit_img": f"{base_url}/WhatsApp%20Image%202026-02-19%20at%201.43.54%20PM.jpeg"
        }
    }
    result = profiles.get(req.body_type)
    if not result:
        raise HTTPException(status_code=404, detail="Body type not found.")
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)