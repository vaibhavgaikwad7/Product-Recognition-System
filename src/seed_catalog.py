"""Seed the SQLite catalog with one row per Fashion-MNIST class.

Run:
    python -m src.seed_catalog
"""
from .catalog import init_db, list_products, upsert_product

# class_id -> (sku, name, category, price, description)
SEED = {
    0: ("TOP-1001", "Classic Cotton T-Shirt",  "Tops",       19.99, "Soft crew-neck tee"),
    1: ("BTM-2001", "Slim Fit Trousers",        "Bottoms",    49.99, "Tailored slim trousers"),
    2: ("OUT-3001", "Wool Pullover Sweater",    "Outerwear",  64.99, "Warm knit pullover"),
    3: ("DRS-4001", "A-Line Casual Dress",      "Dresses",    54.99, "Comfortable everyday dress"),
    4: ("OUT-3002", "Long Wool Coat",           "Outerwear", 129.99, "Mid-length winter coat"),
    5: ("FTW-5001", "Leather Sandals",          "Footwear",   34.99, "Open-toe leather sandals"),
    6: ("TOP-1002", "Button-Down Shirt",        "Tops",       44.99, "Cotton long-sleeve shirt"),
    7: ("FTW-5002", "Running Sneakers",         "Footwear",   79.99, "Cushioned athletic sneakers"),
    8: ("ACC-6001", "Canvas Tote Bag",          "Accessories", 29.99, "Everyday canvas tote"),
    9: ("FTW-5003", "Leather Ankle Boots",      "Footwear",   99.99, "Side-zip ankle boots"),
}


def seed() -> None:
    init_db()
    for class_id, (sku, name, category, price, desc) in SEED.items():
        upsert_product(sku, class_id, name, category, price, desc)
    rows = list_products()
    print(f"Seeded {len(rows)} products:")
    for r in rows:
        print(f"  [{r['class_id']:>2}] {r['sku']:<10} {r['name']:<28} {r['category']:<12} ${r['price']:.2f}")


if __name__ == "__main__":
    seed()
