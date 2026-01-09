from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Placeholder API")

# Define the data structure
class Product(BaseModel):
    id: int
    name: str
    description: str
    price: float
    in_stock: bool

# Initial mock data
products_db = [
    {"id": 1, "name": "Wireless Mouse", "description": "Ergonomic 2.4GHz mouse", "price": 25.99, "in_stock": True},
    {"id": 2, "name": "Mechanical Keyboard", "description": "RGB Backlit Blue Switches", "price": 75.50, "in_stock": True},
    {"id": 3, "name": "USB-C Hub", "description": "7-in-1 adapter for laptops", "price": 42.00, "in_stock": False},
]

@app.get("/")
async def root():
    return {"status": "Online", "message": "Welcome to your Placeholder API"}

# GET: Fetch all products
@app.get("/products", response_model=List[Product])
async def get_products():
    return products_db

# GET: Fetch a single product by ID
@app.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: int):
    product = next((item for item in products_db if item["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

# POST: Add a new product
@app.post("/products", status_code=201)
async def create_product(product: Product):
    products_db.append(product.dict())
    return product

# DELETE: Remove a product
@app.delete("/products/{product_id}")
async def delete_product(product_id: int):
    global products_db
    products_db = [item for item in products_db if item["id"] != product_id]
    return {"message": f"Product {product_id} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)