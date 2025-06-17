from pydantic import BaseModel, Field
from typing import Optional, List

from chainette import (
    Step,
    Chain,
    Branch,
    SamplingParams,
    register_engine,
)


class RawDesc(BaseModel):
    """Incoming noisy e‑commerce text."""

    text: str = Field(..., description="messy product description")
    source: Optional[str] = Field(None, description="source of the product description")
    confidence: Optional[float] = Field(None, description="confidence in raw data quality")


class Attr(BaseModel):
    """Clean, structured product attributes."""

    brand: str
    model: str
    price_eur: float
    category: Optional[str] = None
    features: Optional[List[str]] = None
    availability: Optional[str] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

register_engine(
    name="llama3",
    model="meta-llama/Llama-3.2-3B-Instruct",
    dtype="float16",
    gpu_memory_utilization=0.95,
    lazy=True,
    max_model_len=2048
)

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

extract = Step(
    id="extract",
    name="Extract attributes",
    input_model=RawDesc,
    output_model=Attr,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0),
    system_prompt=(
        "You are a specialized e-commerce data extraction system. Parse the provided product description "
        "to identify key product attributes. Analyze carefully for brand name, exact model designation, "
        "and price in euros. When possible, also extract product category, key features, and availability."
    ),
    user_prompt="Product description: {{text}}\nSource: {{source}}\n\nExtract all structured product information.",
)

fr = Step(
    id="fr",
    name="French version",
    input_model=Attr,
    output_model=Attr,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.3),
    system_prompt=(
        "You are a professional French e-commerce content localizer. Translate the product information to "
        "French while maintaining the precision of technical specifications. Ensure brand names remain "
        "recognizable while adapting model names only when there are standard French equivalents. "
        "Keep the price value unchanged."
    ),
    user_prompt=(
        "Produit à localiser en français:\n"
        "- Marque: {{brand}}\n"
        "- Modèle: {{model}}\n"
        "- Prix: {{price_eur}} €\n"
        "Catégorie: {{category}}\n"
        "Caractéristiques: {{features}}\n"
        "Disponibilité: {{availability}}"
    ),
)

es = Step(
    id="es",
    name="Spanish version",
    input_model=Attr,
    output_model=Attr,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.3),
    system_prompt=(
        "You are a professional Spanish e-commerce content localizer. Translate the product information to "
        "Spanish while maintaining the precision of technical specifications. Ensure brand names remain "
        "recognizable while adapting model names only when there are standard Spanish equivalents. "
        "Keep the price value unchanged."
    ),
    user_prompt=(
        "Producto para localizar en español:\n"
        "- Marca: {{brand}}\n"
        "- Modelo: {{model}}\n"
        "- Precio: {{price_eur}} €\n"
        "Categoría: {{category}}\n"
        "Características: {{features}}\n"
        "Disponibilidad: {{availability}}"
    ),
)

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

chain = Chain(
    name="Extract & Translate",
    steps=[
        extract,
        [
            Branch(name="fr", steps=[fr]),
            Branch(name="es", steps=[es]),
        ],
    ],
    batch_size=4,
)


if __name__ == "__main__":
    chain.run([
        RawDesc(
            text="🔥 Apple iPhone 15 Pro, 128 Go, Titanium Finish, 48MP Camera, 999 €, In Stock",
            source="electronics-direct.com",
            confidence=0.95
        ),
        RawDesc(
            text="Samsung Galaxy S23 Ultra - Écran Dynamic AMOLED 6.8\" - 12GB RAM - 512GB - Prix: 1299€",
            source="mobile-deals.fr",
            confidence=0.89
        )
    ])