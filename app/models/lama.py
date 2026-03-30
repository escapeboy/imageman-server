from app.model_registry import registry


def get_lama():
    cached = registry.get("lama")
    if cached:
        return cached
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()
    registry.put("lama", lama)
    return lama
