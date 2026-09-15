"""La guía existe y cubre lo importante (que no se desactualice en silencio)."""
from pathlib import Path

def test_guide_exists_and_covers():
    g = Path("docs/guia-usuario.md").read_text(encoding="utf-8").lower()
    for kw in ("hodl", "backend", "data.json", "disclaimer", "honest",
               "lora", "pendiente", "--backend", "exit", "fees"):
        assert kw in g, f"guía no cubre: {kw}"

def test_readme_links_guide():
    assert "guia-usuario" in Path("README.md").read_text(encoding="utf-8")
