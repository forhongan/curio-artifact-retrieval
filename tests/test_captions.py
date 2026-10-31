from curio.represent.captions import visual_caption


def test_no_date_culture_or_source_leakage():
    item = {
        "title": "Vase from the eighteenth century, Ming dynasty, 1644–1911",
        "object_name": "Vase",
        "medium": "Porcelain",
        "period": "Qing",
        "culture": "China",
        "tags": ["Qianlong"],
        "motifs": ["dragon"],
    }
    caption = visual_caption(item)
    assert "vase" in caption and "dragon" in caption and "porcelain" in caption
    assert all(
        x not in caption.lower()
        for x in ["qing", "ming", "1644", "china", "qianlong", "eighteenth", "century"]
    )
