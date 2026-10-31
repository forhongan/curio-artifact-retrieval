"""Conservative visual captions, separate from collection metadata search text."""

import re

# Dates/attributions are not learnable reliably from a photograph.
NONVISUAL = re.compile(
    r"\b(?:\d{3,4}|B\.?C\.?E?\.?|C\.?E\.?|century|dynasty|reign|period|"
    r"Ming|Qing|Song|Tang|Han|Yuan|Shang|Zhou|Qianlong|Kangxi|Yongzheng|"
    r"Xuande|Chenghua|Hongzhi|Zhengde|Jiajing|Longqing|Wanli|Tianqi|Chongzhen|"
    r"Shunzhi|Jiaqing|Daoguang|Xianfeng|Tongzhi|Guangxu|Xuantong|Hongwu|Yongle|"
    r"Qianlong|Kangxi|Yongzheng|Republic|Neolithic|Buddhist|Buddhism|"
    r"Jingdezhen|Longquan|Dehua|China|Chinese|museum|gift|collection|attributed)\b",
    re.I,
)


def clean_visual_phrase(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(
        r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)\s*(?:[-–]\s*)?(?:century)?\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?:明|清|唐|宋|元|汉|漢|商|周|晋|晉|隋|辽|遼|金)(?:朝|代)|(?:乾隆|康熙|雍正|中国|中國)|公元(?:前)?|\d+年",
        " ",
        text,
    )
    # Remove parenthetical accession/date notes, then individual nonvisual terms.
    text = re.sub(r"\([^)]*(?:\d{3,4}|dynasty|century|reign)[^)]*\)", " ", text, flags=re.I)
    text = NONVISUAL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:.-–")
    return text


def visual_caption(item: dict) -> str:
    object_name = clean_visual_phrase(
        item.get("object_name") or item.get("object_type") or "object"
    )
    title = clean_visual_phrase(item.get("title", ""))
    medium = clean_visual_phrase(item.get("medium", ""))
    # Title often includes the only useful observable shape/motif detail.
    subject = title if title and len(title.split()) <= 22 else object_name
    parts = [f"A photograph of {subject.lower()}"]
    if medium:
        parts.append(f"made of {medium.lower()[:180]}")
    motifs = [clean_visual_phrase(x) for x in item.get("motifs", [])]
    if any(motifs):
        parts.append("decorated with " + ", ".join(x for x in motifs[:4] if x))
    # Raw tags are intentionally excluded: museum subject tags may be semantic,
    # nonvisual or incidental. Colors derived from background are also excluded.
    return ", ".join(parts) + "."
