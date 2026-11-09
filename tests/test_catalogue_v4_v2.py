import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("catalogue_v4_v2", SCRIPTS / "build_cv_v4_catalogue_v2.py")
builder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder_module)


def fixture(raw, *, source="cleveland", item_id="cma_101719", split="train", group=None):
    item = {
        "item_id": item_id, "source": source, "source_id": str(raw.get("id") or raw.get("objectID")),
        "split": split, "object_group": group or item_id,
        "image_path": f"data/processed/images/{item_id}.jpg", "image_sha256": "0" * 64,
        "source_url": "https://example.test/object",
    }
    return item


def build(tmp_path, raw, *, item_id=None, split="train", group=None):
    source = raw_source(raw)
    raw_dir = tmp_path / "data/raw" / source
    raw_dir.mkdir(parents=True, exist_ok=True)
    item = fixture(raw, source=source, item_id=item_id or f"{source}_{raw.get('id') or raw.get('objectID')}",
                   split=split, group=group)
    (raw_dir / f"{item['source_id']}.json").write_text(json.dumps(raw), encoding="utf-8")
    return builder_module.CatalogueBuilderV2(tmp_path), item


def raw_source(raw):
    return "met" if "objectID" in raw else "cleveland"


def cma_raw(**changes):
    raw = {
        "id": 101719, "title": "Snuff Bottle (stopper)", "type": "Glass",
        "technique": "glass", "support_materials": [], "culture": ["China, Qing dynasty"],
        "creation_date": "1700s", "description": None, "did_you_know": None,
        "artists_tags": [], "accession_number": "1920.1", "related_works": [],
    }
    raw.update(changes)
    return raw


def test_cma_technique_is_not_material_and_stopper_is_component(tmp_path):
    raw = cma_raw(technique="carved wood", support_materials=["wood"])
    builder, item = build(tmp_path, raw)
    label = builder.build_one(item)
    derived = label["derived_ontology"]
    assert derived["material"] == ["wood"]
    assert {facet["value"] for facet in derived["technique_facets"]} == {"carved"}
    assert derived["component_role"]["value"] == "stopper"
    assert derived["object_type"] == ["stopper"]
    assert derived["photographed_subject"]["image_verified"] is False


def test_cma_explicit_technique_material_is_used_for_broad_type(tmp_path):
    jewelry = cma_raw(id=125743, title="Buckle", type="Jewelry", technique="agate")
    builder, item = build(tmp_path, jewelry)
    assert builder.build_one(item)["derived_ontology"]["material"] == ["agate"]

    textile = cma_raw(id=155786, title="Brocade with Hares", type="Textile",
                      technique="Silk: tabby, brocaded; gold thread")
    builder, item = build(tmp_path, textile)
    assert builder.build_one(item)["derived_ontology"]["material"] == ["textile"]

    wood_stand = cma_raw(id=135011, title="Covered Jar (stand)", type="Wood",
                         technique="carved wood", support_materials=[])
    builder, item = build(tmp_path, wood_stand)
    assert builder.build_one(item)["derived_ontology"]["material"] == ["wood"]


def test_desc_other_objects_and_red_tea_remain_unknown(tmp_path):
    raw = cma_raw(
        title="Bowl", id=76525, type="Ceramic", technique="White stoneware with ivory-white glaze",
        description=("This bowl has an ivory-white glaze. Red tea was best in pale green bowls. "
                     "Other bowls typically had dark glazes."),
    )
    builder, item = build(tmp_path, raw)
    derived = builder.build_one(item)["derived_ontology"]
    assert "white" in derived["colors"]
    assert "red" not in derived["colors"]
    assert "green" not in derived["colors"]
    assert "black" not in derived["colors"]
    assert {row["reason"] for row in derived["unattributed_mentions"]} >= {
        "named_non_surface_phrase", "non_object_color_subject",
    }


def test_component_and_stand_material_remain_item_level(tmp_path):
    raw = cma_raw(
        title="Covered Jar (stand)", id=135011, type="Wood",
        technique="porcelain with blue underglaze decoration", support_materials=[],
        description=None,
    )
    builder, item = build(tmp_path, raw)
    derived = builder.build_one(item)["derived_ontology"]
    assert derived["object_type"] == ["stand"]
    assert derived["component_role"]["parent_form_facets"][0]["value"] == "jar"
    assert derived["material"] == ["wood"]
    assert "porcelain" not in derived["material"]
    assert {"underglaze"} <= {facet["value"] for facet in derived["technique_facets"]}


def test_component_parent_uses_title_head_and_coating_colors_stay_separate(tmp_path):
    raw = cma_raw(
        id=140271,
        title="Stand for Screen with European Figures (obverse) and Landscape (reverse)",
        type="Metalwork",
        technique="cast bronze",
        description=None,
    )
    builder, item = build(tmp_path, raw)
    derived = builder.build_one(item)["derived_ontology"]
    assert derived["object_type"] == ["stand"]
    assert [facet["value"] for facet in derived["component_role"]["parent_form_facets"]] == ["screen"]
    assert derived["colors"] == []

    ink = cma_raw(
        id=121892,
        title="Box with Ink Cakes: Green Ink Cake in Shape of Coiled Dragon",
        type="Miscellaneous",
        technique="Molded ink in green",
        description=None,
    )
    builder, item = build(tmp_path, ink)
    derived = builder.build_one(item)["derived_ontology"]
    assert derived["object_type"] == ["ink cake"]
    assert derived["component_role"]["parent_form_facets"][0]["value"] == "box"
    assert "green" in derived["colors"]
    assert "green" in derived["coating_colors"]


def test_met_material_support_excludes_enamel_and_keeps_coral_stopper(tmp_path):
    raw = {
        "objectID": 41314, "title": "Snuff bottle with camel wrangler and Manchu horse rider",
        "objectName": "Snuff bottle", "classification": "Snuff Bottles", "period": "Qing dynasty",
        "medium": "Porcelain with overglazed enamel colors (Jingdezhen ware), coral stopper",
        "tags": [], "objectDate": "18th century",
    }
    builder, item = build(tmp_path, raw)
    derived = builder.build_one(item)["derived_ontology"]
    assert derived["material"] == ["porcelain", "coral"]
    assert "enamel" not in derived["material"]
    assert {"enameled"} <= {facet["value"] for facet in derived["technique_facets"]}


def test_mixed_material_records_do_not_erase_primary_evidence(tmp_path):
    raw = cma_raw(title="Ceremonial Uniform: Back Metal Plate", type="Arms and Armor",
                  technique="metal (brass?); lined in blue silk")
    builder, item = build(tmp_path, raw)
    materials = builder.build_one(item)["derived_ontology"]["material"]
    assert set(materials) == {"metal", "silk"}

    raw = {"objectID": 48103, "title": "Bowl", "objectName": "Bowl",
           "classification": "Ceramics",
           "medium": "Grey porcelaneous ware with silver band on edge (Jian ware?)",
           "tags": []}
    builder, item = build(tmp_path, raw)
    materials = builder.build_one(item)["derived_ontology"]["material"]
    assert set(materials) == {"ceramic", "silver"}


def test_object_use_and_religious_context_are_not_motifs(tmp_path):
    raw = cma_raw(title="Bird feeder", type="Ceramic", technique="stoneware",
                  description="Birds are traditionally kept in cages.")
    builder, item = build(tmp_path, raw)
    derived = builder.build_one(item)["derived_ontology"]
    assert "bird" not in derived["motifs"]
    assert any(x["reason"] == "object_use_not_motif" for x in derived["unattributed_mentions"])

    for title, description in (("Buddhist Wheel of the Law", ""),
                               ("Sutra text", "This Buddhist text was translated by a monk.")):
        raw = cma_raw(title=title, type="Miscellaneous", technique="ink", description=description)
        builder, item = build(tmp_path, raw)
        assert "figures" not in builder.build_one(item)["derived_ontology"]["motifs"]


def test_broad_sculpture_class_does_not_imply_human_figure(tmp_path):
    raw = cma_raw(title="Miniature Votive Stupa", type="Sculpture", technique="steatite")
    builder, item = build(tmp_path, raw)
    assert "figure" not in builder.build_one(item)["derived_ontology"]["object_type"]


def supplement_base(task, tmp_path):
    image = tmp_path / task["image_path"]
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    task["image_sha256"] = builder_module.digest(image)
    record = tmp_path / task["official_catalog"]["record_path"]
    return {
        "item_id": task["item_id"], "image_path": task["image_path"],
        "image_sha256": builder_module.digest(image),
        "catalog_record_path": task["official_catalog"]["record_path"],
        "catalog_record_sha256": builder_module.digest(record),
        "reviewer": "fixture", "created_at": "2026-11-08T00:00:00Z",
    }


def test_ai_supplement_accepts_no_claims_with_reason(tmp_path):
    raw = cma_raw(id=101719, title="Snuff Bottle with Dragon (stopper)")
    builder, item = build(tmp_path, raw)
    task = builder.build_one(item)
    labels = {task["item_id"]: task}
    base = supplement_base(task, tmp_path)
    result = builder_module.validate_supplement_payload(
        {**base, "claims": [], "no_supplement_reason": "Record and image are sufficient."}, labels, tmp_path
    )
    assert result["accepted"]
    assert result["claims"] == []
    missing_reason = builder_module.validate_supplement_payload(
        {**base, "claims": []}, labels, tmp_path
    )
    assert not missing_reason["accepted"]
    assert missing_reason["errors"][0]["code"] == "no_supplement_reason_required"


def test_ai_supplement_records_visible_observation_with_evidence(tmp_path):
    raw = cma_raw(id=101719, title="Snuff Bottle with Dragon (stopper)")
    builder, item = build(tmp_path, raw)
    task = builder.build_one(item)
    labels = {task["item_id"]: task}
    base = supplement_base(task, tmp_path)
    claim = {
        "field": "visible_motifs", "value": "dragon",
        "catalogue_evidence": {
            "field": "title", "quote": "Snuff Bottle with Dragon (stopper)",
            "relation": "provides_context",
        },
        "image_evidence": {"region": "front wall", "observation": "A blue dragon is clearly visible."},
        "observation_status": "image_only_obvious",
        "training_role": "candidate_append_only",
        "visibility": "clear", "confidence": 0.95,
    }
    result = builder_module.validate_supplement_payload({**base, "claims": [claim]}, labels, tmp_path)
    assert result["accepted"]
    assert result["claims"][0]["value"] == "dragon"
    assert task["official_catalog"]["title"] == "Snuff Bottle with Dragon (stopper)"
    assert task["ai_visible_supplement"] is None


def test_ai_supplement_rejects_unchecked_or_rewritten_catalogue(tmp_path):
    raw = cma_raw(id=101719, title="Snuff Bottle with Dragon (stopper)")
    builder, item = build(tmp_path, raw)
    task = builder.build_one(item)
    labels = {task["item_id"]: task}
    base = supplement_base(task, tmp_path)
    claim = {
        "field": "visible_motifs", "value": "dragon",
        "catalogue_evidence": {
            "field": "title", "quote": "Invented title", "relation": "supports_claim"
        },
        "image_evidence": {"region": "front wall", "observation": "A dragon is visible."},
        "observation_status": "image_only_obvious",
        "training_role": "candidate_append_only",
        "visibility": "clear", "confidence": 0.95,
    }
    result = builder_module.validate_supplement_payload({**base, "claims": [claim]}, labels, tmp_path)
    assert not result["accepted"]
    assert result["errors"][0]["code"] == "catalogue_evidence_not_in_record"
    old = {
        **base, "claims": [{"field": "visible_motifs", "value": "dragon",
                             "evidence": "Visible dragon", "catalog_checked": True,
                             "observation_status": "catalogue_confirmed_visible",
                             "training_role": "candidate_append_only",
                             "visibility": "clear", "confidence": 0.95}],
    }
    assert not builder_module.validate_supplement_payload(old, labels, tmp_path)["accepted"]
    assert task["derived_ontology"]["motifs"] == ["dragon"]
    assert task["ai_visible_supplement"] is None


def test_ai_supplement_allows_whitespace_normalized_exact_quote(tmp_path):
    raw = cma_raw(id=101719, title="Snuff Bottle:  Dragon")
    builder, item = build(tmp_path, raw)
    task = builder.build_one(item)
    labels = {task["item_id"]: task}
    base = supplement_base(task, tmp_path)
    claim = {
        "field": "visible_motifs", "value": "dragon",
        "catalogue_evidence": {
            "field": "title", "quote": "Snuff Bottle: Dragon", "relation": "provides_context"
        },
        "image_evidence": {"region": "front", "observation": "A dragon is visible."},
        "observation_status": "image_only_obvious",
        "training_role": "candidate_append_only",
        "visibility": "clear", "confidence": 0.95,
    }
    result = builder_module.validate_supplement_payload({**base, "claims": [claim]}, labels, tmp_path)
    assert result["accepted"]


def test_ai_supplement_rejects_paraphrased_punctuation(tmp_path):
    raw = cma_raw(id=101719, title="Snuff Bottle: Dragon")
    builder, item = build(tmp_path, raw)
    task = builder.build_one(item)
    base = supplement_base(task, tmp_path)
    claim = {
        "field": "visible_motifs", "value": "dragon",
        "catalogue_evidence": {
            "field": "title", "quote": "Snuff Bottle Dragon", "relation": "provides_context"
        },
        "image_evidence": {"region": "front", "observation": "A dragon is visible."},
        "observation_status": "image_only_obvious",
        "training_role": "candidate_append_only",
        "visibility": "clear", "confidence": 0.95,
    }
    result = builder_module.validate_supplement_payload(
        {**base, "claims": [claim]}, {task["item_id"]: task}, tmp_path
    )
    assert not result["accepted"]
    assert result["errors"][0]["code"] == "catalogue_evidence_not_in_record"


def test_ai_supplement_rejects_missing_visibility_evidence(tmp_path):
    raw = cma_raw(id=101719, title="Snuff Bottle with Dragon (stopper)")
    builder, item = build(tmp_path, raw)
    task = builder.build_one(item)
    labels = {task["item_id"]: task}
    base = supplement_base(task, tmp_path)
    claim = {
        "field": "visible_motifs", "value": "dragon",
        "catalogue_evidence": {
            "field": "title", "quote": "Snuff Bottle with Dragon (stopper)",
            "relation": "provides_context"
        },
        "image_evidence": {"region": "front"},
        "observation_status": "image_only_obvious",
        "training_role": "candidate_append_only",
        "visibility": "clear", "confidence": 0.95,
    }
    result = builder_module.validate_supplement_payload({**base, "claims": [claim]}, labels, tmp_path)
    assert not result["accepted"]
    assert any(error["code"] == "image_evidence_incomplete" for error in result["errors"])


def test_ai_supplement_rejects_partial_visibility_even_in_model_phase(tmp_path):
    raw = cma_raw(id=101719, title="Snuff Bottle with Dragon (stopper)")
    builder, item = build(tmp_path, raw)
    task = builder.build_one(item)
    base = supplement_base(task, tmp_path)
    claim = {
        "field": "visible_motifs", "value": "dragon",
        "catalogue_evidence": {
            "field": "title", "quote": "Snuff Bottle with Dragon (stopper)",
            "relation": "provides_context",
        },
        "image_evidence": {"region": "front", "observation": "A partial motif is visible."},
        "observation_status": "catalogue_confirmed_visible",
        "training_role": "candidate_append_only",
        "visibility": "partial", "confidence": 0.95,
    }
    result = builder_module.validate_supplement_payload(
        {**base, "claims": [claim]}, {task["item_id"]: task}, tmp_path, phase="model"
    )
    assert not result["accepted"]
    assert any(error["code"] == "invalid_visibility" for error in result["errors"])


def test_ai_supplement_distinguishes_catalogue_confirmed_and_obvious_image_only(tmp_path):
    raw = cma_raw(id=101719, title="Snuff Bottle (stopper)")
    builder, item = build(tmp_path, raw)
    task = builder.build_one(item)
    base = supplement_base(task, tmp_path)
    claim = {
        "field": "visible_palette", "value": ["blue", "white"],
        "catalogue_evidence": {
            "field": "title", "quote": "Snuff Bottle (stopper)",
            "relation": "does_not_state_property",
        },
        "image_evidence": {"region": "front", "observation": "Blue decoration on a white body is clearly visible."},
        "observation_status": "image_only_obvious",
        "training_role": "candidate_append_only", "visibility": "clear", "confidence": 0.9,
    }
    result = builder_module.validate_supplement_payload(
        {**base, "claims": [claim]}, {task["item_id"]: task}, tmp_path
    )
    assert result["accepted"]
    claim["observation_status"] = "catalogue_confirmed_visible"
    result = builder_module.validate_supplement_payload(
        {**base, "claims": [claim]}, {task["item_id"]: task}, tmp_path
    )
    assert not result["accepted"]
    assert any(error["code"] == "observation_status_conflicts_with_catalogue_evidence"
               for error in result["errors"])


def test_candidate_matches_cover_named_concepts(tmp_path):
    raw = cma_raw(
        id=1, title="Covered Flask with Painted Scene and Medallion", type="Ceramic",
        technique="painted, carved relief, incised",
        description="This pair of two vessels shown standing with blue-and-white glaze and patterned surface.",
    )
    builder, item = build(tmp_path, raw)
    label = builder.build_one(item)
    candidates = builder_module.ontology_candidates([label])
    by_value = {row["candidate"]: row for row in candidates}
    assert by_value["flask"]["item_coverage"] == 1
    assert by_value["covered"]["item_coverage"] == 1
    assert by_value["painted"]["item_coverage"] == 1
    assert by_value["carved"]["item_coverage"] == 1
    assert by_value["scene"]["item_coverage"] == 1
    assert by_value["medallion"]["item_coverage"] == 1
