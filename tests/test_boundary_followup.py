from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize("column", ["human_grade", "notes"])
def test_followup_generation_preserves_existing_human_work(tmp_path, column):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_boundary_followup.py"
    loaded = runpy.run_path(str(script))
    main = loaded["main"]
    main.__globals__["OUT"] = tmp_path
    target = tmp_path / "independent_labels_pending.csv"
    original = f"{column}\nreviewed\n"
    target.write_text(original)
    with pytest.raises(ValueError, match="overwrite existing human annotations"):
        main()
    assert target.read_text() == original
