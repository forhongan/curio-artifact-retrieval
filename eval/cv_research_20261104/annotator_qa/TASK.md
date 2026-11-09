# Independent pixel annotation QA

User explicitly requests independent GPT-6.1 Terra image annotation. Work in F:/Curio. Read only this file, docs/VISUAL_LABELING.md, and your own directory aliases.json/sheet_01.jpg through sheet_04.jpg. Do not read catalogs, blind_map, other annotations or predictions.

Actually VIEW each sheet with tools.view_image. There are24 images (6 per sheet). After each sheet append its6 rows to this directory labels.jsonl, UTF8. All aliases exactly once. JSON keys: alias, object_type (visible form vocabulary from guideline), colors {blue,white,green,red,black,yellow,brown,pink}:0/1/null, shape {narrow_mouth,wide_mouth,handles,footed,round_body}:0/1/null, confidence(high/medium/low), uncertainty(text), annotator='gpt-6.1-terra-qa'. No captions/motifs needed.

Color excludes background, grayscale cannot establish colors. Narrow mouth <=half max belly width or distinct thin neck; wide mouth open bowl/plate; round_body is a bulging spherical/ovoid belly, not a flat circular plate/mirror. Handles visible ears or handles; footed means raised legs/pedestal, not ordinary low ring foot. Unknown null, not a guessed negative. Do not infer material/era. Report completion and uncertainty.

This is an independent annotation repeatability audit. Do not read any previous annotations. Your labels do not replace or tune the frozen evaluation labels.
