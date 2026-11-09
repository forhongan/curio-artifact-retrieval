"""Real loopback API acceptance against saved full-gallery decisions."""
import json
from pathlib import Path
import shutil

import numpy as np
import requests
from scipy.special import expit

from curio.represent.encoders import Encoder
from curio.retrieve.attribute_cv import attribute_scores, parse_attributes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/relevance_v5'


def main():
    url='http://127.0.0.1:8000'
    session=requests.Session()
    health=session.get(url+'/api/health',timeout=10).json()
    assert health['indexed_images']==4692
    queries=[json.loads(s) for s in (ROOT/'eval/relevance_v5/queries.jsonl').read_text(encoding='utf-8').splitlines()]
    column=next(i for i,q in enumerate(queries) if q['qid']=='v5_green_vase')
    query=queries[column]['text_en']
    ids=json.loads((ROOT/'artifacts/siglip_adapted/item_ids.json').read_text(encoding='utf-8'))
    scores=np.load(OUT/'scores.npz')
    thresholds=json.loads((OUT/'thresholds.json').read_text(encoding='utf-8'))
    # Match the online single-query FP16 encoding contract. Offline batching can
    # move threshold-adjacent examples slightly; measure this rather than hiding it.
    manifest=json.loads((ROOT/'artifacts/siglip_adapted/manifest.json').read_text())
    encoder=Encoder('siglip',adapter_path=ROOT/manifest['adapter_path'])
    vector=encoder.encode_texts([query])[0]
    cached_vector=np.load(OUT/'text_vectors.npz')['siglip'][column]
    text_cosine=float(vector@cached_vector/(np.linalg.norm(vector)*np.linalg.norm(cached_vector)))
    assert text_cosine > .99999,text_cosine
    raw_error_bound=float(np.linalg.norm(vector/np.linalg.norm(vector)-cached_vector/np.linalg.norm(cached_vector)))+1e-6
    generic_vector=encoder.encode_texts(['a photograph of a museum object'])[0]
    vectors=np.load(ROOT/'artifacts/siglip_adapted/image_embeddings.npy')
    vectors=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    raw=vectors@(vector/np.linalg.norm(vector))
    generic=vectors@(generic_vector/np.linalg.norm(generic_vector))
    normalized=(raw-raw.mean())/max(float(raw.std()),1e-6)
    directory=ROOT/'artifacts/cv_research_20261104/fresh'
    feature_ids=json.loads((ROOT/'artifacts/classical_cv/item_ids.json').read_text())
    lookup={v:i for i,v in enumerate(feature_ids)}
    predictions=np.load(directory/'raw_color_predictions.npy')[[lookup[v] for v in ids]]
    heads=np.load(directory/'raw_color_heads.npz')
    auxiliary,_=attribute_scores(predictions,list(heads['classes']),parse_attributes(query))
    weight=json.loads((OUT/'color_binding.json').read_text())['weight']
    color=normalized+weight*(auxiliary-auxiliary.mean())/max(float(auxiliary.std()),1e-6)
    config=json.loads((OUT/'boundary_production_lora_en.json').read_text())
    logistic=expit(np.column_stack([raw,raw-generic,normalized])@np.array(config['coefficients'])+config['intercept'])
    replay={'dense':raw,'aux_raw_color':color,'calibrated':logistic}
    checks=[]
    for method,key in [('dev_cutoff','dense'),('dev_color','aux_raw_color'),('dev_calibrated','calibrated')]:
        response=session.post(url+'/api/search/set',json={'query':query,'method':method,'top_k':1},timeout=90)
        response.raise_for_status()
        result=response.json()
        score_key=f'production_lora/{key}/en'
        expected={ids[i] for i in np.flatnonzero(replay[key]>=thresholds[score_key])}
        saved={ids[i] for i in np.flatnonzero(scores[score_key][:,column]>=thresholds[score_key])}
        delta=float(np.max(np.abs(replay[key]-scores[score_key][:,column])))
        assert delta < {'dense':raw_error_bound,'aux_raw_color':.005,'calibrated':.001}[key],(method,delta)
        actual=set(result['matched_item_ids'])
        assert actual==expected,(method,sorted(actual^expected))
        assert len(result['results'])==result['matched_count'] and len(result['preview'])<=1
        assert result['matched_count']+result['rejected_count']==4692
        checks.append({'method':method,'matched':len(actual),'matches_single_query_replay':True,
                       'offline_batch_decision_differences':sorted(actual^saved),
                       'max_score_difference_from_offline_batch':delta,'top_k_is_preview_only':True})
    reference_root=ROOT/'artifacts/dynamic_reference_api'
    cache=reference_root/'cache/http'
    shutil.copytree(ROOT/'artifacts/qualified_reference_v5/qianlong/http',cache,dirs_exist_ok=True)
    response=session.post(url+'/api/search/reference/dynamic',json={
        'query':'Qianlong famille rose vase','mode':'qualified','reference_limit':8,'top_k':5,'offline':True},timeout=90)
    response.raise_for_status()
    reference=response.json()
    assert reference['status']=='ranked' and reference['reference_count']==8
    assert reference['gallery_count']==4692 and len(reference['results'])==5
    assert reference['candidate_catalogue_used_for_ranking'] is False
    for row in reference['references']:
        image=session.get(url+row['image_url'],timeout=10)
        image.raise_for_status()
        assert image.headers['content-type'].startswith('image/')
        assert row['download_verified'] and row['source_support']['status']=='source_supported'
        assert row['attribution']
    report={'status':'passed','text_batch_vs_single_cosine':text_cosine,'dense_numerical_error_bound':raw_error_bound,'set_checks':checks,'reference':{k:reference.get(k) for k in ['audit_id','reference_count','gallery_count','status','acquisition']},
            'reference_images_served_and_verified':8,'catalogue_ranking':False}
    (OUT/'api_acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
