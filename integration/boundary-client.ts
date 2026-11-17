/** Copy this client into your frontend. No page, layout, or styling dependency. */
export type Evidence = {condition:string; label:string; source?:string; status:string; value?:string; reason?:string; mouth_body_ratio?:number; ratio_limit?:number};
export type BoundaryItem = {item_id:string; title:string; rank:number; yes:boolean; score:number; image_url:string; thumb_url:string; condition_evidence:Evidence[]};
export type BoundaryJob = {job_id:string; status:'queued'|'running'|'complete'|'failed'|'cancelled'; complete:boolean; processed:number; total:number; matched_count:number; rejected_count?:number; results?:BoundaryItem[]; uncertain_count?:number; notices?:string[]; execution_mode?:'snapshot_replay'; live_inference?:boolean; error?:string};
export function boundaryClient(baseUrl = '') {
  const base = baseUrl.replace(/\/$/, '');
  async function json<T>(path:string, init?:RequestInit):Promise<T> {
    const r = await fetch(base + path, init);
    const data = await r.json();
    if (!r.ok) throw new Error(typeof data.detail==='string' ? data.detail : JSON.stringify(data.detail ?? data));
    return data as T;
  }
  const get = (id:string) => json<BoundaryJob>('/api/search/set/jobs/'+encodeURIComponent(id));
  const cancel = (id:string) => json<BoundaryJob>('/api/search/set/jobs/'+encodeURIComponent(id)+'/cancel',{method:'POST'});
  return {
    imageUrl:(path:string) => /^https?:\/\//.test(path) ? path : base + path,
    get, cancel,
    detail:(id:string) => json<Record<string,unknown>>('/api/items/'+encodeURIComponent(id)),
    async search(query:string, onProgress?:(job:BoundaryJob)=>void, signal?:AbortSignal) {
      if(signal?.aborted) throw new DOMException('Aborted','AbortError');
      // Obtain the job ID even if cancellation arrives during creation, so the
      // backend job can be cancelled instead of being abandoned in the queue.
      let job = await json<BoundaryJob>('/api/search/set/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,method:'catalogue_visual'})});
      try {
        while(true) {
          if(signal?.aborted) throw new DOMException('Aborted','AbortError');
          onProgress?.(job);
          if(job.status==='failed') throw new Error(job.error || 'Boundary inference failed');
          if(job.status==='cancelled') throw new DOMException('Cancelled','AbortError');
          if(job.complete) return job;
          await new Promise(resolve=>setTimeout(resolve,1000));
          job=await get(job.job_id);
        }
      } finally {
        if(!job.complete && signal?.aborted) await cancel(job.job_id).catch(()=>{});
      }
    }
  };
}
