'use strict';
const channel=new BroadcastChannel('curio-polyu-presentation-v1');
const current=document.getElementById('current'),next=document.getElementById('nextSlide');
let state=null,base='',styles='';
function send(action){channel.postMessage({type:'command',action});}
function preview(index,revealed){
  const doc=new DOMParser().parseFromString(base,'text/html');
  const slide=doc.querySelectorAll('.slide')[index];if(!slide)return '';
  slide.className='slide active';slide.removeAttribute('style');
  if(index===6){const q=slide.querySelector('#questionValue'),v=slide.querySelector('#measuredValue');if(q)q.hidden=revealed;if(v)v.hidden=!revealed;}
  slide.querySelectorAll('[hidden]').forEach(el=>{if(el.hidden)el.style.display='none';});
  const href=new URL('index.html',location.href).href;
  return '<!doctype html><html><head><meta charset="utf-8"><base href="'+href+'"><style>'+styles+'\nhtml,body{overflow:hidden!important}.slide{visibility:visible!important;opacity:1!important;transform:none!important}.slide [data-r]{opacity:1!important;transform:none!important;transition:none!important}.bfill{width:var(--w)!important}.qline{stroke-dashoffset:0!important;animation:none!important}.qdot{opacity:1!important;animation:none!important}</style></head><body><div id="stage" style="transform:none;left:0;top:0">'+slide.outerHTML+'</div></body></html>';
}
function resize(){document.querySelectorAll('.screen').forEach(s=>{const f=s.querySelector('.scaled-slide'),scale=Math.min(s.clientWidth/1440,s.clientHeight/810);if(f){f.style.transform='scale('+scale+')';f.style.left=((s.clientWidth-1440*scale)/2)+'px';f.style.top=((s.clientHeight-810*scale)/2)+'px';}});}
function time(seconds){return String(Math.floor(seconds/60)).padStart(2,'0')+':'+String(seconds%60).padStart(2,'0')+' / 07:00';}
channel.onmessage=e=>{const m=e.data;if(!m||m.type!=='state')return;state=m;
  document.getElementById('counter').textContent=String(m.idx+1).padStart(2,'0')+' / 14';document.getElementById('slideTitle').textContent=m.title;
  document.getElementById('clock').textContent=time(m.elapsed);document.getElementById('timer').textContent=m.running?'暂停计时':'开始计时';
  document.getElementById('en').textContent=m.notes.en;document.getElementById('zh').textContent=m.notes.zh;
  if(base){current.srcdoc=preview(m.idx,m.resultRevealed);next.srcdoc=m.idx<13?preview(m.idx+1,false):'';}
};
for(const id of ['currentScreen','nextScreen']){const f=document.createElement('div');f.className='scaled-slide';document.getElementById(id).append(f);f.append(id==='currentScreen'?current:next);}
fetch('index.html').then(r=>r.text()).then(html=>{base=html;styles=[...new DOMParser().parseFromString(html,'text/html').querySelectorAll('style')].map(s=>s.textContent).join('\n');channel.postMessage({type:'ready'});});
document.getElementById('next').onclick=()=>send('next');document.getElementById('prev').onclick=()=>send('prev');document.getElementById('timer').onclick=()=>send('timer');document.getElementById('fullscreen').onclick=()=>send('fullscreen');
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();if(!e.repeat)send('next');}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();send('prev');}else if(e.key.toLowerCase()==='f')send('fullscreen');else if(e.key==='Home')send('home');else if(e.key==='End')send('end');});
window.addEventListener('resize',resize);new ResizeObserver(resize).observe(document.querySelector('main'));
window.addEventListener('beforeunload',()=>channel.postMessage({type:'bye'}));
