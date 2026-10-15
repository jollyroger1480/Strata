import type { MonitorState, Session } from './types';

export function chatMetrics(session:Session|null, monitor:MonitorState|null, configuredContext?:number) {
  const messages=session?.messages??[];
  const replies=messages.filter(m=>m.role==='assistant'&&!m.timeSkipId);
  const latest=[...replies].reverse().find(m=>m.inputTokens!==undefined);
  const remote=replies.at(-1)?.remoteModel===true;
  const active=!remote&&monitor?.context?.sessionId===session?.id&&!!session;
  const used=active?monitor!.context.usedTokens:latest?latest.inputTokens!+(latest.tokens??0):null;
  const max=remote?(replies.at(-1)?.contextMax||null):configuredContext??(active?monitor!.context.maxTokens:latest?.contextMax??null);
  const streaming=replies.find(m=>m.streaming);
  const speed=streaming?.tps??[...replies].reverse().find(m=>m.tps!==undefined)?.tps??null;
  const ttfts=replies.flatMap(m=>m.ttftMs===undefined?[]:[m.ttftMs]);
  const hasUsage=session?.statsAvailable===true;
  return {streaming:!!streaming,estimated:streaming?.usageSource==='estimated',hasReplies:replies.length>0,example:!!session?.showcase||replies.some(m=>m.usageSource==='demo'),
    speed:active&&monitor?.phase==='generating'?monitor.outputTps:speed,
    total:session?.statsAvailable===false?null:hasUsage?session.tokensIn+session.tokensOut+(streaming?.tokens??0)+(streaming?.inputTokens??0):replies.every(m=>m.inputTokens!==undefined&&m.tokens!==undefined)?replies.reduce((n,m)=>n+m.inputTokens!+m.tokens!,0):null,
    input:hasUsage?session.tokensIn:null,output:hasUsage?session.tokensOut:null,
    llmMs:replies.reduce((n,m)=>n+(m.durationMs??0),0),
    toolMs:replies.reduce((n,m)=>n+(m.tools??[]).reduce((v,t)=>v+(t.durationMs??0),0),0),
    ttft:ttfts.length?ttfts.reduce((n,v)=>n+v,0)/ttfts.length:null,
    used,max,pct:used!==null&&max?Math.min(100,100*used/max):null,
    turns:messages.filter(m=>m.role==='user').length,
    steps:messages.reduce((n,m)=>n+(m.tools?.length??0),0)};
}
export const compactTokens=(n:number|null)=>n===null?'—':new Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(n);
export function elapsed(ms:number){if(ms<1000)return `${Math.round(ms)} ms`;const sec=Math.round(ms/1000);return sec<60?`${sec}s`:`${Math.floor(sec/60)}m ${sec%60}s`;}
