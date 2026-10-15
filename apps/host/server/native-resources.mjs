/** Headroom is a worker/memory budget, not a hard utilisation throttle. */
export function resourceSettings(config,hardware,profile){
 if(!['work','performance'].includes(profile))throw new Error('Choose a resource preference.');
 const args=[...(config.args??[])];
 const remove=flag=>{for(let at=args.indexOf(flag);at>=0;at=args.indexOf(flag))args.splice(at,2);};
 const value=flag=>{const at=args.indexOf(flag);return at>=0?Number(args[at+1])||0:0;};
 const baseline=config.ui_resource_baseline??{workers:value('--pool-workers'),reserve:Math.max(Number(config.vram_reserve_mib)||0,value('--vram-reserve-mib'))};
 if(profile==='performance'&&config.ui_resource_profile==='work'){
  remove('--pool-workers');if(baseline.workers>0)args.push('--pool-workers',String(baseline.workers));
 }
 const cores=Number(hardware?.cpu?.cores)||0;
 if(profile==='work'&&cores>1){
  const at=args.indexOf('--pool-workers'),previous=at>=0?Number(args[at+1]):0;
  const workers=Math.max(1,Math.floor(cores*.8));remove('--pool-workers');
  args.push('--pool-workers',String(previous>0?Math.min(previous,workers):workers));
 }
 // Preserve calibrated worker counts and the installation's safety/vision reserve.
 const safety=Math.max(config.vision?2048:700,baseline.reserve);
 const placement=Array.isArray(config.gpu)?config.gpu:[config.gpu??0];
 const gpuMemory=(hardware?.gpus??[]).filter(g=>placement.includes(g.index)).map(g=>g.vram_gb).filter(n=>Number.isFinite(n)&&n>0);
 const reserve=profile==='work'&&gpuMemory.length?Math.max(safety,Math.floor(Math.min(...gpuMemory)*1024*.2)):safety;
 remove('--vram-reserve-mib');args.push('--vram-reserve-mib',String(reserve));
 return {args,vram_reserve_mib:reserve,ui_resource_baseline:baseline};
}
