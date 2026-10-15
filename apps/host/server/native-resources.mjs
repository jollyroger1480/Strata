/** Headroom is a worker/memory budget, not a hard utilisation throttle. */
export function resourceSettings(config,hardware,profile,model={}){
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
 const placement=Array.isArray(config.gpu)?config.gpu:[config.gpu??0];
 const gpuMemory=(hardware?.gpus??[]).filter(g=>placement.includes(g.index)).map(g=>g.vram_gb).filter(n=>Number.isFinite(n)&&n>0);
 /* The engine warns below ~1-1.5 GiB of free VRAM (the driver then pages GPU memory to system RAM and decoding falls to a few tokens a
    second, #2065), so neither profile may reserve less: 1536 MiB on a card of 16 GB or less, 1024 above; with pictures 2048. */
 const floor=gpuMemory.length&&Math.min(...gpuMemory)>16?1024:1536;
 const safety=Math.max(config.vision?2048:floor,baseline.reserve);
 const reserve=profile==='work'&&gpuMemory.length?Math.max(safety,Math.floor(Math.min(...gpuMemory)*1024*.2)):safety;
 remove('--vram-reserve-mib');args.push('--vram-reserve-mib',String(reserve));
 // RAM: "leave 20% for my work" keeps the model's RAM within 80% of the PC. A model whose experts do not fit there (the
 // setup rule: experts plus 10 GB) keeps only that many GiB of them in RAM and reads the rest from the model files
 // (--resident-budget-gib, which implies --mmap-experts). Full performance takes the budget out again.
 if(baseline.ramBudget)remove('--resident-budget-gib');
 const ram=Number(hardware?.ram_gb)||0,arena=Number(model?.arenaGb)||0;let ramBudget=false;
 if(profile==='work'&&ram>0&&arena>0&&!['--resident-experts','--mmap-experts','--resident-budget-gib'].some(f=>args.includes(f))&&ram*.8<arena+10){
  args.push('--resident-budget-gib',String(Math.max(8,Math.min(Math.floor(ram*.8)-10,Math.floor(arena/1.073741824)))));ramBudget=true;
 }
 return {args,vram_reserve_mib:reserve,ui_resource_baseline:{...baseline,ramBudget}};
}
