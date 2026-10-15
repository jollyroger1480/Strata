/** Prepared vision files stay in the source config; activation belongs to a loaded instance. */
export function runtimeVision(config,requested){
 const mode=requested??(config.ui_vision_enabled===false?'no':config.ui_vision_device);
 if(mode!==undefined&&!['yes','no','cpu','gpu'].includes(mode))throw new Error('Choose whether to enable picture reading.');
 if(mode==='no')return undefined;
 if(mode===undefined)return config.vision;
 if(!config.vision)throw new Error('Prepare this model’s vision files before enabling picture reading.');
 return {...config.vision,...(mode==='cpu'?{gpu:false}:mode==='gpu'?{gpu:true}:{})};
}
