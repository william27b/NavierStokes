/* Everything heavy runs off the UI thread, on WebGPU or single-threaded WASM. */
importScripts('./vendor/ort.webgpu.min.js','./graph.js');
let session=null,config=null,backend='wasm',modelBytes=null,cache=new Map(),gpuError='';
ort.env.wasm.numThreads=1; // Works on GitHub Pages without COOP/COEP headers.
ort.env.wasm.wasmPaths=new URL('./vendor/',self.location.href).href;
ort.env.logLevel='error';
async function initialize(forceCPU=false){
  config=await(await fetch('./model.json')).json();
  postMessage({type:'progress',text:'Downloading model and runtime…'});
  modelBytes=await(await fetch('./model.onnx')).arrayBuffer();
  if(!forceCPU&&self.navigator.gpu){try{const adapter=await navigator.gpu.requestAdapter();if(adapter){postMessage({type:'progress',text:'Preparing WebGPU…'});session=await ort.InferenceSession.create(modelBytes,{executionProviders:['webgpu'],graphOptimizationLevel:'all'});backend='webgpu';}}catch(error){gpuError=String(error);console.warn('WebGPU unavailable; using WebAssembly.',error);}}
  if(!session){postMessage({type:'progress',text:'Preparing CPU fallback…'});try{session=await ort.InferenceSession.create(modelBytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});}catch(error){throw new Error((gpuError?'GPU: '+gpuError+'; ':'')+'CPU: '+String(error));}backend='wasm';}
  postMessage({type:'ready',config:{...config,device:backend}});
}
async function run(message){
  const start=performance.now();let g;try{g=FlowGraph.build(message.solid);}catch(error){error.drawing=true;throw error;}const graphMs=performance.now()-start;
  const key=Array.from(g.solid).join('');if(cache.has(key)){postMessage({type:'prediction',...cache.get(key),revision:message.revision,total_ms:performance.now()-start,cached:true});return;}
  const feeds={};for(const[name,t]of Object.entries(g.feeds))feeds[name]=new ort.Tensor(t.type,t.data,t.dims);
  let outputs,modelStart=performance.now();
  try{
  try{outputs=await session.run(feeds);}catch(error){if(backend!=='webgpu')throw error;postMessage({type:'progress',text:'Switching to CPU compatibility mode…'});await session.release();session=await ort.InferenceSession.create(modelBytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});backend='wasm';postMessage({type:'backend',device:backend});outputs=await session.run(feeds);}
  const data=outputs.prediction.data,pressure=new Array(4096).fill(null),ux=new Array(4096).fill(null),uy=new Array(4096).fill(null);let maxSpeed=0;
  for(let i=0;i<g.fluid.length;i++){const cell=g.fluid[i];if(!Number.isFinite(data[3*i]+data[3*i+1]+data[3*i+2]))throw new Error('The model returned nonfinite fields for this drawing.');ux[cell]=data[3*i]*.05;uy[cell]=data[3*i+1]*.05;pressure[cell]=data[3*i+2]*.0025+1/3;maxSpeed=Math.max(maxSpeed,Math.hypot(ux[cell],uy[cell]));}
  const result={nx:64,ny:64,pressure,ux,uy,fluid_cells:g.fluid.length,hierarchy_nodes:g.nodes,graph_ms:graphMs,model_ms:performance.now()-modelStart,max_speed:maxSpeed,device:backend};cache.set(key,result);if(cache.size>12)cache.delete(cache.keys().next().value);
  postMessage({type:'prediction',...result,revision:message.revision,total_ms:performance.now()-start,cached:false});
  }finally{for(const t of Object.values(feeds))t.dispose();if(outputs)for(const t of Object.values(outputs))t.dispose();}
}
self.onmessage=async event=>{const message=event.data;try{if(message.type==='init')await initialize(message.forceCPU);else if(message.type==='predict'){if(!session)throw new Error('Model is still loading.');await run(message);}}catch(error){postMessage({type:'error',revision:message.revision,error:error.message||String(error),drawing:!!error.drawing,initialization:message.type==='init'});}};
