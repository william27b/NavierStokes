/* Geometry-only hierarchy, matching multiscale_graph.py. No solver labels. */
(function(scope){
'use strict';
const NX=64,NY=64;
function slots(indices,n){const groups=Array.from({length:n},()=>[]);for(let i=0;i<indices.length;i++)groups[indices[i]].push(i);const k=Math.max(1,...groups.map(g=>g.length));const data=new Int32Array(n*k).fill(indices.length);for(let i=0;i<n;i++)data.set(groups[i],i*k);return{data,dims:[n,k],type:'int32'};}
function tensor(data,dims,type='float32'){return{data:type==='float32'?Float32Array.from(data):data,dims,type};}
function validate(mask){if(mask.length!==NX*NY)throw new Error('Expected a 64 × 64 mask.');const solid=Uint8Array.from(mask);for(let x=0;x<NX;x++){solid[x]=1;solid[(NY-1)*NX+x]=1;}for(let y=1;y<NY-1;y++){solid[y*NX]=0;solid[y*NX+NX-1]=0;}
  const seen=new Uint8Array(NX*NY),q=new Int32Array(NX*NY);let first=solid.findIndex(v=>!v),head=0,tail=1;q[0]=first;seen[first]=1;
  while(head<tail){const i=q[head++],x=i%NX,y=Math.floor(i/NX);for(const j of[x>0?i-1:-1,x<NX-1?i+1:-1,y>0?i-NX:-1,y<NY-1?i+NX:-1])if(j>=0&&!solid[j]&&!seen[j]){seen[j]=1;q[tail++]=j;}}
  if(tail!==solid.reduce((sum,v)=>sum+!v,0))throw new Error('Fluid is disconnected. Erase a gap through the barrier, or fill any enclosed fluid pocket.');return solid;
}
function distances(solid){const row=new Float64Array(NX*NY).fill(1e9),result=new Float64Array(NX*NY);
  for(let y=0;y<NY;y++){let nearest=-10000;for(let x=0;x<NX;x++){if(solid[y*NX+x])nearest=x;row[y*NX+x]=(x-nearest)**2;}nearest=10000;for(let x=NX-1;x>=0;x--){if(solid[y*NX+x])nearest=x;row[y*NX+x]=Math.min(row[y*NX+x],(x-nearest)**2);}}
  for(let y=0;y<NY;y++)for(let x=0;x<NX;x++){let best=1e9;for(let yy=0;yy<NY;yy++)best=Math.min(best,row[yy*NX+x]+(y-yy)**2);result[y*NX+x]=Math.sqrt(best);}return result;
}
function coarseLevel(positions,edges,weights,features,width){const n=weights.length,parent=Int32Array.from({length:n},(_,i)=>i);function find(i){while(parent[i]!==i){parent[i]=parent[parent[i]];i=parent[i];}return i;}
  const [src,dst]=edges;for(let e=0;e<src.length;e++){const a=src[e],b=dst[e];if(Math.floor(positions[2*a]/width)!==Math.floor(positions[2*b]/width)||Math.floor(positions[2*a+1]/width)!==Math.floor(positions[2*b+1]/width))continue;const ra=find(a),rb=find(b);if(ra!==rb)parent[Math.max(ra,rb)]=Math.min(ra,rb);}
  const ids=new Map(),mapping=new Int32Array(n);for(let i=0;i<n;i++){const r=find(i);if(!ids.has(r))ids.set(r,ids.size);mapping[i]=ids.get(r);}const m=ids.size;
  const nextPos=new Float64Array(m*2),nextFeatures=new Float64Array(m*12),nextWeights=new Float64Array(m);
  for(let i=0;i<n;i++){const c=mapping[i],w=weights[i];nextWeights[c]+=w;nextPos[2*c]+=positions[2*i]*w;nextPos[2*c+1]+=positions[2*i+1]*w;for(let j=0;j<12;j++)nextFeatures[c*12+j]+=features[i*12+j]*w;}
  for(let i=0;i<m;i++){nextPos[2*i]/=nextWeights[i];nextPos[2*i+1]/=nextWeights[i];for(let j=0;j<12;j++)nextFeatures[i*12+j]/=nextWeights[i];}
  const unique=new Set();for(let e=0;e<src.length;e++){const a=mapping[src[e]],b=mapping[dst[e]];if(a!==b)unique.add(a*m+b);}const keys=[...unique].sort((a,b)=>a-b);
  return{mapping,positions:nextPos,features:nextFeatures,weights:nextWeights,edges:[Int32Array.from(keys,k=>Math.floor(k/m)),Int32Array.from(keys,k=>k%m)]};
}
function build(mask){const solid=validate(mask),ids=new Int32Array(NX*NY).fill(-1),fluid=[];for(let i=0;i<solid.length;i++)if(!solid[i]){ids[i]=fluid.length;fluid.push(i);}const n=fluid.length,dist=distances(solid);
  let positions=new Float64Array(n*2),features=new Float64Array(n*12),weights=new Float64Array(n).fill(1);const bcMask=new Uint8Array(n*3),bcValues=new Float32Array(n*3);
  for(let i=0;i<n;i++){const cell=fluid[i],x=cell%NX,y=Math.floor(cell/NX);positions.set([x,y],2*i);features.set([x/63,y/63,+(x===0),+(x===63),x>0?solid[cell-1]:0,x<63?solid[cell+1]:0,y>0?solid[cell-NX]:0,y<63?solid[cell+NX]:0,dist[cell]/62,.01/.05,0,((.53-.5)/3)/.01],i*12);if(x===0){bcMask[i*3]=1;bcMask[i*3+1]=1;bcValues[i*3]=.2;}if(x===63){bcMask[i*3+1]=1;bcMask[i*3+2]=1;}}
  const a=[],b=[];for(let y=0;y<NY;y++)for(let x=0;x<NX-1;x++){const i=y*NX+x;if(ids[i]>=0&&ids[i+1]>=0){a.push(ids[i]);b.push(ids[i+1]);}}for(let y=0;y<NY-1;y++)for(let x=0;x<NX;x++){const i=y*NX+x;if(ids[i]>=0&&ids[i+NX]>=0){a.push(ids[i]);b.push(ids[i+NX]);}}let edges=[Int32Array.from(a.concat(b)),Int32Array.from(b.concat(a))];
  const feeds={x:tensor(features,[n,12]),bc_mask:tensor(bcMask,[n,3],'bool'),bc_values:tensor(bcValues,[n,3])},nodes=[];
  for(let k=0;k<5;k++){const count=weights.length,[src,dst]=edges,e=src.length;nodes.push(count);const edge=new Float32Array(e*2),degree=new Float32Array(count);for(let j=0;j<e;j++){edge[2*j]=(positions[2*dst[j]]-positions[2*src[j]])/(2**k);edge[2*j+1]=(positions[2*dst[j]+1]-positions[2*src[j]+1])/(2**k);degree[dst[j]]++;}for(let i=0;i<count;i++)degree[i]=Math.max(1,degree[i]);
    feeds['src'+k]=tensor(src,[e],'int32');feeds['dst'+k]=tensor(dst,[e],'int32');feeds['edge'+k]=tensor(edge,[e,2]);feeds['incoming'+k]=slots(dst,count);feeds['degree'+k]=tensor(degree,[count,1]);
    if(k>0){const geom=new Float32Array(count*13);for(let i=0;i<count;i++){geom.set(features.slice(i*12,i*12+12),i*13);geom[i*13+12]=weights[i]/(2**k)**2;}feeds['geom'+k]=tensor(geom,[count,13]);}
    if(k<4){const next=coarseLevel(positions,edges,weights,features,2**(k+1));feeds['weight'+k]=tensor(weights,[count,1]);feeds['parent_weight'+k]=tensor(next.weights,[next.weights.length,1]);feeds['map'+k]=tensor(next.mapping,[count],'int32');feeds['children'+k]=slots(next.mapping,next.weights.length);positions=next.positions;features=next.features;weights=next.weights;edges=next.edges;}
  }
  return{feeds,solid,fluid:Int32Array.from(fluid),nodes};
}
scope.FlowGraph={build,validate};if(typeof module!=='undefined')module.exports=scope.FlowGraph;
})(typeof self!=='undefined'?self:globalThis);
