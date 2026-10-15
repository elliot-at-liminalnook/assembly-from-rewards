/**
 * A small, dependency-free WebGL studio.
 *
 * This file ONLY renders. It cannot choose actions or write to the Q table.
 * Rust supplies discrete hand/peg coordinates; we interpolate those coordinates,
 * then solve a two-link visual arm to put the gripper in the right place.
 * The white robot is an illustrative rig, not a simulated Franka mechanism.
 *
 * Reading order: math → procedural meshes → shaders → scene → animation.
 * No downloaded models, textures, images, shaders, or rendering libraries.
 */

// ----- Small vector/matrix helpers. Matrices are column-major, as WebGL expects.
const add = (a,b) => a.map((v,i) => v+b[i]);
const sub = (a,b) => a.map((v,i) => v-b[i]);
const mul = (a,k) => a.map(v => v*k);
const dot = (a,b) => a.reduce((s,v,i) => s+v*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const length = a => Math.hypot(...a);
const unit = a => mul(a,1/(length(a)||1));
function multiply(a,b) {
  const c = new Float32Array(16);
  for(let j=0;j<4;j++) for(let i=0;i<4;i++) for(let k=0;k<4;k++) c[j*4+i]+=a[k*4+i]*b[j*4+k];
  return c;
}
function lookAt(eye,target) {
  const z=unit(sub(eye,target)), x=unit(cross([0,1,0],z)), y=cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
}
function perspective(fov,aspect,near=.1,far=40) {
  const f=1/Math.tan(fov/2), d=1/(near-far);
  return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*d,-1,0,0,2*far*near*d,0]);
}
function ortho(w,h,near=.1,far=20) {
  return new Float32Array([2/w,0,0,0,0,2/h,0,0,0,0,-2/(far-near),0,0,0,-(far+near)/(far-near),1]);
}
function transform(p,scale=[1,1,1],axis=[0,1,0]) {
  const y=unit(axis), x=unit(cross(y,Math.abs(y[2])<.95?[0,0,1]:[1,0,0])), z=cross(x,y);
  return new Float32Array([...mul(x,scale[0]),0,...mul(y,scale[1]),0,...mul(z,scale[2]),0,...p,1]);
}
function normalMatrix(m) {
  // Our transforms have orthogonal basis vectors. Dividing each by its squared
  // length is the inverse-transpose, including non-uniform scale on the links.
  const n=[];
  for(let i=0;i<3;i++) { const v=[m[i*4],m[i*4+1],m[i*4+2]]; n.push(...mul(v,1/dot(v,v))); }
  return new Float32Array(n);
}

// ----- Meshes. Every vertex contains xyz position followed by xyz normal.
function vertex(out,p,n) { out.push(...p,...n); }
function triangle(out,a,b,c) { for(const v of [a,b,c]) vertex(out,v.p,v.n); }

/** Revolve a radius/height profile around the Y axis. Used for shells, joints,
 * collars, and chamfered metal. Averaged profile tangents give smooth normals. */
function lathe(profile,segments=64,closed=false) {
  const out=[], rows=profile.length;
  const point=(j,i) => {
    const p=profile[j], prev=profile[closed?(j+rows-1)%rows:Math.max(0,j-1)], next=profile[closed?(j+1)%rows:Math.min(rows-1,j+1)];
    const angle=i/segments*Math.PI*2, c=Math.cos(angle), s=Math.sin(angle);
    const n=unit([(next[1]-prev[1])*c,prev[0]-next[0],(next[1]-prev[1])*s]);
    return {p:[p[0]*c,p[1],p[0]*s],n};
  };
  for(let j=0;j<rows-(closed?0:1);j++) for(let i=0;i<segments;i++) {
    const a=point(j,i), b=point(j,i+1), c=point((j+1)%rows,i+1), d=point((j+1)%rows,i);
    triangle(out,a,b,d); triangle(out,b,c,d);
  }
  return out;
}
function sphere() {
  const profile=[];
  for(let i=0;i<=24;i++) { const a=i*Math.PI/24; profile.push([Math.sin(a),-Math.cos(a)]); }
  return lathe(profile,48);
}
/** A rounded box is six grids projected onto a smaller box plus a sphere.
 * Small bevels catch the studio light and make machined edges legible. */
function roundedBox(size,radius,steps=8) {
  const out=[], half=size.map(v=>v/2), inner=half.map(v=>v-radius);
  for(let face=0;face<3;face++) for(const sign of [-1,1]) {
    const u=(face+1)%3,v=(face+2)%3;
    const point=(i,j) => {
      const p=[0,0,0]; p[face]=sign*half[face]; p[u]=(2*i/steps-1)*half[u]; p[v]=(2*j/steps-1)*half[v];
      const base=p.map((value,k)=>Math.max(-inner[k],Math.min(inner[k],value)));
      const n=unit(sub(p,base)); return {p:add(base,mul(n,radius)),n};
    };
    for(let i=0;i<steps;i++) for(let j=0;j<steps;j++) {
      const a=point(i,j),b=point(i+1,j),c=point(i+1,j+1),d=point(i,j+1);
      if(sign>0) { triangle(out,a,b,d); triangle(out,b,c,d); }
      else { triangle(out,a,d,b); triangle(out,b,d,c); }
    }
  }
  return out;
}

const VERTEX = `
attribute vec3 aPosition, aNormal;
uniform mat4 uModel, uVP, uLightVP;
uniform mat3 uNormal;
varying vec3 vPosition, vNormal;
varying vec4 vShadow;
void main(){
  vec4 world=uModel*vec4(aPosition,1.0);
  vPosition=world.xyz; vNormal=normalize(uNormal*aNormal);
  vShadow=uLightVP*world; gl_Position=uVP*world;
}`;
const FRAGMENT = `
precision highp float;
varying vec3 vPosition, vNormal;
varying vec4 vShadow;
uniform vec3 uColor, uEye;
uniform float uMetal, uRoughness, uFloor, uShadowSize;
uniform sampler2D uShadow;
float depth(vec4 v){return dot(v,vec4(1.0,1.0/255.0,1.0/65025.0,1.0/16581375.0));}
float visibility(vec3 n,vec3 l){
  vec3 p=vShadow.xyz/vShadow.w*.5+.5;
  if(p.x<0.0||p.x>1.0||p.y<0.0||p.y>1.0) return 1.0;
  float shadow=0.0, bias=max(.0018,.003*(1.0-dot(n,l)));
  for(int x=-2;x<=2;x++) for(int y=-2;y<=2;y++){
    float d=depth(texture2D(uShadow,p.xy+vec2(float(x),float(y))*4.5/uShadowSize));
    shadow+=p.z-bias>d ? 0.0:1.0;
  }
  return shadow/25.0;
}
void main(){
  vec3 n=normalize(vNormal), l=normalize(vec3(-3.5,7.0,4.0));
  vec3 view=normalize(uEye-vPosition), h=normalize(l+view);
  float diffuse=max(dot(n,l),0.0), shade=visibility(n,l);
  float hemi=.5+.5*n.y;
  vec3 ambient=mix(vec3(.14,.17,.19),vec3(.50,.53,.56),hemi);
  float rim=pow(1.0-max(dot(n,view),0.0),4.0);
  vec3 fresnel=mix(vec3(.045),uColor,uMetal);
  float exponent=mix(120.0,16.0,uRoughness);
  vec3 spec=fresnel*pow(max(dot(n,h),0.0),exponent)*(1.6-uRoughness)*shade;
  // Soft fill from the opposite side keeps black joints readable.
  float fill=max(dot(n,normalize(vec3(4.0,3.0,-3.0))),0.0)*.16;
  vec3 color=uColor*(ambient+diffuse*shade*.76+fill)+spec+rim*.09;
  if(uFloor>.5){
    float baseContact=exp(-dot(vPosition.xz-vec2(-.5,-.9),vPosition.xz-vec2(-.5,-.9))*5.0);
    color*=1.0-.15*baseContact;
    float fog=smoothstep(4.0,15.0,length(vPosition.xz));
    color=mix(color,vec3(.66,.69,.68),fog);
  }
  // A faint deterministic grain prevents sterile, perfectly flat surfaces.
  float grain=fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453)-.5;
  color=pow(max(color,vec3(0.0)),vec3(1.0/2.2));
  gl_FragColor=vec4(color+grain*.006,1.0);
}`;
const SHADOW_VERTEX=`attribute vec3 aPosition; uniform mat4 uModel,uVP;
void main(){gl_Position=uVP*uModel*vec4(aPosition,1.0);}`;
const SHADOW_FRAGMENT=`precision highp float;
void main(){vec4 p=fract(gl_FragCoord.z*vec4(1.0,255.0,65025.0,16581375.0));
p-=p.yzww*vec4(1.0/255.0,1.0/255.0,1.0/255.0,0.0);gl_FragColor=p;}`;

export class Studio {
  constructor(canvas, options = {}) {
    this.canvas=canvas;
    this.options=options;
    const gl=this.gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});
    if(!gl) throw new Error('This browser could not create a WebGL canvas.');
    this.program=this.compile(VERTEX,FRAGMENT);
    this.shadowProgram=this.compile(SHADOW_VERTEX,SHADOW_FRAGMENT);
    this.meshes={}; this.makeMeshes(); this.makeShadow();
    this.lightVP=multiply(ortho(7.5,7.5,.1,16),lookAt([-3.5,7,4],[0,.5,0]));
    this.pose=this.poseFor({x:2,y:1,peg:0,depth:0,held:false});
    this.yaw=-.62; this.elevation=.36; this.distance=5.8; this.detail=options.detail !== false;
    this.pending=0; this.motion=null; this.detailRect=null;
    this.resizeObserver=new ResizeObserver(()=>this.requestDraw());
    this.resizeObserver.observe(canvas);
    this.pointerControls();
    canvas.addEventListener('webglcontextlost',event=>{
      event.preventDefault(); cancelAnimationFrame(this.pending);
      const note=document.getElementById(options.noteId || 'render-note');
      if(note)note.textContent='3D view paused: graphics context lost. Reload to restore. Learning remains independent.';
    });
    canvas.addEventListener('webglcontextrestored',()=>location.reload());
    this.requestDraw();
  }

  compile(vertex,fragment) {
    const gl=this.gl;
    const shader=(type,text)=>{
      const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);
      if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
    const names=['uModel','uVP','uNormal','uLightVP','uColor','uEye','uMetal','uRoughness','uFloor','uShadow','uShadowSize'];
    return {p,position:gl.getAttribLocation(p,'aPosition'),normal:gl.getAttribLocation(p,'aNormal'),...Object.fromEntries(names.map(n=>[n,gl.getUniformLocation(p,n)]))};
  }
  mesh(name,vertices) {
    const gl=this.gl,b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.STATIC_DRAW);
    this.meshes[name]={buffer:b,count:vertices.length/6};
  }
  makeMeshes() {
    this.mesh('sphere',sphere());
    this.mesh('floor',roundedBox([40,.04,40],.01,1));
    this.mesh('board',roundedBox([3.5,.18,1.15],.045));
    this.mesh('foot',roundedBox([3.3,.09,.98],.035));
    this.mesh('palm',roundedBox([.46,.18,.28],.055));
    this.mesh('rail',roundedBox([.39,.055,.18],.012));
    this.mesh('finger',roundedBox([.065,.27,.13],.014));
    this.mesh('pad',roundedBox([.035,.15,.115],.009));
    this.mesh('badge',roundedBox([.11,.035,.013],.009,4));
    this.mesh('slot',roundedBox([.042,.008,.007],.002,2));
    this.mesh('line',roundedBox([.20,.003,.006],.002,2));
    this.mesh('base',lathe([[0,0],[.29,0],[.32,.035],[.32,.13],[.28,.17],[.21,.18],[.18,.24],[.17,.38],[.14,.43],[0,.43]]));
    this.mesh('link',lathe([[0,0],[.09,.015],[.14,.06],[.165,.16],[.15,.32],[.13,.62],[.12,.85],[.10,.96],[0,1]]));
    this.mesh('joint',lathe([[0,-.13],[.13,-.13],[.17,-.10],[.175,-.07],[.175,.07],[.17,.10],[.13,.13],[0,.13]]));
    this.mesh('cap',lathe([[0,0],[.128,0],[.135,.01],[.135,.025],[.122,.04],[0,.04]]));
    this.mesh('wrist',lathe([[0,0],[.10,0],[.15,.025],[.17,.07],[.17,.17],[.14,.21],[.10,.23],[0,.23]]));
    this.mesh('peg',lathe([[0,0],[.06,0],[.075,.015],[.075,.49],[.064,.52],[0,.52]]));
    this.mesh('collar',lathe([[.105,0],[.16,0],[.18,.012],[.18,.027],[.16,.04],[.105,.04],[.097,.03],[.097,.01]],64,true));
    this.mesh('disk',lathe([[0,0],[1,0],[1,.01],[0,.01]],64));
    this.mesh('screw',lathe([[0,0],[.025,0],[.031,.007],[.031,.014],[.025,.02],[0,.02]],24));
  }
  makeShadow() {
    const gl=this.gl;
    this.shadowSize=Math.min(this.options.shadowSize || 2048,gl.getParameter(gl.MAX_TEXTURE_SIZE));
    this.shadowTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,this.shadowTexture);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,this.shadowSize,this.shadowSize,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    this.shadowBuffer=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,this.shadowBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.shadowTexture,0);
    const depth=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,depth);gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT16,this.shadowSize,this.shadowSize);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,depth);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('The shadow framebuffer is unavailable.');
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  }

  // Map the educational state to positions. Positive Y is up. These distances
  // are visual units, not measured dimensions or new physics parameters.
  poseFor(s) {
    const pegBottom=.26+(s.held?s.y*.68:0)-s.depth*.10;
    return {handX:-1.3+s.x*.65,handY:1.12+s.y*.68-(s.held?s.depth*.10:0),pegX:-1.3+s.peg*.65,pegY:pegBottom,gap:s.held?.105:.20};
  }
  setState(s,animate=false,duration=360) {
    const target=this.poseFor(s);
    this.motion=animate?{from:{...this.pose},to:target,start:performance.now(),duration}:null;
    if(!animate)this.pose=target;
    this.canvas.setAttribute('aria-label',`3D view: hand at position ${s.x}, ${s.y?'raised':'lowered'}; peg ${s.depth===2?'fully inserted':s.depth===1?'partly inserted':s.held?'held':'on the table'}. Drag to rotate the view.`);
    this.requestDraw();
  }
  requestDraw() {
    if(!this.pending)this.pending=requestAnimationFrame(t=>{this.pending=0;this.draw(t);});
  }
  camera(view) {
    const presets={studio:[-.62,.36,5.8],front:[0,.20,6.2],overhead:[.15,1.15,6.1]};
    [this.yaw,this.elevation,this.distance]=presets[view]||presets.studio;
    this.requestDraw();
  }
  pointerControls() {
    let drag=null;
    this.canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);});
    this.canvas.addEventListener('pointermove',e=>{
      if(!drag)return;
      this.yaw-=(e.clientX-drag[0])*.008;
      this.elevation=Math.max(.12,Math.min(1.35,this.elevation+(e.clientY-drag[1])*.006));
      drag=[e.clientX,e.clientY];this.requestDraw();
    });
    this.canvas.addEventListener('pointerup',()=>{drag=null;});
    this.canvas.addEventListener('pointercancel',()=>{drag=null;});
    this.canvas.addEventListener('keydown',e=>{
      if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(e.key))return;
      e.preventDefault();
      if(e.key==='Home')this.camera('studio');
      else if(e.key==='ArrowLeft')this.yaw-=.1;
      else if(e.key==='ArrowRight')this.yaw+=.1;
      else this.elevation=Math.max(.12,Math.min(1.35,this.elevation+(e.key==='ArrowUp'?.08:-.08)));
      this.requestDraw();
    });
  }

  scene() {
    const items=[];
    const white=[.79,.81,.80], dark=[.045,.055,.061], metal=[.40,.46,.48], board=[.56,.60,.60], gold=[.87,.56,.075];
    const put=(mesh,p,color,scale=[1,1,1],axis=[0,1,0],metallic=0,rough=.45,floor=0)=>items.push({mesh,model:transform(p,scale,axis),color,metallic,rough,floor});
    put('floor',[0,-.04,0],[.55,.58,.57],[1,1,1],[0,1,0],0,.9,1);
    put('foot',[0,.075,.27],dark);
    put('board',[0,.15,.27],board,[1,1,1],[0,1,0],.45,.5);
    // Fasteners, washer seats, engraved rail ticks, and a dark bore ground the
    // toy in an assembly-fixture visual language, without adding new tasks.
    for(const x of [-1.58,1.58]) for(const z of [-.15,.69]) {
      put('disk',[x,.241,z],dark,[.045,1,.045]);
      put('screw',[x,.243,z],metal,[1,1,1],[0,1,0],.8,.3);
      put('slot',[x,.266,z],dark);
    }
    for(let i=0;i<5;i++) {
      const x=-1.3+i*.65;
      put('line',[x,.242,.68],[.20,.28,.28]);
      for(let j=0;j<=i;j++)put('disk',[x+(j-i/2)*.04,.242,.74],[.24,.31,.31],[.009,.2,.009]);
    }
    put('disk',[1.3,.241,.22],[.018,.027,.027],[.105,.2,.105]);
    put('collar',[1.3,.222,.22],metal,[1,1,1],[0,1,0],.85,.25);
    put('peg',[this.pose.pegX,this.pose.pegY,.22],gold,[1,1,1],[0,1,0],.7,.28);
    // A subtle turned-metal ring near the peg's head.
    put('disk',[this.pose.pegX,this.pose.pegY+.47,.22],[.63,.38,.045],[.076,.7,.076],[0,1,0],.75,.3);

    const shoulder=[-.5,.58,-.9], wrist=[this.pose.handX,this.pose.handY,.22];
    put('disk',[-.5,.005,-.9],dark,[.38,3,.38]);
    put('base',[-.5,.025,-.9],white);
    for(const x of [-.76,-.24])for(const z of [-1.13,-.67])put('screw',[x,.04,z],metal);

    // Two-link inverse kinematics: choose the elbow on the upward side of the
    // shoulder-to-wrist line. The learner only supplied the wrist destination.
    const vector=sub(wrist,shoulder), distance=Math.min(length(vector),2.49), direction=unit(vector);
    const upper=1.36, lower=1.16;
    const along=(upper*upper-lower*lower+distance*distance)/(2*distance);
    const away=Math.sqrt(Math.max(0,upper*upper-along*along));
    const up=unit(sub([0,1,0],mul(direction,dot(direction,[0,1,0]))));
    const elbow=add(add(shoulder,mul(direction,along)),mul(up,away));
    const axle=unit(cross(direction,up));
    const link=(a,b,thickness=1)=>{
      const v=sub(b,a), d=length(v), start=add(a,mul(unit(v),.10));
      put('link',start,white,[thickness,d-.18,thickness],v);
    };
    link(shoulder,elbow,1.14); link(elbow,wrist,.91);
    for(const [point,scale] of [[shoulder,1.15],[elbow,1],[wrist,.8]]) {
      put('joint',point,dark,[scale,scale,scale],axle,.2,.5);
      for(const sign of [-1,1]) {
        const axis=mul(axle,sign), p=add(point,mul(axis,.13*scale));
        put('cap',p,white,[scale,scale,scale],axis,.1,.35);
        put('disk',add(p,mul(axis,.043*scale)),metal,[.045*scale,.8,.045*scale],axis,.65,.3);
      }
    }
    // Down-facing wrist, gripper carriage, parallel fingers, and black pads.
    put('wrist',add(wrist,[0,-.19,0]),white,[.78,.8,.78],[0,1,0],.1,.35);
    put('palm',add(wrist,[0,-.30,0]),white,[1,1,1],[0,1,0],.1,.4);
    put('rail',add(wrist,[0,-.412,0]),metal,[1,1,1],[0,1,0],.8,.28);
    put('badge',add(wrist,[0,-.29,.145]),dark);
    for(const sign of [-1,1]) {
      put('finger',add(wrist,[sign*this.pose.gap,-.54,0]),metal,[1,1,1],[0,1,0],.65,.35);
      put('pad',add(wrist,[sign*(this.pose.gap-.033),-.59,.004]),dark);
      put('screw',add(wrist,[sign*this.pose.gap,-.49,.07]),dark,[.55,.55,.55],[0,0,1]);
    }
    this.items=items;
  }

  drawItems(program,shadow=false) {
    const gl=this.gl;
    for(const item of this.items) {
      if(shadow&&item.floor)continue;
      const mesh=this.meshes[item.mesh];
      gl.bindBuffer(gl.ARRAY_BUFFER,mesh.buffer);
      gl.enableVertexAttribArray(program.position);gl.vertexAttribPointer(program.position,3,gl.FLOAT,false,24,0);
      gl.uniformMatrix4fv(program.uModel,false,item.model);
      if(!shadow) {
        gl.enableVertexAttribArray(program.normal);gl.vertexAttribPointer(program.normal,3,gl.FLOAT,false,24,12);
        gl.uniformMatrix3fv(program.uNormal,false,normalMatrix(item.model));
        gl.uniform3fv(program.uColor,item.color);gl.uniform1f(program.uMetal,item.metallic);gl.uniform1f(program.uRoughness,item.rough);gl.uniform1f(program.uFloor,item.floor);
      }
      gl.drawArrays(gl.TRIANGLES,0,mesh.count);
    }
  }
  drawCamera(eye,target,viewport,projection) {
    const gl=this.gl,p=this.program;
    gl.viewport(...viewport);gl.scissor(...viewport);
    gl.clearColor(.82,.84,.83,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.useProgram(p.p);gl.uniformMatrix4fv(p.uVP,false,multiply(projection,lookAt(eye,target)));
    gl.uniformMatrix4fv(p.uLightVP,false,this.lightVP);gl.uniform3fv(p.uEye,eye);
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.shadowTexture);gl.uniform1i(p.uShadow,0);gl.uniform1f(p.uShadowSize,this.shadowSize);
    this.drawItems(p);
  }
  draw(time) {
    const gl=this.gl;
    if(gl.isContextLost())return;
    if(this.motion) {
      const t=Math.min(1,(time-this.motion.start)/this.motion.duration), e=t*t*(3-2*t);
      for(const key of Object.keys(this.pose))this.pose[key]=this.motion.from[key]+(this.motion.to[key]-this.motion.from[key])*e;
      if(t===1)this.motion=null;
    }
    const ratio=Math.min(devicePixelRatio||1,2), width=Math.round(this.canvas.clientWidth*ratio),height=Math.round(this.canvas.clientHeight*ratio);
    if(!width||!height)return;
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    this.scene();
    gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER,this.shadowBuffer);gl.viewport(0,0,this.shadowSize,this.shadowSize);
    gl.clearColor(1,1,1,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.shadowProgram.p);gl.uniformMatrix4fv(this.shadowProgram.uVP,false,this.lightVP);this.drawItems(this.shadowProgram,true);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.enable(gl.SCISSOR_TEST);
    const target=[0,.88,0], radius=this.distance;
    const eye=add(target,[Math.sin(this.yaw)*Math.cos(this.elevation)*radius,Math.sin(this.elevation)*radius,Math.cos(this.yaw)*Math.cos(this.elevation)*radius]);
    this.drawCamera(eye,target,[0,0,width,height],perspective(.60,width/height));
    const detail=document.getElementById('detail-frame');
    if(this.detail) {
      const rect=this.canvas.getBoundingClientRect(), inset=detail.getBoundingClientRect();
      const viewport=[Math.round((inset.left-rect.left)*ratio),Math.round((rect.bottom-inset.bottom)*ratio),Math.round(inset.width*ratio),Math.round(inset.height*ratio)];
      const focus=[this.pose.pegX,this.pose.pegY+.36,.22];
      this.drawCamera(add(focus,[.60,.35,1.6]),focus,viewport,ortho(1.03*viewport[2]/viewport[3],1.03,.1,8));
    }
    gl.disable(gl.SCISSOR_TEST);
    if(this.motion)this.requestDraw();
  }
}
