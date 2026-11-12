import numpy as np, json, struct, os, shutil, trimesh
from geom_common import *
from night_tex import light_I, lin
final=list(np.load('final.npy',allow_pickle=True)); MATS=json.load(open('mats.json'))
MATS['Rubber'][2]=0.55
MATS['Green']=[[40,255,100],0.0,0.4,None]
G={g['name']:g for g in final}
def uv_ramp(I): I=np.clip(np.asarray(I,float),0,1); return np.c_[(I*255+0.5)/256,np.full(len(I),0.5)]
# ---- merge Mask into Rubber (material budget)
m=G.pop('Mask'); r=G['Rubber']; off=len(r['v'])
r['v']=np.vstack([r['v'],m['v']]); r['n']=np.vstack([r['n'],m['n']]); r['f']=np.vstack([r['f'],m['f']+off])
final=[g for g in final if g['name']!='Mask']
# ---- per-vertex baked intensity -> ramp UV (meshes without atlas UVs)
for g in final:
    if g['uv'] is None:
        n=g['n']; g['uv']=uv_ramp(light_I(g['v'][:,1],n[:,1],np.sqrt(np.clip(1-n[:,1]**2,0,1))))
# ---- lights (static emissive bits)
def sphere(c,rad,sc=(1,1,1)):
    s=trimesh.creation.icosphere(subdivisions=1,radius=rad); v=s.vertices*np.array(sc)+c; return v,s.faces,s.vertex_normals
def addbit(name,c,rad,sc=(1,1,1)):
    v,f,n=sphere(np.asarray(c),rad,sc); g=G[name] if name in G else None
    if g is None:
        g=dict(name=name,v=np.zeros((0,3)),n=np.zeros((0,3)),f=np.zeros((0,3),int),uv=np.zeros((0,2))); G[name]=g; final.append(g)
    o=len(g['v']); g['v']=np.vstack([g['v'],v]); g['n']=np.vstack([g['n'],n]); g['f']=np.vstack([g['f'],f+o]); g['uv']=np.vstack([g['uv'],uv_ramp(np.ones(len(v)))])
red=G['Red']['v']
ip=np.argmax(red[:,0]); ist=np.argmin(red[:,0])
addbit('Red',red[ip]+[-0.12,-0.05,0],0.16)          # port (left, +X) = red
addbit('Green',red[ist]+[0.12,-0.05,0],0.16)        # starboard = green
yc,hh,w=fus_profile(L); addbit('White',[0,float(yc),-L/2+0.18],0.15)             # tail light
yc,hh,w=fus_profile(33.0); addbit('Red',[0,float(yc+hh)+0.02,L/2-33.0],0.17,(1,0.6,1))   # top beacon
yc,hh,w=fus_profile(47.5); addbit('Red',[0,float(yc-hh)-0.02,L/2-47.5],0.17,(1,0.6,1))   # belly beacon
for sd in (1,-1): addbit('White',red[ip if sd==1 else ist]+[sd*-0.12,-0.05,0.55],0.09)   # wingtip white
# ---- placement + symmetric bbox (identical to day variant)
NOSE_DIST=10.0; YAW=np.radians(-45.0); c,sn=np.cos(YAW),np.sin(YAW); Ry=np.array([[c,0,sn],[0,1,0],[-sn,0,c]])
nose=np.array([0,0,L/2])
for g in final: g['v']=(g['v']-nose)@Ry.T+np.array([0,0,-NOSE_DIST]); g['n']=g['n']@Ry.T
av=np.vstack([g['v'] for g in final]); mn,mx=av.min(0),av.max(0); a_=max(abs(mn[0]),abs(mx[0])); b_=max(abs(mn[2]),abs(mx[2]))
g=G['Rubber']
for sx in (-1,1):
    for sz in (-1,1):
        n0=len(g['v']); g['v']=np.vstack([g['v'],[[sx*a_,0,sz*b_],[sx*a_+0.001,0,sz*b_],[sx*a_,0.001,sz*b_]]])
        g['n']=np.vstack([g['n'],[[0,0,1.0]]*3]); g['f']=np.vstack([g['f'],[[n0,n0+1,n0+2]]]); g['uv']=np.vstack([g['uv'],uv_ramp(np.zeros(3))])
allv=np.vstack([g['v'] for g in final]); print('bounds',allv.min(0).round(3),allv.max(0).round(3))
ATLAS={'Fuselage':('tex_fuselage.jpg','tex_fuselage_night.jpg'),'Fin':('tex_fin.jpg','tex_fin_night.jpg')}
GLASS_E=[0.09,0.055,0.025]
def emissive_factor(name):
    if name in ATLAS: return [1,1,1]
    if name=='Glass': return GLASS_E
    return [float(max(x,0.05)) for x in lin(MATS[name][0])]
# ---------------- GLB
bin_=bytearray(); views=[]; accs=[]
def addview(data,target=None):
    while len(bin_)%4: bin_.append(0)
    off=len(bin_); bin_.extend(data); v={'buffer':0,'byteOffset':off,'byteLength':len(data)}
    if target: v['target']=target
    views.append(v); return len(views)-1
def addacc(arr,ctype,typ,target,minmax=False):
    bv=addview(arr.tobytes(),target); a={'bufferView':bv,'componentType':ctype,'count':len(arr),'type':typ}
    if minmax: a['min']=arr.min(0).tolist(); a['max']=arr.max(0).tolist()
    accs.append(a); return len(accs)-1
images=[];textures=[];materials=[];meshes=[];nodes=[];texidx={}
def tex(path,mime):
    if path not in texidx:
        bv=addview(open(path,'rb').read()); images.append({'bufferView':bv,'mimeType':mime,'name':path})
        textures.append({'source':len(images)-1,'sampler':0}); texidx[path]=len(textures)-1
    return {'index':texidx[path]}
for g in final:
    name=g['name']; col,met,rough,_=MATS[name]
    pbr={'baseColorFactor':list(lin(col))+[1.0],'metallicFactor':met,'roughnessFactor':rough}
    mat={'name':name,'pbrMetallicRoughness':pbr,'doubleSided':False,'emissiveFactor':emissive_factor(name)}
    if name in ATLAS:
        pbr['baseColorTexture']=tex(ATLAS[name][0],'image/jpeg'); mat['emissiveTexture']=tex(ATLAS[name][1],'image/jpeg')
    elif name!='Glass': mat['emissiveTexture']=tex('ramp_night.png','image/png')
    materials.append(mat)
    attrs={'POSITION':addacc(g['v'].astype(np.float32),5126,'VEC3',34962,True),'NORMAL':addacc(g['n'].astype(np.float32),5126,'VEC3',34962)}
    if name!='Glass': attrs['TEXCOORD_0']=addacc(g['uv'].astype(np.float32),5126,'VEC2',34962)
    ind=addacc(g['f'].astype(np.uint32).ravel(),5125,'SCALAR',34963)
    meshes.append({'name':name,'primitives':[{'attributes':attrs,'indices':ind,'material':len(materials)-1}]}); nodes.append({'name':name,'mesh':len(meshes)-1})
while len(bin_)%4: bin_.append(0)
gl={'asset':{'version':'2.0','generator':'a350-ar procedural night variant','copyright':'Unofficial fan model, not affiliated with JAL or Airbus'},
    'scene':0,'scenes':[{'nodes':[len(nodes)]}],'nodes':nodes+[{'name':'JAL_A350_900_NIGHT','children':list(range(len(nodes)))}],
    'meshes':meshes,'materials':materials,'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(bin_)}],
    'images':images,'textures':textures,'samplers':[{'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071}]}
js=json.dumps(gl,separators=(',',':')).encode()
while len(js)%4: js+=b' '
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(bin_))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(bin_),0x004E4942)+bytes(bin_)
open('jal-a350-900-night.glb','wb').write(out); print('glb',len(out),'tris',sum(len(g['f']) for g in final),'materials',len(materials))
# ---------------- USDZ
from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf,Vt,UsdUtils
shutil.rmtree('usd_night',ignore_errors=True); os.makedirs('usd_night/textures')
for t in ('tex_fuselage.jpg','tex_fin.jpg','tex_fuselage_night.jpg','tex_fin_night.jpg','ramp_night.png'): shutil.copy(t,'usd_night/textures/'+t)
path='usd_night/jal-a350-900-night.usdc'
st=Usd.Stage.CreateNew(path); UsdGeom.SetStageUpAxis(st,UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st,1.0)
root=UsdGeom.Xform.Define(st,'/JAL_A350_900_NIGHT'); st.SetDefaultPrim(root.GetPrim()); P='/JAL_A350_900_NIGHT'
def texnode(mp,nm,file,cs,scale=None):
    rd=UsdShade.Shader.Define(st,f'{mp}/{nm}Reader'); rd.CreateIdAttr('UsdPrimvarReader_float2'); rd.CreateInput('varname',Sdf.ValueTypeNames.String).Set('st')
    tx=UsdShade.Shader.Define(st,f'{mp}/{nm}'); tx.CreateIdAttr('UsdUVTexture')
    tx.CreateInput('file',Sdf.ValueTypeNames.Asset).Set('./textures/'+file); tx.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(cs)
    tx.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('clamp'); tx.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('clamp')
    if scale is not None: tx.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
    tx.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(),'result'); tx.CreateOutput('rgb',Sdf.ValueTypeNames.Float3); return tx
for g in final:
    name=g['name']; col,met,rough,_=MATS[name]
    mesh=UsdGeom.Mesh.Define(st,f'{P}/{name}')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(g['v'].astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(g['f']))); mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(g['f'].ravel().tolist()))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(g['n'].astype(np.float32))); mesh.SetNormalsInterpolation('vertex'); mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(*g['v'].min(0).tolist()),Gf.Vec3f(*g['v'].max(0).tolist())])
    mp=f'{P}/Looks/{name}'; mat=UsdShade.Material.Define(st,mp); sh=UsdShade.Shader.Define(st,mp+'/PBR'); sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(met)); sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(rough))
    if name!='Glass':
        uv=g['uv'].copy(); uv[:,1]=1-uv[:,1]
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    if name in ATLAS:
        d=texnode(mp,'diffuseTex',ATLAS[name][0],'sRGB'); sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(d.ConnectableAPI(),'rgb')
        e=texnode(mp,'emissiveTex',ATLAS[name][1],'sRGB'); sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(e.ConnectableAPI(),'rgb')
    else:
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*lin(col).tolist()))
        if name=='Glass': sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*GLASS_E))
        else:
            ef=emissive_factor(name); e=texnode(mp,'emissiveTex','ramp_night.png','sRGB',scale=(ef[0],ef[1],ef[2],1.0))
            sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(e.ConnectableAPI(),'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface'); UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
st.GetRootLayer().Save()
print('usdz pkg',UsdUtils.CreateNewARKitUsdzPackage(path,'jal-a350-900-night.usdz'),os.path.getsize('jal-a350-900-night.usdz'))
from pxr import UsdValidation
s=Usd.Stage.Open('jal-a350-900-night.usdz'); reg=UsdValidation.ValidationRegistry(); names=[m.name for m in reg.GetAllValidatorMetadata()]
errs=UsdValidation.ValidationContext([reg.GetOrLoadValidatorByName(n) for n in names]).Validate(s)
print('usd validators',len(names),'issues',len(errs)); [print(' -',e.GetMessage()[:160]) for e in errs]
print('up',UsdGeom.GetStageUpAxis(s),'mpu',UsdGeom.GetStageMetersPerUnit(s))
bb=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default']).ComputeWorldBound(s.GetDefaultPrim()).ComputeAlignedRange(); print('usd bbox',bb.GetMin(),bb.GetMax())
