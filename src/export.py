import numpy as np, json, struct, os, shutil
from geom_common import *
final=np.load('final.npy',allow_pickle=True); MATS=json.load(open('mats.json'))
def lin(c): c=np.asarray(c)/255.0; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
# ---------------- GLB
bin_=bytearray(); views=[]; accs=[]
def addview(data,target=None):
    while len(bin_)%4: bin_.append(0)
    off=len(bin_); bin_.extend(data); v={'buffer':0,'byteOffset':off,'byteLength':len(data)}
    if target: v['target']=target
    views.append(v); return len(views)-1
def addacc(arr,ctype,typ,target,minmax=False):
    bv=addview(arr.tobytes(),target)
    a={'bufferView':bv,'componentType':ctype,'count':len(arr),'type':typ}
    if minmax: a['min']=arr.min(0).tolist(); a['max']=arr.max(0).tolist()
    accs.append(a); return len(accs)-1
images=[];textures=[];materials=[];meshes=[];nodes=[]
texidx={}
for g in final:
    name=g['name']; col,met,rough,tex=MATS[name]
    pbr={'baseColorFactor':list(lin(col))+[1.0],'metallicFactor':met,'roughnessFactor':rough}
    if tex:
        if tex not in texidx:
            bv=addview(open(tex,'rb').read()); images.append({'bufferView':bv,'mimeType':'image/jpeg','name':tex})
            textures.append({'source':len(images)-1,'sampler':0}); texidx[tex]=len(textures)-1
        pbr['baseColorTexture']={'index':texidx[tex]}
    materials.append({'name':name,'pbrMetallicRoughness':pbr,'doubleSided':False})
    attrs={'POSITION':addacc(g['v'].astype(np.float32),5126,'VEC3',34962,True),
           'NORMAL':addacc(g['n'].astype(np.float32),5126,'VEC3',34962)}
    if g['uv'] is not None: attrs['TEXCOORD_0']=addacc(g['uv'].astype(np.float32),5126,'VEC2',34962)
    ind=addacc(g['f'].astype(np.uint32).ravel(),5125,'SCALAR',34963)
    meshes.append({'name':name,'primitives':[{'attributes':attrs,'indices':ind,'material':len(materials)-1}]})
    nodes.append({'name':name,'mesh':len(meshes)-1})
while len(bin_)%4: bin_.append(0)
gl={'asset':{'version':'2.0','generator':'a350-ar procedural (fan model)','copyright':'Unofficial fan model, not affiliated with JAL or Airbus'},
    'scene':0,'scenes':[{'nodes':[len(nodes)]}],
    'nodes':nodes+[{'name':'JAL_A350_900','children':list(range(len(nodes)))}],
    'meshes':meshes,'materials':materials,'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(bin_)}],
    'images':images,'textures':textures,'samplers':[{'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071}]}
js=json.dumps(gl,separators=(',',':')).encode()
while len(js)%4: js+=b' '
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(bin_))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(bin_),0x004E4942)+bytes(bin_)
open('jal-a350-900.glb','wb').write(out); print('glb',len(out))
# ---------------- USDZ
from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf,Vt,UsdUtils
os.makedirs('usd_tmp',exist_ok=True)
os.makedirs('usd_tmp/textures',exist_ok=True); [shutil.copy(t,'usd_tmp/textures/'+t) for t in ('tex_fuselage.jpg','tex_fin.jpg')]
path='usd_tmp/jal-a350-900.usdc'
if os.path.exists(path): os.remove(path)
st=Usd.Stage.CreateNew(path); UsdGeom.SetStageUpAxis(st,UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st,1.0)
root=UsdGeom.Xform.Define(st,'/JAL_A350_900'); st.SetDefaultPrim(root.GetPrim())
for g in final:
    name=g['name']; col,met,rough,tex=MATS[name]
    mesh=UsdGeom.Mesh.Define(st,f'/JAL_A350_900/{name}')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(g['v'].astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(g['f'])))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(g['f'].ravel().tolist()))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(g['n'].astype(np.float32))); mesh.SetNormalsInterpolation('vertex')
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(*g['v'].min(0).tolist()),Gf.Vec3f(*g['v'].max(0).tolist())])
    mat=UsdShade.Material.Define(st,f'/JAL_A350_900/Looks/{name}')
    sh=UsdShade.Shader.Define(st,f'/JAL_A350_900/Looks/{name}/PBR'); sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(met))
    sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(rough))
    if tex:
        uv=g['uv'].copy(); uv[:,1]=1-uv[:,1]
        pv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        rd=UsdShade.Shader.Define(st,f'/JAL_A350_900/Looks/{name}/stReader'); rd.CreateIdAttr('UsdPrimvarReader_float2')
        rd.CreateInput('varname',Sdf.ValueTypeNames.String).Set('st')
        tx=UsdShade.Shader.Define(st,f'/JAL_A350_900/Looks/{name}/diffuseTex'); tx.CreateIdAttr('UsdUVTexture')
        tx.CreateInput('file',Sdf.ValueTypeNames.Asset).Set('./textures/'+tex)
        tx.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
        tx.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('clamp'); tx.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('clamp')
        tx.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(),'result')
        tx.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(),'rgb')
    else:
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*lin(col).tolist()))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
st.GetRootLayer().Save()
ok=UsdUtils.CreateNewARKitUsdzPackage(path,'jal-a350-900.usdz'); print('usdz pkg',ok,os.path.getsize('jal-a350-900.usdz'))
from pxr import UsdValidation
s=Usd.Stage.Open('jal-a350-900.usdz')
reg=UsdValidation.ValidationRegistry(); names=[m.name for m in reg.GetAllValidatorMetadata()]
errs=UsdValidation.ValidationContext([reg.GetOrLoadValidatorByName(n) for n in names]).Validate(s)
print('usd validators',len(names),'issues',len(errs)); [print(' -',e.GetMessage()[:160]) for e in errs]
print('up',UsdGeom.GetStageUpAxis(s),'mpu',UsdGeom.GetStageMetersPerUnit(s))
bb=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default']).ComputeWorldBound(s.GetDefaultPrim()).ComputeAlignedRange()
print('usd bbox',bb.GetMin(),bb.GetMax(),bb.GetSize())
