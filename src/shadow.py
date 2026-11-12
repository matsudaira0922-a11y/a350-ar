import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt
PPM=8
def build(final, png='shadow_decal.png'):
    allv=np.vstack([g['v'] for g in final]); a=float(max(abs(allv[:,0].min()),abs(allv[:,0].max()))); b=float(max(abs(allv[:,2].min()),abs(allv[:,2].max())))
    W,Hh=int(2*a*PPM)+1,int(2*b*PPM)+1
    bands=[(-1,1.2,0.25,0.80),(1.2,3.6,0.9,0.45),(3.6,7.0,1.6,0.40),(7.0,99,2.6,0.30)]
    masks=[Image.new('L',(W,Hh),0) for _ in bands]; dr=[ImageDraw.Draw(m) for m in masks]
    for g in final:
        v=g['v']
        if g['name']=='Rubber' and len(v)>0: pass
        for f in g['f']:
            p=v[f]
            if np.ptp(p[:,1])>6: yy=p[:,1].mean()
            else: yy=p[:,1].mean()
            if (np.abs(p[:,0])>=a-0.01).all() and (np.abs(p[:,2])>=b-0.01).all(): continue   # skip 1 mm anchors
            for i,(lo,hi,_,_) in enumerate(bands):
                if lo<=yy<hi:
                    dr[i].polygon([((q[0]+a)*PPM,(q[2]+b)*PPM) for q in p],fill=255); break
    alpha=np.zeros((Hh,W))
    keep=np.ones((Hh,W))
    for m,(lo,hi,sg,w) in zip(masks,bands):
        al=gaussian_filter(np.asarray(m).astype(float)/255,sg*PPM)*w
        keep*=(1-np.clip(al,0,1))
    alpha=1-keep; alpha=np.clip(alpha*1.0,0,0.78)
    img=np.zeros((Hh,W,4),np.uint8); img[...,3]=(alpha*255).astype(np.uint8)
    Image.fromarray(img,'RGBA').save(png,optimize=True)
    return a,b,png
def add_to_stage(st,root,a,b,png,texnode_fn=None):
    y=0.004
    mesh=UsdGeom.Mesh.Define(st,f'{root}/ContactShadow')
    pts=np.array([[-a,y,-b],[-a,y,b],[a,y,b],[a,y,-b]],np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts)); mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3,3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0,1,2,0,2,3]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.array([[0,1,0]]*4,np.float32))); mesh.SetNormalsInterpolation('vertex'); mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(-a,y,-b),Gf.Vec3f(a,y,b)])
    uv=np.array([[0,1],[0,0],[1,0],[1,1]],np.float32)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv))
    mp=f'{root}/Looks/ContactShadow'; mat=UsdShade.Material.Define(st,mp); sh=UsdShade.Shader.Define(st,mp+'/PBR'); sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0,0,0)); sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.0)
    rd=UsdShade.Shader.Define(st,mp+'/Reader'); rd.CreateIdAttr('UsdPrimvarReader_float2'); rd.CreateInput('varname',Sdf.ValueTypeNames.String).Set('st')
    tx=UsdShade.Shader.Define(st,mp+'/Tex'); tx.CreateIdAttr('UsdUVTexture'); tx.CreateInput('file',Sdf.ValueTypeNames.Asset).Set('./textures/'+png)
    tx.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
    tx.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('clamp'); tx.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('clamp')
    tx.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(),'result'); tx.CreateOutput('a',Sdf.ValueTypeNames.Float)
    sh.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(tx.ConnectableAPI(),'a')
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface'); UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
