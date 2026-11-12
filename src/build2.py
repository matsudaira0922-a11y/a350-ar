import numpy as np, trimesh, json
from geom_common import *
def Z(s): return L/2-np.asarray(s,float)
MESHES=[]  # dicts v,f,n,uv,mat

def vnormals(v,f):
    m=trimesh.Trimesh(v,f,process=False); return np.array(m.vertex_normals)
def signed_vol(v,f):
    a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]]; return np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6
def orient_closed(v,f):
    return f if signed_vol(v,f)>0 else f[:,::-1].copy()
def add(v,f,mat,uv=None,n=None,orient=True):
    v=np.asarray(v,float); f=np.asarray(f,np.int64)
    if orient: f=orient_closed(v,f)
    if n is None: n=vnormals(v,f)
    MESHES.append(dict(v=v,f=f,n=n,uv=uv,mat=mat))

def loft(rings,cap0=False,cap1=False,wrap=True):
    R=len(rings); M=len(rings[0]); V=np.vstack(rings); F=[]
    J=M if wrap else M-1
    for i in range(R-1):
        for j in range(J):
            a=i*M+j; b=i*M+(j+1)%M; c=(i+1)*M+(j+1)%M; d=(i+1)*M+j
            F+=[[a,b,c],[a,c,d]]
    V=list(V)
    for idx,flag in ((0,cap0),(R-1,cap1)):
        if not flag: continue
        ci=len(V); V.append(np.mean(rings[idx],0))
        for j in range(M):
            a=idx*M+j; b=idx*M+(j+1)%M; F.append([ci,b,a] if idx==0 else [ci,a,b])
    return np.array(V),np.array(F)

def revolve(profile,center,axis,nseg=48,e1=None):
    axis=np.asarray(axis,float); axis/=np.linalg.norm(axis)
    if e1 is None:
        e1=np.cross(axis,[0,1,0]) if abs(axis[1])<0.9 else np.cross(axis,[1,0,0])
    e1=e1/np.linalg.norm(e1); e2=np.cross(axis,e1)
    t=np.linspace(0,2*np.pi,nseg,endpoint=False)
    rings=[np.asarray(center)+a*axis+r*(np.outer(np.cos(t),e1)+np.outer(np.sin(t),e2)) for a,r in profile]
    return loft(rings)

# ------------------------------------------------------------------ fuselage (UV-mapped atlas)
M=96
th=np.linspace(0,2*np.pi,M,endpoint=False)
s_st=np.unique(np.round(np.concatenate([10.0*(np.linspace(0,1,46)**1.7),np.arange(10.5,40,0.9),
      40+(L-40)*np.linspace(0,1,40)**0.85,[L/3,2*L/3]]),4))
yc,hh,w=fus_profile(s_st)
rings=[np.c_[w[i]*np.sin(th), yc[i]-hh[i]*np.cos(th), np.full(M,Z(s_st[i]))] for i in range(len(s_st))]
Vr,Fr=loft(rings)
Fr=orient_closed(Vr,Fr)
Nr=vnormals(Vr,Fr).reshape(len(s_st),M,3)
Nr[0]=[0,0,1]
flip = signed_vol(Vr,loft(rings)[1])<0
# build UV mesh: seam duplicate (M+1), split strips
Vs,Ns,UVs,Fs=[],[],[],[]
edges=[0,L/3,2*L/3,L]
off=0
for k in range(3):
    idx=np.where((s_st>=edges[k]-1e-3)&(s_st<=edges[k+1]+1e-3))[0]
    for i in idx:
        ring=np.vstack([rings[i],rings[i][:1]]); nr=np.vstack([Nr[i],Nr[i][:1]])
        thv=np.r_[th,2*np.pi]
        u=(s_st[i]-edges[k])/(L/3); v=(k*(STRIP+1)+theta_to_v(thv)*STRIP)/TEX
        Vs.append(ring); Ns.append(nr); UVs.append(np.c_[np.full(M+1,u),v])
    R=len(idx)
    for a in range(R-1):
        for j in range(M):
            p=off+a*(M+1)+j; q=p+1; r_=p+(M+1)+1; s_=p+(M+1)
            Fs+=[[p,q,r_],[p,r_,s_]] if not flip else [[p,r_,q],[p,s_,r_]]
    off+=R*(M+1)
add(np.vstack(Vs),np.array(Fs),'Fuselage',uv=np.vstack(UVs),n=np.vstack(Ns),orient=False)
# APU exhaust cap
yce,hhe,we=fus_profile(L)
v,f=revolve([(0.005,0.0),(0.005,we*0.98),(0.15,we*0.75),(0.15,0.0)],[0,yce,Z(L)+0.0],[0,0,1],32)
add(v,f,'DarkMetal')

# ------------------------------------------------------------------ airfoils
def airfoil(n=18,tc=0.12,camber=0.02,p=0.4):
    x=(1-np.cos(np.linspace(0,np.pi,n)))/2
    yt=5*tc*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    yc_=np.where(x<p,camber/p**2*(2*p*x-x**2),camber/(1-p)**2*((1-2*p)+2*p*x-x**2))
    up=np.c_[x,yc_+yt][::-1]; lo=np.c_[x,yc_-yt][1:-1]
    return np.vstack([up,lo])   # TE->upper->LE->lower->(TE)
def section(le, chord, cdir, ndir, tc, camber=0.02, n=18):
    a=airfoil(n,tc,camber)
    return np.asarray(le)+np.outer(a[:,0]*chord,cdir)+np.outer(a[:,1]*chord,ndir)
def mirror_x(v,f): v=v.copy(); v[:,0]*=-1; return v,f[:,::-1].copy()

# ------------------------------------------------------------------ wing
DIH=np.radians(6.0); YW=ZC-1.75
WB=[0,3.0,10.5,30.4]; WLE=[19.5,21.6,26.85,40.8]; WCH=[17.4,15.5,8.8,2.4]; WTC=[0.15,0.14,0.11,0.09]; WINC=[3.0,3.0,1.5,-1.0]
def wp(b):
    return (np.interp(b,WB,WLE),np.interp(b,WB,WCH),np.interp(b,WB,WTC),YW+b*np.tan(DIH),np.radians(np.interp(b,WB,WINC)))
def wing_rings():
    rings=[]
    for b in np.unique(np.r_[np.linspace(0,30.4,28),3.0,10.5]):
        le,ch,tc,y,inc=wp(b)
        c=np.array([0,-np.sin(inc),-np.cos(inc)]); n=np.array([-np.sin(DIH)*np.cos(inc),np.cos(DIH)*np.cos(inc),-np.sin(inc)])
        rings.append(section([b,y,Z(le)],ch,c,n,tc))
    return rings
wr=wing_rings(); v,f=loft(wr)
add(v,f,'WingGrey')
# sharklet (red): bend radius so outer skin hits SPAN/2
Rb=(SPAN/2-30.4-0.03)/(1-np.sin(DIH))
P0=np.array([30.4,YW+30.4*np.tan(DIH)]); C0=P0+Rb*np.array([-np.sin(DIH),np.cos(DIH)])
sh=[]; arc=Rb*(np.pi/2-DIH); tot=arc+1.7
for psi in np.linspace(0,np.pi/2-DIH,12):
    phi=DIH+psi; pt=C0+Rb*np.array([np.sin(phi),-np.cos(phi)]); sh.append((pt,phi,Rb*psi/tot))
for e in np.linspace(0.3,1.7,5):
    pt=C0+Rb*np.array([1,0])+np.array([0,e]); sh.append((pt,np.pi/2,(arc+e)/tot))
rings=[]
for pt,phi,u in sh:
    ch=2.4*(1-u)+0.80*u; le=40.8+3.0*u**1.25; tc=0.09-0.01*u; inc=np.radians(-1+1*u)
    c=np.array([0,-np.sin(inc),-np.cos(inc)]); n=np.array([-np.sin(phi),np.cos(phi),0.0])
    rings.append(section([pt[0],pt[1],Z(le)],ch,c,n,tc,camber=0.01))
v,f=loft(rings,cap1=True); add(v,f,'Red')
# flap track fairings
for b in (7.6,13.2,18.6,23.6):
    le,ch,tc,y,inc=wp(b)
    s0=le+0.42*ch; s1=le+ch+0.30*ch+0.6; ylo=y-0.035*ch
    rs=[]
    for t in np.linspace(0,1,16):
        hh_=max(0.02,min(0.62,ch*0.07+0.15)*np.sin(np.pi*min(1,t*1.15))**0.6) if t<1 else 0.02
        ww=max(0.02,0.30*np.sin(np.pi*t)**0.5)
        a=np.linspace(0,2*np.pi,16,endpoint=False)
        rs.append(np.c_[b+ww*np.cos(a), ylo-0.05-hh_*0.5+hh_*0.5*np.sin(a)*1.6, np.full(16,Z(s0+(s1-s0)*t))])
    v,f=loft(rs,True,True); add(v,f,'WingGrey')
# mirror all wing parts so far (except fuselage & apu)
for m in list(MESHES[2:]):
    v,f=mirror_x(m['v'],m['f']); n=m['n'].copy(); n[:,0]*=-1
    MESHES.append(dict(v=v,f=f,n=n,uv=None,mat=m['mat']))

# ------------------------------------------------------------------ belly / wing-body fairing
rs=[]
for t in np.linspace(0,1,34):
    s=14.5+(45.5-14.5)*t; bump=np.sin(np.pi*t)**0.38
    a=np.linspace(0,2*np.pi,48,endpoint=False)
    ca,sa=np.cos(a),np.sin(a)
    xx=3.30*bump*np.sign(ca)*np.abs(ca)**0.8; yy=(ZC-1.25)+1.88*bump*np.sign(sa)*np.abs(sa)**0.8
    rs.append(np.c_[xx,yy,np.full(48,Z(s))])
v,f=loft(rs); add(v,f,'WingGrey')

# ------------------------------------------------------------------ tailplanes
HS_LE0,HS_C0,HS_LE1,HS_C1,HS_B=58.6,6.8,64.7,1.95,9.45; HDIH=np.radians(7); HY=ZC+0.45
rings=[]
for b in np.linspace(0,HS_B,14):
    u=b/HS_B; le=HS_LE0+(HS_LE1-HS_LE0)*u; ch=HS_C0+(HS_C1-HS_C0)*u
    c=np.array([0,0,-1.0]); n=np.array([-np.sin(HDIH),np.cos(HDIH),0])
    rings.append(section([b,HY+b*np.tan(HDIH),Z(le)],ch,c,n,0.10-0.02*u,camber=0.0))
v,f=loft(rings,cap1=True); add(v,f,'White')
v2,f2=mirror_x(v,f); add(v2,f2,'White')
# fin with planar livery UVs (split per side)
rings=[]
for y in np.linspace(FIN_Y0,H,16):
    le=float(fin_le(y)); ch=float(fin_chord(y)); tc=0.11-0.03*(y-FIN_Y0)/(H-FIN_Y0)
    rings.append(section([0,y,Z(le)],ch,np.array([0,0,-1.0]),np.array([1.0,0,0]),tc,camber=0.0,n=22))
v,f=loft(rings,cap1=True); f=orient_closed(v,f); n=vnormals(v,f)
fn=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
VV,NN,UU,FF=[],[],[],[]
for side in (1,-1):
    sel=f[(fn[:,0]*side)>=0] if side==1 else f[fn[:,0]<0]
    uniq,inv=np.unique(sel.ravel(),return_inverse=True)
    vv=v[uniq]; s=L/2-vv[:,2]; u=(s-FIN_S0)/FIN_SZ; vq=(FIN_Y1-vv[:,1])/FIN_SZ
    if side==-1: u=1-u
    base=sum(len(x) for x in VV)
    VV.append(vv); NN.append(n[uniq]); UU.append(np.c_[u,vq]); FF.append(inv.reshape(-1,3)+base)
add(np.vstack(VV),np.vstack(FF),'Fin',uv=np.vstack(UU),n=np.vstack(NN),orient=False)

# ------------------------------------------------------------------ engines (Trent XWB)
ENG_B=10.3; ENG_S0=19.8; ENG_Y=2.78
inlet=[(1.45,1.47),(1.0,1.50),(0.45,1.53),(0.12,1.57),(0.0,1.66)]
lip=[(0.0,1.66),(0.04,1.74),(0.15,1.81),(0.38,1.87)]
cowl=[(0.38,1.87),(1.0,1.93),(1.9,1.955),(3.0,1.93),(4.1,1.84),(5.0,1.70),(5.8,1.52),(6.45,1.36)]
nozzle=[(6.45,1.36),(6.35,1.22),(6.2,1.12)]
core=[(6.2,1.12),(6.9,1.08),(7.7,0.97),(8.4,0.80),(8.75,0.70)]
corenoz=[(8.75,0.70),(8.6,0.60)]
plug=[(8.6,0.60),(9.0,0.50),(9.5,0.30),(9.9,0.06),(9.92,0.0)]
def cowl_r(d): xs=[p[0] for p in cowl]; return np.interp(d,xs,[p[1] for p in cowl])
def engine(xc):
    c=np.array([xc,ENG_Y,Z(ENG_S0)]); ax=np.array([0,0,-1.0])
    for prof,mat in ((inlet,'DarkMetal'),(lip,'Metal'),(cowl,'White'),(nozzle,'DarkMetal'),(core,'DarkMetal'),(corenoz,'DarkMetal'),(plug,'DarkMetal')):
        v,f=revolve(prof,c,ax,64)
        # orient: normals away from axis for outer, toward axis for inner duct
        fn=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]); cen=v[f].mean(1)
        rad=cen-c; rad[:,2]=0
        sgn=np.sign(np.einsum('ij,ij->i',fn,rad).sum())
        want=-1 if prof is inlet else 1
        if sgn!=want: f=f[:,::-1].copy()
        add(v,f,mat,orient=False)
    # fan face: hub disc + spinner + blades
    v,f=revolve([(1.47,0.0),(1.47,1.47)],c,ax,48); 
    fn=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    if fn[:,2].sum()<0: f=f[:,::-1].copy()
    add(v,f,'Fan',orient=False)
    v,f=revolve([(0.85,0.0),(0.95,0.18),(1.15,0.36),(1.35,0.45),(1.46,0.46)],c,ax,32)
    fn=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]); cen=v[f].mean(1)-c
    if np.einsum('ij,ij->i',fn,cen).sum()<0: f=f[:,::-1].copy()
    add(v,f,'Metal',orient=False)
    bl_v,bl_f=[],[]
    for k in range(22):
        ang=2*np.pi*k/22; rs_=[]
        for r in np.linspace(0.44,1.45,6):
            tw=np.radians(62-35*(r-0.44)/1.0); ch=0.32+0.18*(r-0.44)
            radial=np.array([np.cos(ang),np.sin(ang),0]); tang=np.array([-np.sin(ang),np.cos(ang),0])
            cd=np.cos(tw)*np.array([0,0,-1.0])+np.sin(tw)*tang; nd=np.cross(radial,cd); nd/=np.linalg.norm(nd)
            ctr=c+np.array([0,0,-1.25])+r*radial*np.array([1,1,0])
            sq=np.array([[-0.5,-1],[0.5,-1],[0.5,1],[-0.5,1]])
            rs_.append(ctr+np.outer(sq[:,0]*ch,cd)+np.outer(sq[:,1]*0.012,nd))
        v,f=loft(rs_,True,True); bl_f.append(f+sum(len(x) for x in bl_v)); bl_v.append(v)
    v=np.vstack(bl_v); f=np.vstack(bl_f)
    # orient each blade outward
    add(v,f,'Fan',orient=False,n=None)
    # pylon
    rs=[]; yw=YW+ENG_B*np.tan(DIH)
    for s in np.linspace(ENG_S0+0.55,32.2,22):
        d=s-ENG_S0
        ybot=ENG_Y+(cowl_r(d)-0.12 if d<6.45 else 1.2+(yw-0.25-ENG_Y-1.2)*min(1,(d-6.45)/4))
        ytop=min(yw+0.15, ybot+0.25+(yw+0.15-ybot)*min(1,(d-0.55)/5.5))
        ytop=max(ytop,ybot+0.05)
        hw=0.36*min(1,(d-0.55)/1.2+0.25)*(1 if s<30.5 else max(0.15,(32.2-s)/1.7))
        a=np.linspace(0,2*np.pi,14,endpoint=False); ca,sa=np.cos(a),np.sin(a)
        rs.append(np.c_[xc+hw*np.sign(ca)*np.abs(ca)**0.4,(ybot+ytop)/2+(ytop-ybot)/2*np.sign(sa)*np.abs(sa)**0.4,np.full(14,Z(s))])
    v,f=loft(rs,True,True); add(v,f,'White')
for xc in (ENG_B,-ENG_B): engine(xc)

# ------------------------------------------------------------------ landing gear
def cyl(p0,p1,r,mat,n=14):
    p0,p1=np.asarray(p0,float),np.asarray(p1,float); ax=p1-p0; Lc=np.linalg.norm(ax)
    v,f=revolve([(0,0),(0,r),(Lc,r),(Lc,0)],p0,ax/Lc,n); add(v,f,mat)
def box(c,size,mat,rot=None):
    m=trimesh.creation.box(size)
    if rot is not None: m.apply_transform(rot)
    m.apply_translation(c); add(m.vertices,m.faces,mat)
def wheel(c,r,width,hub_side):
    c=np.asarray(c,float); ax=np.array([1.0,0,0])
    hw=width/2; prof=[(-hw*0.7,r*0.62),(-hw,r*0.70),(-hw,r*0.88),(-hw*0.85,r*0.98),(-hw*0.5,r),(hw*0.5,r),(hw*0.85,r*0.98),(hw,r*0.88),(hw,r*0.70),(hw*0.7,r*0.62)]
    v,f=revolve(prof,c,ax,28,e1=np.array([0,1.0,0])); add(v,f,'Rubber',orient=True)
    v,f=revolve([(-hw*0.72,0),(-hw*0.72,r*0.63),(-hw*0.5,r*0.6),(-hw*0.55,r*0.25),(-hw*0.62,0.0)],c,ax,20,e1=np.array([0,1.0,0]))
    v2,f2=revolve([(hw*0.62,0.0),(hw*0.55,r*0.25),(hw*0.5,r*0.6),(hw*0.72,r*0.63),(hw*0.72,0)],c,ax,20,e1=np.array([0,1.0,0]))
    vv=np.vstack([v,v2]); ff=np.vstack([f,f2+len(v)])
    add(vv,ff,'Metal',orient=True)
# nose gear
NS=4.9; nr=0.53; fy=fus_profile(NS); ybody=fy[0]-fy[1]
for sx in (-0.33,0.33): wheel([sx,nr,Z(NS)],nr,0.36,1)
cyl([0,nr,Z(NS)-0.0],[0,nr+0.25,Z(NS)+0.0],0.06,'Metal')
cyl([-0.36,nr,Z(NS)],[0.36,nr,Z(NS)],0.07,'Metal')
cyl([0,nr+0.05,Z(NS)+0.1],[0,1.35,Z(NS)+0.1],0.11,'Metal')
cyl([0,1.3,Z(NS)+0.1],[0,ybody+0.3,Z(NS)+0.1],0.16,'Gear')
cyl([0,1.6,Z(NS)+0.12],[0,ybody+0.1,Z(NS)-1.3],0.07,'Gear')  # drag brace
for sx in (-0.58,0.58): box([sx,ybody-0.22,Z(NS+0.6)],[0.04,0.55,2.0],'White')
# main gear (4-wheel bogie each side, track 10.6, wheelbase 28.67)
MS=NS+28.67; TR=5.3; mr=0.66
for sd in (1,-1):
    x=sd*TR
    for ds in (-0.95,0.95):
        for dx in (-0.52,0.52): wheel([x+dx,mr,Z(MS+ds)],mr,0.50,1)
        cyl([x-0.85,mr,Z(MS+ds)],[x+0.85,mr,Z(MS+ds)],0.09,'Metal')
    box([x,mr+0.05,Z(MS)],[0.26,0.24,2.4],'Gear')
    cyl([x,mr+0.1,Z(MS)],[x,1.9,Z(MS)],0.15,'Metal')
    cyl([x,1.8,Z(MS)],[x,YW+abs(x)*np.tan(DIH)-0.1,Z(MS)],0.22,'Gear')
    cyl([x,2.0,Z(MS)],[sd*2.2,ZC-2.0,Z(MS-0.3)],0.09,'Gear')   # side stay
    cyl([x,1.6,Z(MS)+0.2],[x,1.2,Z(MS)+0.35],0.05,'Metal')     # torque link
    box([x+sd*0.40,2.55,Z(MS)],[0.04,1.7,1.25],'White')        # leg door

# ------------------------------------------------------------------ cockpit windows (real geometry)
import nosewin as nw
POL=nw.build_polys()
def add_patch(V,F_,N,mat): add(V,F_,mat,n=N,orient=False)
for side in ('L','R'):
    # black mask skin
    V,F_,N=nw.fill_patch(POL['mask_'+side],0.020,0.05); add_patch(V,F_,N,'Mask')
    for nm in ('ws','sw1','sw2'):
        P=POL[nm+'_'+side]
        V,F_,N=nw.ring_strip(nw.dense(P,0.04),0.024,0.034); add_patch(V,F_,N,'Metal')      # silver rim
        Pin=nw.offset_poly(nw.dense(P,0.04),-0.024)
        V,F_,N=nw.fill_patch(Pin,0.040,0.04); add_patch(V,F_,N,'Glass')
# centre post between windscreens + wiper arms
V,F_,N=nw.fill_patch(np.array([[-0.045,5.88],[0.045,5.88],[0.045,6.70],[-0.045,6.70]]),0.030,0.03); add_patch(V,F_,N,'Mask')
def surf_cyl(x0,y0,x1,y1,r,off,mat):
    a=nw.surf_point(x0,y0,off); b=nw.surf_point(x1,y1,off)
    if a and b: cyl(a[0],b[0],r,mat,8)
for sx in (-1,1):
    surf_cyl(sx*0.10,5.80,sx*0.30,5.73,0.008,0.07,'Rubber')     # wiper arm
    surf_cyl(sx*0.10,5.80,sx*0.14,5.62,0.007,0.05,'Rubber')     # wiper pivot
def probe(x,y,mat='DarkMetal',ring_r=0.17):
    r=nw.surf_point(x,y,0.0)
    if not r: return
    p,n=r
    cyl(p,p+n*0.07,0.012,mat,8)
    cyl(p+n*0.07,p+n*0.07+np.array([0,0,-0.1]),0.006,mat,6)
    # red dashed ring
    t=np.linspace(0,2*np.pi,48,endpoint=False); ring=np.c_[x+ring_r*np.cos(t),y+ring_r*np.sin(t)]
    V,F_,N=nw.ring_strip(ring,0.012,0.012)
    keep=np.array([(i//2)%2==0 for i in range(len(F_)//2) for _ in (0,1)])
    add_patch(V,F_[keep],N,'Red')
for sx in (-1,1):
    probe(sx*1.52,4.64)
    # static port panel: rounded rect with red dashed outline + grey disc
    hw,hh=0.15,0.24; rr=0.04
    pts=[]
    for cx_,cy_,a0 in ((hw-rr,hh-rr,0),(-hw+rr,hh-rr,90),(-hw+rr,-hh+rr,180),(hw-rr,-hh+rr,270)):
        for a in np.radians(np.linspace(a0,a0+90,6)): pts.append((cx_+rr*np.cos(a),cy_+rr*np.sin(a)))
    pts=np.array(pts); pts=pts+np.array([sx*1.59,5.33])
    V,F_,N=nw.ring_strip(nw.dense(pts,0.03),0.010,0.010)
    keep=np.array([(i//5)%2==0 for i in range(len(F_)//2) for _ in (0,1)]); add_patch(V,F_[keep],N,'Red')
    t=np.linspace(0,2*np.pi,24,endpoint=False); disc=np.c_[sx*1.59+0.075*np.cos(t),5.33+0.075*np.sin(t)]
    V,F_,N=nw.fill_patch(disc,0.014,0.03); add_patch(V,F_,N,'Metal')
# small outlined panel on top of nose ahead of the windows
t=np.linspace(0,2*np.pi,60,endpoint=False)
ov=np.c_[0.34*np.cos(t),0.085*np.sin(t)]; rot=np.radians(14); R2=np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
ov=ov@R2.T+np.array([-0.62,7.50])
V,F_,N=nw.ring_strip(ov,0.014,0.010); add_patch(V,F_,N,'WingGrey')

# ------------------------------------------------------------------ assemble per material
MATS={ # name: (baseColor sRGB, metallic, roughness, texture)
 'Fuselage':((255,255,255),0.0,0.42,'tex_fuselage.jpg'),
 'Fin':((255,255,255),0.0,0.42,'tex_fin.jpg'),
 'White':((240,241,243),0.0,0.36,None),
 'WingGrey':((178,183,190),0.15,0.48,None),
 'Red':((204,0,17),0.0,0.38,None),
 'Metal':((214,217,222),1.0,0.24,None),
 'DarkMetal':((70,72,78),0.85,0.45,None),
 'Fan':((52,54,60),0.9,0.38,None),
 'Gear':((150,154,160),0.6,0.42,None),
 'Rubber':((22,22,24),0.0,0.88,None),
 'Mask':((20,22,27),0.0,0.30,None),
 'Glass':((14,24,42),0.35,0.07,None)}
REMAP={'Fan':'DarkMetal','Gear':'WingGrey'}
for m in MESHES: m['mat']=REMAP.get(m['mat'],m['mat'])
groups={}
for m in MESHES: groups.setdefault(m['mat'],[]).append(m)
final=[]
for name,ms in groups.items():
    vs,ns,uvs,fs=[],[],[],[]; off=0
    for m in ms:
        vs.append(m['v']); ns.append(m['n']); fs.append(m['f']+off); off+=len(m['v'])
        if m['uv'] is not None: uvs.append(m['uv'])
    final.append(dict(name=name,v=np.vstack(vs),n=np.vstack(ns),f=np.vstack(fs),uv=np.vstack(uvs) if uvs else None))
allv=np.vstack([g['v'] for g in final]); mn,mx=allv.min(0),allv.max(0)
print('raw bounds',mn.round(3),mx.round(3),'size',(mx-mn).round(4))
target=np.array([SPAN,H,L]); sc=target/(mx-mn); print('scale corr',sc)
ctr=np.array([(mn[0]+mx[0])/2,mn[1],(mn[2]+mx[2])/2])
for g in final:
    g['v']=(g['v']-ctr)*sc
    n=g['n']/sc; g['n']=n/np.linalg.norm(n,axis=1,keepdims=True).clip(1e-9)
allv=np.vstack([g['v'] for g in final]); print('final bounds',allv.min(0).round(4),allv.max(0).round(4))
print('tris',sum(len(g['f']) for g in final),'verts',len(allv))
np.save('final.npy',np.array(final,dtype=object),allow_pickle=True)
json.dump({k:[list(v[0]),v[1],v[2],v[3]] for k,v in MATS.items()},open('mats.json','w'))
