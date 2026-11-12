import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay
from geom_common import *
PXM=177.0
def c2m(cx,cy):
    nx=(300+1.367*cx)/1.92; ny=(420+1.367*cy)/1.92
    return (nx-632.0)/PXM, (RH+ZC)-(ny-20)/PXM
def poly_m(pts): return np.array([c2m(*p) for p in pts])
def chaikin(P,it=3):
    P=np.asarray(P)
    for _ in range(it):
        Q=[]
        for i in range(len(P)):
            a,b=P[i],P[(i+1)%len(P)]; Q+=[0.75*a+0.25*b,0.25*a+0.75*b]
        P=np.array(Q)
    return P
def mirror(P): Q=P.copy(); Q[:,0]*=-1; return Q[::-1]
def offset_poly(P,d):
    # simple inset by moving points toward centroid-ish normal
    P=np.asarray(P); n=len(P); out=[]
    ar=0.5*np.sum(P[:,0]*np.roll(P[:,1],-1)-np.roll(P[:,0],-1)*P[:,1]); sgn=1 if ar>0 else -1
    for i in range(n):
        t=P[(i+1)%n]-P[(i-1)%n]; t/=np.linalg.norm(t)+1e-12
        nrm=np.array([t[1],-t[0]])*sgn   # outward for ccw
        out.append(P[i]+nrm*d)
    return np.array(out)
# ---------- digitised outlines (crop coords of Commons nose close-up), left (viewer-left) half
MASK_L=[(668,66),(560,52),(450,48),(380,52),(320,72),(262,128),(212,205),(185,262),(185,290),(205,303),(262,298),(330,278),(400,254),(480,243),(560,248),(630,256),(668,266)]
WS_L=[(470,72),(560,76),(645,96),(648,150),(642,214),(610,234),(520,222),(410,205),(402,194),(430,130)]
SW1_L=[(385,68),(420,70),(428,84),(400,150),(330,232),(285,268),(258,272),(246,262),(262,230),(330,150)]
SW2_L=[(325,84),(332,86),(332,98),(290,160),(235,235),(208,275),(194,276),(191,262),(205,222),(255,150)]
def build_polys():
    out={}
    for name,P in (('mask',MASK_L),('ws',WS_L),('sw1',SW1_L),('sw2',SW2_L)):
        Pm=chaikin(poly_m(P),3 if name!='mask' else 2)
        out[name+'_L']=Pm; out[name+'_R']=mirror(Pm)
    return out
# ---------- surface lifting
def F(x,y,s):
    yc,hh,w=fus_profile(s); return (x/w)**2+((y-yc)/hh)**2
def lift(x,y):
    ss=np.linspace(0.02,14,1400); 
    f=F(x,y,ss)
    idx=np.where(f<=1.0)[0]
    if len(idx)==0: return None
    i=idx[0]
    if i==0: return ss[0]
    a,b=ss[i-1],ss[i]
    for _ in range(40):
        m=(a+b)/2
        if F(x,y,m)>1: a=m
        else: b=m
    return (a+b)/2
def surf_point(x,y,off):
    s=lift(x,y)
    if s is None: return None
    h=1e-3
    gx=(F(x+h,y,s)-F(x-h,y,s))/(2*h); gy=(F(x,y+h,s)-F(x,y-h,s))/(2*h)
    gz=-(F(x,y,s+h)-F(x,y,s-h))/(2*h)   # z=L/2-s
    n=np.array([gx,gy,gz]); n/=np.linalg.norm(n)
    P=np.array([x,y,L/2-s])+n*off
    return P,n
def fill_patch(poly,off,step=0.045):
    poly=np.asarray(poly); path=Path(poly)
    # boundary points densified
    pts=[]
    for i in range(len(poly)):
        a,b=poly[i],poly[(i+1)%len(poly)]; k=max(1,int(np.linalg.norm(b-a)/step))
        for t in np.arange(k)/k: pts.append(a+(b-a)*t)
    pts=np.array(pts)
    mn,mx=poly.min(0),poly.max(0)
    gx,gy=np.meshgrid(np.arange(mn[0],mx[0],step),np.arange(mn[1],mx[1],step))
    g=np.c_[gx.ravel(),gy.ravel()]; g=g[path.contains_points(g,radius=-step*0.6)]
    P2=np.vstack([pts,g]); tri=Delaunay(P2)
    cen=P2[tri.simplices].mean(1); keep=path.contains_points(cen)
    F_=tri.simplices[keep]
    V=[];N=[];ok=[]
    for p in P2:
        r=surf_point(p[0],p[1],off)
        if r is None: V.append([0,0,0]); N.append([0,0,1]); ok.append(False)
        else: V.append(r[0]); N.append(r[1]); ok.append(True)
    V=np.array(V);N=np.array(N);ok=np.array(ok)
    F_=F_[ok[F_].all(1)]
    # orient outward
    fn=np.cross(V[F_[:,1]]-V[F_[:,0]],V[F_[:,2]]-V[F_[:,0]])
    flip=(np.einsum('ij,ij->i',fn,N[F_].mean(1))<0)
    F_[flip]=F_[flip][:,::-1]
    return V,F_,N
def ring_strip(poly,width,off,step=0.03):
    """closed outline strip of given width (inward) drawn on surface"""
    poly=np.asarray(poly); inner=offset_poly(poly,-width)
    # densify both consistently
    n=len(poly)
    rings=[]
    for P in (poly,inner): rings.append(P)
    V=[];N=[];F_=[]
    for P in rings:
        for p in P:
            r=surf_point(p[0],p[1],off); 
            V.append(r[0] if r else [0,0,0]); N.append(r[1] if r else [0,0,1])
    for i in range(n):
        a=i; b=(i+1)%n; c=n+(i+1)%n; d=n+i
        F_+=[[a,b,c],[a,c,d]]
    V=np.array(V);N=np.array(N);F_=np.array(F_)
    fn=np.cross(V[F_[:,1]]-V[F_[:,0]],V[F_[:,2]]-V[F_[:,0]])
    flip=(np.einsum('ij,ij->i',fn,N[F_].mean(1))<0); F_[flip]=F_[flip][:,::-1]
    return V,F_,N
def dense(P,step=0.04):
    P=np.asarray(P); out=[]
    for i in range(len(P)):
        a,b=P[i],P[(i+1)%len(P)]; k=max(1,int(np.linalg.norm(b-a)/step))
        out+= [a+(b-a)*t/k for t in range(k)]
    return np.array(out)
