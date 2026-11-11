import numpy as np, trimesh
from trimesh.visual.material import PBRMaterial
L, SPAN, H = 66.80, 64.75, 17.05
W_F, H_F = 5.96, 6.09   # fuselage width/height
ZC = 4.6                # fuselage centerline height above ground (approx)
N = 40

def loft(rings, cap=True):
    """rings: list of (M,3) arrays, same M, closed loops."""
    M = len(rings[0]); V = np.vstack(rings); F=[]
    for i in range(len(rings)-1):
        for j in range(M):
            a=i*M+j; b=i*M+(j+1)%M; c=(i+1)*M+(j+1)%M; d=(i+1)*M+j
            F += [[a,b,c],[a,c,d]]
    if cap:
        for idx,ring in ((0,rings[0]),(len(rings)-1,rings[-1])):
            ci=len(V); V=np.vstack([V,ring.mean(0)])
            for j in range(M):
                a=idx*M+j; b=idx*M+(j+1)%M
                F.append([ci,b,a] if idx==0 else [ci,a,b])
    m=trimesh.Trimesh(V,np.array(F),process=True)
    m.fix_normals()
    return m

# ---- fuselage: s = distance from nose (0..L), frame: X right, Y up, Z forward(nose)
def fus_section(s):
    if s < 7.5:
        t = s/7.5; k = np.sqrt(max(1-(1-t)**2.2,0))
        w, h = W_F/2*k, H_F/2*k; yc = ZC - 0.9*(1-t)**2
    elif s < 44:
        w, h, yc = W_F/2, H_F/2, ZC
    else:
        t = (s-44)/(L-44)
        top = ZC+H_F/2 - 1.6*t**1.5
        bot = ZC-H_F/2 + (H_F-1.6-0.9)*t**0.9
        h = max((top-bot)/2,0.35); yc=(top+bot)/2
        w = max(W_F/2*(1-t**1.3*0.88),0.35)
    return w,h,yc
th = np.linspace(0,2*np.pi,N,endpoint=False)
ss = np.concatenate([np.linspace(0,7.5,16)**1, np.linspace(9,44,8), np.linspace(46,L,14)])
ss[0]=0.0
rings=[]
for s in ss:
    w,h,yc = fus_section(s)
    if s==0: w,h=0.05,0.05
    rings.append(np.c_[w*np.cos(th), yc+h*np.sin(th), np.full(N, L/2 - s)])
fus = loft(rings)

# ---- airfoil-based lifting surface
af_x = np.array([0,0.02,0.08,0.2,0.35,0.55,0.75,1.0])
af_t = np.array([0,0.045,0.08,0.10,0.095,0.075,0.045,0.005])  # half thickness/chord ratio scale
def airfoil(chord, thick):
    up = np.c_[af_x*chord, af_t*chord*thick/0.10]
    lo = np.c_[af_x[::-1][1:-1]*chord, -af_t[::-1][1:-1]*chord*thick/0.10*0.8]
    return np.vstack([up, lo])  # (x along chord, t)
def surface(stations, vertical=False):
    """stations: list of (le_s, chord, pos_x, pos_y, thick). Chord along -Z from LE (s = dist from nose)."""
    rings=[]
    for le_s, chord, px, py, thick in stations:
        a = airfoil(chord, thick)
        z = L/2 - (le_s + a[:,0])
        if vertical: rings.append(np.c_[px + a[:,1], np.full(len(a),py), z])
        else:        rings.append(np.c_[np.full(len(a),px), py + a[:,1], z])
    return loft(rings)

# wings: semi-span reaches SPAN/2 at sharklet tip
sweep = np.tan(np.radians(31.9))
def wing(side):
    st=[]
    root_s, root_c, ybase = 23.0, 13.5, ZC-1.9
    pts = [(0.0,root_c,0.13),(2.9,root_c,0.13),(10.5,7.6,0.11),(29.6,2.6,0.09)]
    for x,c,t in pts:
        le = root_s + x*sweep + (0 if x<10 else 0)
        if x>=2.9 and x<=10.5: le = root_s + x*sweep + (x-2.9)*0.25
        if x>10.5: le = root_s + 10.5*sweep + 7.6*0.25 + (x-10.5)*sweep
        y = ybase + x*np.tan(np.radians(5.5))
        st.append((le,c,side*x,y,t))
    # sharklet blend: curve up and out to x = SPAN/2
    lx,lc,_,ly,_ = (st[-1][0],st[-1][1],None,st[-1][3],None)
    for k,(dx,dy,dc) in enumerate([(1.4,0.35,0.85),(2.3,1.3,0.65),(2.775,3.0,0.45)]):
        st.append((lx+0.6*(k+1)+dy*0.6, lc*dc, side*(29.6+dx), ly+dy, 0.08))
    return surface(st)
wings = [wing(1), wing(-1)]
# wing-body fairing
fair = trimesh.creation.capsule(height=16, radius=1.0, count=[16,16])
fair.apply_scale([3.2,1.4,1.0])
fair.apply_translation([0, ZC-2.3, L/2-31])

# horizontal stabilizer (span ~18.9m)
hs_s, hs_c, hs_y = 56.5, 6.2, ZC+0.9
def hstab(side):
    sw=np.tan(np.radians(34))
    st=[(hs_s,hs_c,0,hs_y,0.10),(hs_s+1.2*sw,hs_c,side*1.2,hs_y+0.1,0.10),(hs_s+9.45*sw,2.0,side*9.45,hs_y+0.9,0.08)]
    return surface(st)
hstabs=[hstab(1),hstab(-1)]
# vertical fin -> top at H
vf_s = 52.5; fsw=np.tan(np.radians(40))
vbase = ZC + 1.8
vfin = surface([(vf_s,10.5,0,vbase,0.11),(vf_s+(H-vbase-0.4)*fsw,3.6,0,H-0.4,0.09),(vf_s+(H-vbase)*fsw+0.3,3.0,0,H,0.08)], vertical=True)
# vertical surface built with py = height: rebuild rings properly
def fin():
    rings=[]
    for le,chord,hgt,thick in [(vf_s,10.5,vbase,0.11),(vf_s+(H-vbase-0.4)*fsw,3.6,H-0.4,0.09),(vf_s+(H-vbase)*fsw+0.2,3.0,H,0.06)]:
        a=airfoil(chord,thick)
        rings.append(np.c_[a[:,1], np.full(len(a),hgt), L/2-(le+a[:,0])])
    return loft(rings)
vfin = fin()

# engines (Trent XWB: nacelle ~3.75m dia, ~7m long)
def nacelle(xpos):
    prof=[(0,1.70),(0.3,1.88),(1.5,1.92),(4.0,1.75),(5.6,1.2),(6.6,0.8),(7.4,0.35)]
    rings=[]; nth=np.linspace(0,2*np.pi,32,endpoint=False)
    s0 = 26.3; yc = ZC-1.75
    for d,r in prof:
        rings.append(np.c_[xpos+r*np.cos(nth), yc+r*np.sin(nth), np.full(32,L/2-(s0+d))])
    n=loft(rings)
    fan=trimesh.creation.cylinder(radius=1.5,height=0.1,sections=32)
    fan.apply_translation([xpos,yc,L/2-s0-0.25])
    pylon=trimesh.creation.box([0.6,1.6,9.0]); pylon.apply_translation([xpos,yc+2.2,L/2-(s0+6.0)])
    return n, fan, pylon
eng=[nacelle(10.3), nacelle(-10.3)]

# landing gear: nose at s=6.6, mains at s=35.3 (wheelbase 28.67), track 10.6
gear=[]; tires=[]
def leg(x,s,top):
    c=trimesh.creation.cylinder(radius=0.18,height=top-0.6,sections=12)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    c.apply_translation([x,(top+0.6)/2,L/2-s]); gear.append(c)
def tire(x,s,r=0.68,w=0.5):
    t=trimesh.creation.cylinder(radius=r,height=w,sections=20)
    t.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
    t.apply_translation([x,r,L/2-s]); tires.append(t)
leg(0,6.6,ZC-2.5); tire(0.32,6.6,0.6,0.35); tire(-0.32,6.6,0.6,0.35)
for side in (1,-1):
    leg(side*5.3,35.3,ZC-2.2)
    for ds in (-1.0,0.6,2.2) if False else (-0.8,0.8):
        for dx in (-0.55,0.55):
            tire(side*5.3+dx,35.3+ds,0.68,0.5)
# cockpit windows (dark band)
cw=trimesh.creation.box([2.6,0.45,1.2]); cw.apply_translation([0,ZC+1.55,L/2-4.6])
cw.apply_transform(np.eye(4))

def mat(name,rgb,metal=0.1,rough=0.5):
    return PBRMaterial(name=name,baseColorFactor=list(rgb)+[1.0],metallicFactor=metal,roughnessFactor=rough)
white=mat('White',(0.93,0.94,0.95),0.05,0.35); grey=mat('Grey',(0.62,0.65,0.70),0.3,0.45)
dark=mat('Dark',(0.08,0.09,0.11),0.1,0.25); black=mat('Tire',(0.04,0.04,0.04),0.0,0.9)
blue=mat('Tail',(0.05,0.22,0.55),0.05,0.4); metalm=mat('Metal',(0.55,0.56,0.58),0.8,0.35)
parts={}
def add(name,m,material):
    m.visual=trimesh.visual.TextureVisuals(material=material); parts[name]=m
add('Fuselage',fus,white); add('Fairing',fair,grey)
add('Wings',trimesh.util.concatenate(wings),grey)
add('HStab',trimesh.util.concatenate(hstabs),grey)
add('Fin',vfin,blue)
add('Nacelles',trimesh.util.concatenate([e[0] for e in eng]),white)
add('Fans',trimesh.util.concatenate([e[1] for e in eng]),dark)
add('Pylons',trimesh.util.concatenate([e[2] for e in eng]),grey)
add('Gear',trimesh.util.concatenate(gear),metalm)
add('Tires',trimesh.util.concatenate(tires),black)
add('CockpitWindows',cw,dark)

# exact fit: compute bounds, scale per axis to exact numbers, origin = ground center
allv=np.vstack([p.vertices for p in parts.values()])
mn,mx=allv.min(0),allv.max(0); size=mx-mn
print('raw size X(span),Y(height),Z(length):',size)
target=np.array([SPAN,H,L]); sc=target/size
ctr=np.array([(mn[0]+mx[0])/2, mn[1], (mn[2]+mx[2])/2])
for p in parts.values():
    p.vertices=(p.vertices-ctr)*sc
    p.fix_normals()
scene=trimesh.Scene()
for k,p in parts.items(): scene.add_geometry(p,node_name=k,geom_name=k)
scene.export('a350-900.glb', include_normals=True)
allv=np.vstack([p.vertices for p in parts.values()])
print('final bounds min',allv.min(0),'max',allv.max(0),'size',allv.max(0)-allv.min(0))
print('tris',sum(len(p.faces) for p in parts.values()))
import pickle; pickle.dump({k:(p.vertices,p.faces,p.visual.material.baseColorFactor,p.visual.material.metallicFactor,p.visual.material.roughnessFactor) for k,p in parts.items()},open('parts.pkl','wb'))
