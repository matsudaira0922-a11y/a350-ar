import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates
from matplotlib.path import Path
from geom_common import *
PPM=100  # decal canvas px per metre
DW,DH=int(L*PPM)+1, 1000  # y 0..10 m
FONT_SANS='/usr/share/fonts/truetype/sand-box/google/Noto Sans/NotoSans-VariableFont_wdth,wght.ttf'
FONT_HUBOT='/usr/share/fonts/truetype/sand-box/google/Hubot Sans/HubotSans-Italic-VariableFont_wdth,wght.ttf'
JAL_RED=(204,0,17); INK=(20,26,46)

def text_img(txt, font_path, size=400, wght=800, wdth=None, shear=0.0, fill=(0,0,0)):
    f=ImageFont.truetype(font_path,size)
    try:
        axes=f.get_variation_axes(); vals=[]
        for a in axes:
            n=a['name'] if isinstance(a['name'],str) else a['name'].decode()
            if n.lower().startswith('weight'): vals.append(wght)
            elif n.lower().startswith('width'): vals.append(wdth if wdth else a['default'])
            else: vals.append(a['default'])
        f.set_variation_by_axes(vals)
    except Exception as e: pass
    bb=f.getbbox(txt); w,h=bb[2]-bb[0]+40, bb[3]-bb[1]+40
    im=Image.new('L',(w+int(abs(shear)*h)+40,h),0)
    ImageDraw.Draw(im).text((20-bb[0]+int(abs(shear)*h),20-bb[1]),txt,font=f,fill=255)
    if shear: im=im.transform(im.size,Image.AFFINE,(1,shear,0,0,1,0),resample=Image.BICUBIC)
    im=im.crop(im.getbbox())
    return im  # alpha mask

def place(layer, mask, s0, s1, ytop, ybot, color, mirror=False):
    """paste mask (L image) stretched into box [s0,s1]x[ybot,ytop] metres"""
    w=int(round((s1-s0)*PPM)); h=int(round((ytop-ybot)*PPM))
    m=mask.resize((w,h),Image.LANCZOS)
    if mirror: m=m.transpose(Image.FLIP_LEFT_RIGHT)
    x=int(round(s0*PPM)); y=int(round((10-ytop)*PPM))
    layer.paste(Image.new('RGB',(w,h),color),(x,y),m)

def rrect_mask(w,h,r,scale=4):
    im=Image.new('L',(w*scale,h*scale),0)
    ImageDraw.Draw(im).rounded_rectangle((0,0,w*scale-1,h*scale-1),r*scale,fill=255)
    return im

def decals(mirror):
    D=Image.new('RGB',(DW,DH),(244,245,246))
    dr=ImageDraw.Draw(D)
    def Y(y): return (10-y)*PPM
    def S(s): return s*PPM
    # --- titles
    ja=text_img('JAPAN AIRLINES',FONT_HUBOT,400,wght=850,wdth=108)
    place(D,ja,7.8,24.7,ZC+1.94,ZC+1.02,INK,mirror)
    # flag
    fx0,fx1,fy0,fy1=25.25,25.95,ZC+1.25,ZC+1.71
    dr.rectangle((S(fx0),Y(fy1),S(fx1),Y(fy0)),fill=(252,252,252),outline=(200,200,205),width=1)
    cx,cy,rr=S((fx0+fx1)/2),Y((fy0+fy1)/2),0.14*PPM; dr.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=(190,0,40))
    # AIRBUS A350 block (s 38.4..49.4)
    blk=Image.new('L',(int(11.0*PPM),int(2.8*PPM)),0)  # 11m x 2.8m, top=ZC+1.95
    airbus=text_img('AIRBUS',FONT_SANS,300,wght=800).resize((int(4.0*PPM),int(0.62*PPM)),Image.LANCZOS)
    blk.paste(airbus,(0,int(0.06*PPM)))
    a350=text_img('A350',FONT_HUBOT,600,wght=900,wdth=100,shear=0.22).resize((int(7.0*PPM),int(2.8*PPM)),Image.LANCZOS)
    a=np.array(a350).astype(float)/255
    hh,ww=a.shape; yy,xx=np.mgrid[0:hh,0:ww]
    period=0.25*PPM; frac=xx/ww
    gap=(0.012+0.15*np.clip(frac-0.05,0,1)**1.2)*PPM
    stripes=((yy % period) > gap).astype(float)
    a=(a*stripes*255).astype(np.uint8)
    blk.paste(Image.fromarray(a),(int(4.0*PPM),0),Image.fromarray(a))
    place(D,blk,38.4,49.4,ZC+1.95,ZC-0.85,JAL_RED,mirror)
    # registration & small type
    reg=text_img('JA01XJ',FONT_SANS,300,wght=600)
    place(D,reg,51.0,53.0,ZC+1.62,ZC+1.12,INK,mirror)
    sm=text_img('AIRBUS A350-900',FONT_SANS,200,wght=700)
    place(D,sm,51.0,53.3,ZC-0.42,ZC-0.58,INK,mirror)
    # --- doors (outline + small window)
    for sc in (5.3,15.25,37.4,49.95):
        w,h=1.07,1.95; x0,x1=S(sc-w/2),S(sc+w/2); y0,y1=Y(ZC+1.12),Y(ZC+1.12-h)
        dr.rounded_rectangle((x0,y0,x1,y1),radius=int(0.22*PPM),outline=(150,152,158),width=3)
        dr.rounded_rectangle((S(sc-0.09),Y(ZC+0.68),S(sc+0.09),Y(ZC+0.44)),radius=5,fill=(55,62,75))
        dr.rectangle((S(sc-0.25),Y(ZC-0.05),S(sc+0.25),Y(ZC-0.12)),outline=(185,187,192),width=1)  # handle plate
    # cargo doors (starboard only on A350)
    if mirror:
        for c0,c1 in ((9.4,12.1),(40.2,42.9)):
            dr.rounded_rectangle((S(c0),Y(ZC-0.55),S(c1),Y(ZC-2.25)),radius=12,outline=(165,167,172),width=2)
    # --- windows
    pitch=0.635
    for a0,a1 in ((6.35,14.45),(16.15,36.55),(38.3,49.3)):
        n=int((a1-a0)/pitch)+1
        for i in range(n):
            sc=a0+i*pitch
            if sc>a1: break
            w,h=0.27,0.40; yc=ZC+0.45
            dr.rounded_rectangle((S(sc-w/2-0.03),Y(yc+h/2+0.03),S(sc+w/2+0.03),Y(yc-h/2-0.03)),radius=int(0.12*PPM),fill=(205,208,214))
            dr.rounded_rectangle((S(sc-w/2),Y(yc+h/2),S(sc+w/2),Y(yc-h/2)),radius=int(0.11*PPM),fill=(38,44,56))
            dr.rectangle((S(sc-w/2+0.04),Y(yc+h/2-0.05),S(sc-w/2+0.08),Y(yc+0.02)),fill=(80,90,108))  # glint
    # small details: static ports, emergency markings near doors
    for sc in (3.4,4.0): dr.ellipse((S(sc)-3,Y(ZC-0.6)-3,S(sc)+3,Y(ZC-0.6)+3),fill=(150,150,155))
    return D

def paint_fuselage():
    Dp=np.asarray(decals(False)).astype(float); Ds=np.asarray(decals(True)).astype(float)
    Image.fromarray(Dp.astype(np.uint8)).resize((DW//4,DH//4)).save('debug_decal_port.png')
    out=np.zeros((TEX,TEX,3))
    rows=np.arange(TEX); cols=np.arange(TEX)
    for k in range(NSTRIP):
        r0=k*(STRIP+1); rr=np.arange(STRIP)
        vloc=(rr+0.5)/STRIP; th=v_to_theta(vloc)
        s=(k+(cols+0.5)/TEX)*L/NSTRIP
        Sg,Tg=np.meshgrid(s,th)  # (STRIP,TEX)
        yc,hh,w=fus_profile(Sg)
        y=yc-hh*np.cos(Tg); side=np.sin(Tg)
        px=Sg*PPM; py=(10-y)*PPM
        img=np.zeros(Sg.shape+(3,))
        for D,msk in ((Dp,side>=0),(Ds,side<0)):
            for c in range(3):
                vals=map_coordinates(D[:,:,c],[py,px],order=1,mode='nearest')
                img[:,:,c]=np.where(msk,vals,img[:,:,c])
        # nose cockpit mask in (s, phi from top in deg)
        phi=np.degrees(np.abs(Tg-np.pi))
        poly=Path([(1.95,0),(2.70,0),(2.98,24),(3.45,48),(4.05,60),(4.45,64),(4.55,69),(4.30,73),(3.75,75),(3.20,72),(2.70,62),(2.30,44),(2.05,22)])
        pts=np.c_[Sg.ravel(),phi.ravel()]
        inm=poly.contains_points(pts).reshape(Sg.shape)
        # soft edge via small blur later; mask colour
        pass  # cockpit mask is real geometry now (nosewin.py)
        # subtle panel lines & belly shade
        for sl in (2.05,13.2,21.4,41.0,52.6,58.0):
            img[np.abs(Sg-sl)<0.012]*=0.9
        belly=np.clip((-np.cos(Tg)-0.55)/0.45,0,1)  # bottom
        img*= (1-0.06*belly)[...,None]
        out[r0:r0+STRIP]=img
        if k<NSTRIP-1: out[r0+STRIP]=img[-1]
    out[-1]=out[-2]
    im=Image.fromarray(np.clip(out,0,255).astype(np.uint8))
    im.save('tex_fuselage.jpg',quality=90,optimize=True,subsampling=0)
    return im

def crane_logo(N=2000):
    R=N/2; im=Image.new('RGBA',(N,N),(0,0,0,0)); d=ImageDraw.Draw(im)
    def P(x,y): return (R+x*R, R+y*R)
    red=JAL_RED+(255,); white=(255,255,255,255)
    d.ellipse((0,0,N-1,N-1),fill=red)
    # white inner field (rounded top, tapering to a point at the base of the neck)
    def tapered(path,w0,w1,col):
        path=np.array(path); n=len(path)
        tang=np.gradient(path,axis=0); tang/=np.linalg.norm(tang,axis=1)[:,None]+1e-9
        nor=np.c_[-tang[:,1],tang[:,0]]; ws=np.linspace(w0,w1,n)[:,None]
        poly=np.vstack([path+nor*ws,(path-nor*ws)[::-1]])
        d.polygon([P(x,y) for x,y in poly],fill=col)
    t=np.linspace(-np.pi*0.5,np.pi*0.5,160)
    top=[(0.80*np.sin(a),-0.08-0.80*np.cos(a)) for a in t]
    right=[(0.80*(1-u)**0.75, -0.08*(1-u)+0.38*u) for u in np.linspace(0,1,40)]
    left=[(-x,y) for x,y in right[::-1]]
    d.polygon([P(*q) for q in top+right+left],fill=white)
    # white feather slits in the red wings
    for sgn in (-1,1):
        for rad,a0,a1,w in ((0.86,176,214,0.024),(0.74,186,224,0.024),(0.62,196,232,0.022)):
            a=np.radians(np.linspace(a0,a1,60))
            path=np.c_[sgn*-rad*np.cos(np.pi-a)*-1*-1, -0.04-rad*np.sin(a)*-1]
            path=np.c_[sgn*rad*np.cos(a), -0.04-rad*np.sin(a)]
            tapered(path,w*1.6,w*0.3,white)
    # neck S-curve (red), from body base up to head
    def bez(p0,p1,p2,p3,n=80):
        t=np.linspace(0,1,n)[:,None]
        return (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t**2*p2+t**3*p3
    p=np.vstack([bez(np.array([-0.02,0.46]),np.array([-0.30,0.20]),np.array([0.05,0.02]),np.array([0.30,-0.20])),
                 bez(np.array([0.30,-0.20]),np.array([0.48,-0.36]),np.array([0.32,-0.58]),np.array([0.10,-0.50]))])
    tapered(p,0.10,0.065,red)
    # head + beak
    d.ellipse((*P(0.02,-0.585),*P(0.20,-0.43)),fill=red)
    d.polygon([P(0.06,-0.565),P(-0.52,-0.545),P(0.06,-0.465)],fill=red)
    # JAL letters
    jal=text_img('JAL',FONT_HUBOT,500,wght=900,wdth=110,shear=0.18).resize((int(1.02*R),int(0.30*R)),Image.LANCZOS)
    im.paste(Image.new('RGBA',jal.size,white),(int(R-0.51*R),int(R+0.50*R)),jal)
    # clip to circle
    m=Image.new('L',(N,N),0); ImageDraw.Draw(m).ellipse((0,0,N-1,N-1),fill=255)
    a=np.array(im); a[...,3]=np.minimum(a[...,3],np.array(m)); return Image.fromarray(a)

def paint_fin(size=1024):
    ppm=size/FIN_SZ
    im=Image.new('RGB',(size,size),(244,245,246)); d=ImageDraw.Draw(im)
    # rudder hinge line & panel lines (subtle)
    for y in np.linspace(FIN_Y0,H,40):
        pass
    ys=np.linspace(7.6,H,50); sx=[(fin_le(y)+fin_chord(y)*0.70-FIN_S0)*ppm for y in ys]
    d.line(list(zip(sx,[(FIN_Y1-y)*ppm for y in ys])),fill=(214,216,220),width=2)
    logo=crane_logo()
    yc=12.55; sc=float(fin_le(yc)+fin_chord(yc)*0.52); D=4.5
    lg=logo.resize((int(D*ppm),int(D*ppm)),Image.LANCZOS)
    im.paste(lg,(int((sc-D/2-FIN_S0)*ppm),int((FIN_Y1-yc-D/2)*ppm)),lg)
    im.save('tex_fin.jpg',quality=92,optimize=True,subsampling=0)
    logo.resize((800,800)).save('debug_logo.png')
    return im

if __name__=='__main__':
    paint_fin(); paint_fuselage(); print('textures done')
