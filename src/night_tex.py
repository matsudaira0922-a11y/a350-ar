import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, gaussian_filter
from geom_common import *
import livery as lv
TINT=np.array([1.0,0.90,0.72])          # warm white floodlight
GLOW=np.array([1.0,0.72,0.38])          # cabin light
def lin(c): c=np.asarray(c,float)/255.0; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def srgb(l): l=np.clip(l,0,1); return np.where(l<=0.0031308,l*12.92,1.055*l**(1/2.4)-0.055)
def light_I(y,ny,nh):
    """baked floodlight intensity from ground: strong low + facing down, fading up."""
    A=0.12+0.88*np.exp(-(np.maximum(y,0)/6.0)**1.25)
    Fc=0.10+0.90*np.clip(np.maximum(-ny,0)*1.0+nh*0.45,0,1)
    return np.clip(A*Fc*1.12,0.03,1.0)
def decals_glow():
    D=Image.new('L',(lv.DW,lv.DH),0); dr=ImageDraw.Draw(D); S=lambda s:s*lv.PPM; Y=lambda y:(10-y)*lv.PPM
    for a0,a1 in ((6.35,14.45),(16.15,36.55),(38.3,49.3)):
        for i in range(int((a1-a0)/0.635)+1):
            sc=a0+i*0.635
            if sc>a1: break
            w,h=0.27,0.40; yc=ZC+0.45
            dr.rounded_rectangle((S(sc-w/2),Y(yc+h/2),S(sc+w/2),Y(yc-h/2)),radius=int(0.11*lv.PPM),fill=255)
    for sc in (5.3,15.25,37.4,49.95):
        dr.rounded_rectangle((S(sc-0.09),Y(ZC+0.68),S(sc+0.09),Y(ZC+0.44)),radius=5,fill=200)
    return np.asarray(D).astype(float)/255
def fuselage_emissive():
    base=np.asarray(Image.open('tex_fuselage.jpg').convert('RGB')).astype(float)
    G=decals_glow()
    out=np.zeros((TEX,TEX,3)); dif=np.zeros((TEX,TEX,3))
    cols=np.arange(TEX)
    for k in range(NSTRIP):
        r0=k*(STRIP+1); rr=np.arange(STRIP)
        th=v_to_theta((rr+0.5)/STRIP); s=(k+(cols+0.5)/TEX)*L/NSTRIP
        Sg,Tg=np.meshgrid(s,th); yc,hh,w=fus_profile(Sg)
        y=yc-hh*np.cos(Tg); den=np.sqrt((hh*np.sin(Tg))**2+(w*np.cos(Tg))**2)+1e-9
        ny=-w*np.cos(Tg)/den; nh=np.abs(hh*np.sin(Tg))/den
        I=light_I(y,ny,nh)
        # soft floodlight pools along the length (lights placed every ~14 m)
        I=I*(0.88+0.12*np.cos(2*np.pi*(Sg-6.0)/14.0)**2)
        bl=lin(base[r0:r0+STRIP])
        e=bl*I[...,None]*TINT
        g=map_coordinates(G,[(10-y)*lv.PPM,Sg*lv.PPM],order=1,mode='nearest')
        e=e+g[...,None]*GLOW*0.62
        out[r0:r0+STRIP]=e
        dif[r0:r0+STRIP]=bl*(0.32+0.68*I[...,None]*TINT)+g[...,None]*GLOW*0.22
        if k<NSTRIP-1: dif[r0+STRIP]=dif[r0+STRIP-1]
        if k<NSTRIP-1: out[r0+STRIP]=e[-1]
    out[-1]=out[-2]
    dif[-1]=dif[-2]
    im=Image.fromarray((srgb(np.clip(out,0,1))*255+0.5).astype(np.uint8)); im.save('tex_fuselage_night.jpg',quality=90,optimize=True,subsampling=0)
    Image.fromarray((srgb(np.clip(dif,0,1))*255+0.5).astype(np.uint8)).save('tex_fuselage_nightdiff.jpg',quality=90,optimize=True,subsampling=0)
def fin_emissive(size=1024):
    base=np.asarray(Image.open('tex_fin.jpg').convert('RGB')).astype(float); ppm=size/FIN_SZ
    rows,cols=np.mgrid[0:size,0:size]; y=FIN_Y1-(rows+0.5)/ppm; s=FIN_S0+(cols+0.5)/ppm
    I=light_I(y,0*y,1+0*y)
    yc=12.55; sc=float(fin_le(yc)+fin_chord(yc)*0.52)
    d2=((s-sc)/3.0)**2+((y-yc)/3.2)**2
    spot=np.exp(-d2**1.4)                      # spotlight from below, flat top, soft edge
    spot=spot*np.clip(0.75+0.25*(y-FIN_Y0)/(H-FIN_Y0),0,1)
    I=np.maximum(I,0.10+0.90*spot)
    e=lin(base)*I[...,None]*TINT
    im=Image.fromarray((srgb(np.clip(e,0,1))*255+0.5).astype(np.uint8)); im.save('tex_fin_night.jpg',quality=92,optimize=True,subsampling=0)
    d=lin(base)*(0.32+0.68*I[...,None]*TINT)
    Image.fromarray((srgb(np.clip(d,0,1))*255+0.5).astype(np.uint8)).save('tex_fin_nightdiff.jpg',quality=92,optimize=True,subsampling=0)
def ramp():
    x=np.arange(256)/255.0
    row=srgb(x[:,None]*TINT[None,:])
    arr=np.repeat(row[None,:,:],4,0); Image.fromarray((arr*255+0.5).astype(np.uint8)).save('ramp_night.png',optimize=True)
if __name__=='__main__':
    fuselage_emissive(); fin_emissive(); ramp(); print('night textures ok')
