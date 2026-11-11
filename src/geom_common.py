import numpy as np
L=66.80; SPAN=64.75; H=17.05
ZC=5.0; RW=2.98; RH=3.045
Y0=ZC-0.80   # nose tip height
YT=ZC+1.35   # tail-cone end centre height
NOSE_TOP=8.6; NOSE_BOT=5.2; NOSE_W=7.0
def fus_profile(s):
    s=np.asarray(s,float)
    tt=np.clip(s/NOSE_TOP,0,1); top_n=Y0+(ZC+RH-Y0)*(1-(1-tt)**2.3)**0.5
    tb=np.clip(s/NOSE_BOT,0,1); bot_n=Y0-(Y0-(ZC-RH))*(1-(1-tb)**2.0)**0.5
    tw=np.clip(s/NOSE_W,0,1);  w_n=RW*(1-(1-tw)**2.2)**0.5
    ta=np.clip((s-40.5)/(L-40.5),0,1); bot_t=(ZC-RH)+((YT-0.30)-(ZC-RH))*ta**1.55
    t2=np.clip((s-53.0)/(L-53.0),0,1); top_t=(ZC+RH)-((ZC+RH)-(YT+0.30))*t2**1.5
    t3=np.clip((s-43.0)/(L-43.0),0,1); w_t=RW-(RW-0.32)*t3**1.45
    top=np.minimum(top_n,top_t); bot=np.maximum(bot_n,bot_t); w=np.minimum(w_n,w_t)
    return (top+bot)/2,(top-bot)/2,w
# fin planform (s along body, y height)
FIN_Y0=6.6; FIN_YK=7.6
def fin_le(y): return 56.0+np.maximum(0,np.asarray(y)-FIN_YK)*0.92
def fin_chord(y):
    y=np.asarray(y,float); t=np.clip((y-FIN_YK)/(H-FIN_YK),0,1); return 10.4+(1.95-10.4)*t
FIN_S0,FIN_Y1,FIN_SZ=54.5,18.5,12.5   # fin texture window
# fuselage atlas
TEX=2048; STRIP=682; NSTRIP=3
def _warp_tables():
    th=np.linspace(0,2*np.pi,4001)
    d=1+2.2*np.exp(-((th-np.pi/2)/0.62)**4)+2.2*np.exp(-((th-3*np.pi/2)/0.62)**4)
    c=np.concatenate([[0],np.cumsum((d[1:]+d[:-1])/2*np.diff(th))]); c/=c[-1]
    return th,c
_TH,_C=_warp_tables()
def theta_to_v(th): return np.interp(th,_TH,_C)
def v_to_theta(v): return np.interp(v,_C,_TH)
def fus_uv(s,th):
    """s scalar/array, th array -> u,v in glTF convention (v down) for strip containing s"""
    k=np.minimum((np.asarray(s)/(L/NSTRIP)).astype(int),NSTRIP-1)
    return k
