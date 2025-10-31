"""
Práctica 3: Sistema musculoesqueletico

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Marco Antonio Garcia Montilla
Número de control: 22211756
Correo institucional: l22211756@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install control
#!pip install slycot

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m 
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

# Datos de la simulación
x0,t0,tend,dt,w,h = 0,0,10,1E-3,7,3.5
n = round((tend - t0)/dt)+1
u1 = np.zeros(n); u1[round(1/dt):round(2/dt)] = 1
t = np.linspace(t0,tend,n)

def cardio(Cp,Cs,R):
    num = [Cs*R,0.75]
    den=[R*(Cs+Cp),1]
    sys = ctrl.tf(num,den)
    return sys

Cp,Cs,R = 100E-6,10E-6,100
syscontrol = cardio(Cp,Cs,R)
print(f'Funcion de transferencia del control:{syscontrol}')

Cp,Cs,R = 100E-6,10E-6,10000
syscaso = cardio(Cp,Cs,R)
print(f'Funcion de transferencia del caso:{syscaso}')


#Respuesta en lazo abierto
_,Fs1 = ctrl.forced_response(syscontrol,t,u1,x0)
_,Fs2 = ctrl.forced_response(syscaso,t,u1,x0)


fg1 = plt.figure()
plt.plot(t,Fs1,'-',linewidth=1,color=np.array([240,128,48])/255,label= 'Fs(1): Control')
plt.plot(t,Fs2,'-',linewidth=1,color=np.array([15,130,140])/255,label= 'Fs(2): Caso')
plt.plot(t,u1,'-',linewidth=1,color=np.array([255,221,51])/255,label= 'Fs(t): Entrada')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1);plt.yticks(np.arange(-0.1,1.1,0.2))
plt.ylabel('t[s]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()

fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema musculoesqueletico python.png',dpi=600,bbox_inches='tight')
fg1.savefig('sistema musculoesqueletico python.pdf')

#controladorPI
def controlador(kP,kI,sys):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    X = ctrl.series(PI,sys)
    sysPI = ctrl.feedback(X,1,sign=-1)
    return sysPI

musPI = controlador(0.025282925393564,42153.5078231096,syscaso)

    
#respuestas en lazo cerrado
_,Fs3 = ctrl.forced_response(musPI,t,Fs1,x0)

fg2 = plt.figure()
plt.plot(t,Fs1,'-',linewidth=1,color=np.array([240,128,48])/255,label= 'Fs(1): Control')
plt.plot(t,Fs2,'-',linewidth=1,color=np.array([15,130,140])/255,label= 'Fs(2): Caso')
plt.plot(t,Fs3,'--',linewidth=1,color=np.array([107,36,12])/255,label= 'Fs(3): Tratamiento')
plt.plot(t,u1,'-',linewidth=1,color=np.array([255,221,51])/255,label= 'Fs(t): Entrada')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1);plt.yticks(np.arange(-0.1,1.1,0.2))
plt.ylabel('t[s]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()

fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('sistema musculoesqueletico python PI.png',dpi=600,bbox_inches='tight')
fg2.savefig('sistema musculoesqueletico python PI.pdf')

