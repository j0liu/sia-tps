
# TP3 SIA - Perceptrón simple y multicapa

## Introducción

Trabajo práctico para la materia Sistemas de Inteligencia Artificial.
Se implementa un perceptrón simple y multicapa para resolver distintos problemas de clasificación y regresión.

[Enunciado](docs/SIA_TP3.pdf)

### Requisitos

- Python3
- pip3
- numpy
- [pipenv](https://pypi.org/project/pipenv/)

### Modos de uso
Para correr los análisis, se debe ejecutar el siguiente comando en la raíz del proyecto:

Ej 1:
```bash 
python3 ej1.py
```
Ej 2:
```bash 
python3 ej2.py
```
Ej 3:
```bash 
python3 ej3.py <xor|paridad|numeros|ruido>
```
Donde:
- xor: Entrena un perceptrón simple para resolver el problema XOR
- paridad: Entrena un perceptrón multicapa para resolver el problema de paridad
- numeros: Entrena un perceptrón multicapa para resolver el problema de clasificación de números
- ruido: Genera un dataset con ruido para poder ver la capacidad de clasificacion de los perceptrones de paridad y numeros


En todos los casos, se debe configurar antes el archivo json del ejercicio correspondiente con aquellos parametros que se desee.
Para más información sobre como configurar los parámetros, revisar las diapositivas. 

### Alumnos
- Dithurbide, Manuel Esteban - 62057
- Liu, Jonathan Daniel - 62533
- Vilamowski, Abril - 62495
- Wischñevsky, David - 62494