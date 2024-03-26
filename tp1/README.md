
# TP1 SIA - Métodos de Búsqueda

## Introducción

Trabajo práctico para la materia Sistemas de Inteligencia Artificial.
Se implementa un motor de búsqueda para resolver el juego sokoban. 

[Enunciado](docs/SIA_TP1.pdf)

### Requisitos

- Python3
- pip3
- numpy
- [pipenv](https://pypi.org/project/pipenv/)

### Modos de uso
Para correr el programa, se debe ejecutar el siguiente comando en la raíz del proyecto:
```bash 
python3 main.py <single|multiple|heuristics> 
```
Donde cada parámetro significa:
- Single: corre la simulación una sola vez, con la configuración provista en el archivo `sokoban_config.json`
- Multiple: corre la simulación múltiples veces, para cada metodo de búsqueda, con la configuración provista en el archivo `sokoban_config.json`. Ignora el método de búsqueda. Recomendación: empezar con todas las optimizaciones en false
- Heuristics: corre la simulación múltiples veces, para cada heurística, con la configuración provista en el archivo `sokoban_config.json`. Ignora la heurística.


### Alumnos
- Dithurbide, Manuel Esteban - 62057
- Liu, Jonathan Daniel - 62533
- Vilamowski, Abril - 62495
- Wischñevsky, David - 62494