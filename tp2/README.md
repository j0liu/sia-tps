
# TP2 SIA - Algortimos Genéticos 

## Introducción

Trabajo práctico para la materia Sistemas de Inteligencia Artificial.
Se implementa un algoritmo genético para encontrar las mejores combinaciones de parametros para un RPG. 

[Enunciado](docs/SIA_TP2.pdf)

### Requisitos

- Python3
- pip3
- numpy
- [pipenv](https://pypi.org/project/pipenv/)

### Modos de uso
Para correr los análisis, se debe ejecutar el siguiente comando en la raíz del proyecto:
```bash 
python3 benchmarks.py <warrior|archer|defender|infiltrate|all> <crossover|replacement|mutation|crossover_selection|boltzmann|deterministic_tournament|replacement_selection|mutation_function|stopping_condition|parents>
```
En todos los casos, se debe configurar antes el archivo config.json con aquellos parametros que se desee mantener constantes. 
Para más información sobre como configurar los parámetros, revisar el archivo `config.json.template`

### Alumnos
- Dithurbide, Manuel Esteban - 62057
- Liu, Jonathan Daniel - 62533
- Vilamowski, Abril - 62495
- Wischñevsky, David - 62494