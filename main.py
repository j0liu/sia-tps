# In[0]: Imports
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
import json
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt

factory = PokemonFactory("pokemon.json")

with open("pokemon.json", "r") as c:
    pokemons = json.load(c)
    pokemon_names = pokemons.keys()

with open("pokeballs.json", "r") as c:
    pokeballs = json.load(c)

def createMatrix(tries):
    pokemon_pokeball_matrix = {} 

    for name in pokemon_names:
        pokemon_pokeball_matrix[name] = {}
        pokemon = factory.create(name, 100, StatusEffect.NONE, 1)
        for ball in pokeballs:
            pokemon_pokeball_matrix[name][ball] = 0
            for _ in range(tries):
                pokemon_pokeball_matrix[name][ball] += attempt_catch(pokemon, ball)[0]
    return pokemon_pokeball_matrix


# In[1]: Exercise 1a

#TODO: Move to another file?
TRIES = 100


pokeball_stats = {}
pokemon_pokeball_matrix = createMatrix(TRIES)
df = pd.DataFrame(pokemon_pokeball_matrix).transpose()

for ball in pokeballs:
    pokeball_stats[ball] = f"{df[ball].sum() / len(pokemon_names):.2f}%"

catchRateDf = pd.DataFrame(pokeball_stats, index=["Catch Rate"])
catchRateDf


#In[2] Exercise 1b

TRIES = 1000

pokemon_pokeball_matrix = createMatrix(TRIES)


df = pd.DataFrame(pokemon_pokeball_matrix)
df = df.applymap(lambda x: x / TRIES)
jolteon_stats_df = df["jolteon"].copy().map(lambda x: x/df["jolteon"]["pokeball"]).to_frame()
snorlax_stats_df = df["snorlax"].copy().map(lambda x: x/df["snorlax"]["pokeball"]).to_frame()
# df
# snorlax_stats_df
# jolteon_stats_df

df.plot.hist(
    bins=30,
    alpha=0.5,
    title="Efficiency of pokeballs when catching Snorlax"
)

plt.show()

# %%
