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

def create_pokemon_pokeball_matrix(tries, level = 100, status_effect = StatusEffect.NONE, health_points = 1):
    pokemon_pokeball_matrix = {} 

    for name in pokemon_names:
        pokemon_pokeball_matrix[name] = {}
        pokemon = factory.create(name, level, status_effect, health_points)
        for ball in pokeballs:
            pokemon_pokeball_matrix[name][ball] = 0
            for _ in range(tries):
                pokemon_pokeball_matrix[name][ball] += attempt_catch(pokemon, ball)[0]
    return pokemon_pokeball_matrix

def create_pokemon_status_matrix(tries, level = 100, ball = 'pokeball', health_points = 1):
    pokemon_status_matrix = {} 

    for name in pokemon_names:
        pokemon_status_matrix[name] = {}
        for status_effect in StatusEffect:
            pokemon = factory.create(name, level, status_effect, health_points)
            pokemon_status_matrix[name][status_effect] = 0
            for _ in range(tries):
                pokemon_status_matrix[name][status_effect] += attempt_catch(pokemon, ball)[0]
    return pokemon_status_matrix


# In[1]: Exercise 1a

#TODO: Move to another file?
TRIES = 100


pokeball_stats = {}
pokemon_pokeball_matrix = create_pokemon_pokeball_matrix(tries=TRIES)
df = pd.DataFrame(pokemon_pokeball_matrix).transpose()

for ball in pokeballs:
    pokeball_stats[ball] = f"{df[ball].sum() / len(pokemon_names):.2f}%"

catchRateDf = pd.DataFrame(pokeball_stats, index=["Catch Rate"])
catchRateDf


#In[2] Exercise 1b

TRIES = 1000

pokemon_pokeball_matrix = create_pokemon_pokeball_matrix(tries=TRIES)


df = pd.DataFrame(pokemon_pokeball_matrix)
original_df = df.copy()
df = df.applymap(lambda x: x / TRIES)
jolteon_stats_df = df["jolteon"].copy().map(lambda x: x/df["jolteon"]["pokeball"]).to_frame()
snorlax_stats_df = df["snorlax"].copy().map(lambda x: x/df["snorlax"]["pokeball"]).to_frame()

jolteon_stats_df
jolteon_stats_df.reset_index(inplace=True)
jolteon_stats_df.columns = ["Pokeball", "Efficiency"]
jolteon_stats_df.plot.bar(x="Pokeball", y="Efficiency", 
                          ylim=(0, 5),
                          title="Efficiency of pokeballs when catching Jolteon",
                          ylabel="Relative efficiency",
                          color=['blue', 'red', 'green', 'orange'],
                          legend=False)

snorlax_stats_df
snorlax_stats_df.reset_index(inplace=True)
snorlax_stats_df.columns = ["Pokeball", "Efficiency"]
snorlax_stats_df.plot.bar(x="Pokeball", y="Efficiency", 
                          ylim=(0, 5),
                          title="Efficiency of pokeballs when catching Snorlax",
                          ylabel="Relative efficiency",
                          color=['blue', 'red', 'green', 'orange'],
                          legend=False)
plt.show()

#In[3] Exercise 2a

TRIES = 10_000

pokemon_status_matrix = create_pokemon_status_matrix(tries=TRIES)

df = pd.DataFrame(pokemon_status_matrix)
odf = df.copy()

df = df.apply(lambda x: x / x[-1])
df

status_effect_stats = df.mean(axis=1).to_frame()
status_effect_stats.columns = ["Average"]
status_effect_stats["Standard Deviation"] = df.std(axis=1)
status_effect_stats = status_effect_stats[:-1]
status_effect_stats.plot.bar(y="Average", 
                            yerr="Standard Deviation",
                            title="Average and standard deviation of status effects",
                            color=['blue', 'red', 'green', 'orange', 'purple'],
                            legend=False)




# In[4] Exercise 2b 

