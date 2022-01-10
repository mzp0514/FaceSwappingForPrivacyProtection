

def mapAgeToAgeGroup(age):
    if age >= 65: age_group = 0
    elif age >= 55 and age < 65: age_group = 1
    elif age >= 45 and age < 55: age_group = 2
    elif age >= 35 and age < 45: age_group = 3
    elif age >= 25 and age < 35: age_group = 4
    elif age >= 15 and age < 25: age_group = 5
    elif age >= 5 and age < 15: age_group = 6
    else: age_group = 7
    return age_group

# [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
def mapRaceToRaceGroup(race):
    if race == "white": race_group = 0
    elif race == "black": race_group = 1
    elif race == "asian": race_group = 2
    elif race == "indian": race_group = 3
    else: race_group = 4
    return race_group  