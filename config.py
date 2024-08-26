import os


# main, Model, mn, root_dir, mod_name
mod_name: tuple = (
    'random_search_RF',
)
root_dir = os.listdir(os.path.dirname(__file__))
class_names = [
    'bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
]
