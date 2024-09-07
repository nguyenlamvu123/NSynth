import os, pickle


debug: bool = False
mod_name: tuple = (
    'random_search_RF',
)
root_dir = os.listdir(os.path.dirname(__file__))
class_names = [
    'bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
]

model_listobj: list = list()  # list chứa các model đã được load

for mt in mod_name:  # tuple danh sách tên các model
    with open(f'{mt}.pickle', 'rb') as f:
        model_obj = pickle.load(f)
    model_listobj.append(model_obj)


class Test_path:
    # Path for Test Files
    data_path = '/home/zaibachkhoa/Documents/Music-Genre-Classification-From-Audio-Files/Music_Instrument_Classification/dataset/test/'
    # data_path = '/home/zaibachkhoa/Documents/Music-Genre-Classification-From-Audio-Files/Music_Instrument_Classification/dataset/valid/'
