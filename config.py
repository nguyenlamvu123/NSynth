import os, pickle


os.environ['TORCH_HOME'] = './pret/'
debug: bool = False
mod_name: tuple = (
    'random_search_RF',
)
root_dir = os.listdir(os.path.dirname(__file__))
class_names = [
    'bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
]
mult_res: int = 4

model_listobj: list = list()  # list chứa các model đã được load

for mt in mod_name:  # tuple danh sách tên các model
    with open(f'{mt}.pickle', 'rb') as f:
        model_obj = pickle.load(f)
    model_listobj.append(model_obj)


def dmu_cli(song) -> list:
    return [
            # "--mp3",
            "--two-stems",
            "vocals",
            "-n",
            "mdx_extra",
            song
        ]


def confirm_each_result_is_sorted(test_Y_hat, result_s):
    tyh = list(test_Y_hat)
    for i, sampl in enumerate(tyh):
        result = result_s[i]
        sampl = list(sampl)
        # confi = {class_names[ind]: sampl[ind] for ind in range(len(sampl))}
        confi = {ind: sampl[ind] for ind in range(len(sampl))}
        lis = [confi[i_] for i_ in result]
        # assert lis[0] == min(lis)
        # assert lis[-1] == max(lis)
        st = lis.copy()
        st.sort()
        assert st == lis


class Test_path:
    # Path for Test Files
    data_path = '/home/zaibachkhoa/Documents/Music-Genre-Classification-From-Audio-Files/Music_Instrument_Classification/dataset/test/'
    # data_path = '/home/zaibachkhoa/Documents/Music-Genre-Classification-From-Audio-Files/Music_Instrument_Classification/dataset/valid/'
