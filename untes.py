import unittest, json
from unittest.mock import patch
from test import feature_extract, model_listobj  # , predi
import test
import numpy as np
import pandas as pd 


class UNTest(unittest.TestCase):
    def setUp(self):
        # Chạy trước mỗi test method
        self.path = 'unt/input/unt1.wav'  # TODO dir path
        self.exp_jso = 'unt/expect/unt1.json'
        self.exp_csv = 'unt/expect/unt1.csv'
        with open(self.exp_jso, "r", encoding="utf-8") as f:
            self.expejso = json.load(f)
        self.expecsv = pd.read_csv(self.exp_csv , index_col=0)
        # self.expecsv = self.expecsv.astype({col: np.float32 for col in self.expecsv.select_dtypes(include='float64').columns})

    def test_feature_extract(self):
        fea = feature_extract(self.path)
        actual = list()
        for a in fea:
            if isinstance(a, np.ndarray):
                acat = [np.round(a, decimals=6) for a in a.tolist()]
                actual.append(acat)
            else:
                actual.append(a)
        # if not actual == self.expejso:
        #     for i in range(len(actual)):
        #         if not actual[i] == self.expejso[i]:
        #             for ii in range(len(actual[i])):
        #                 if not actual[i][ii] == self.expejso[i][ii]:
        #                     sam = actual[i][ii]
        #                     zon = self.expejso[i][ii]
        #                     print()
        self.assertEqual(actual, self.expejso)

    @patch("test.predi")
    def test_value(self, mock_do):
        clf = model_listobj[0]
        test.main(
            (
                self.path,
            ),
            clf=clf,
        )
        # # kiểm tra hàm do_something được gọi với đúng giá trị
        # mock_do.assert_called_once_with(model_listobj[0], self.expecsv)
        args, kwargs = mock_do.call_args
        # assert args[0] == model_listobj[0]
        pd.testing.assert_frame_equal(
            args[1], self.expecsv,
            check_dtype=False,
            # check_index_type=False
        )

    # def test_add_type_error(self):
    #     with self.assertRaises(TypeError):
    #         add("1", 2)  # kiểu không hợp lệ

if __name__ == "__main__":
    unittest.main()