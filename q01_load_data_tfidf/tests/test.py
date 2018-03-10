import unittest
from inspect import getfullargspec
import dill
import sys
from ..build import q01_load_data_tfidf 
import pandas as pd
from pandas.util.testing import assert_frame_equal,assert_series_equal
from numpy.testing import assert_array_equal
from greyatomlib.nlp_day_02_project.q01_load_data_tfidf.build import q01_load_data_tfidf as original

class Testing(unittest.TestCase):
    def setUp(self):

        with open('user_sol.pkl', 'wb') as f:
            dill.dump(q01_load_data_tfidf, f)

        with open('test_sol.pkl', 'wb') as f:
            dill.dump(original, f)
        with open('user_sol.pkl', 'rb') as f:
            self.student_func = dill.load(f)
        with open('test_sol.pkl', 'rb') as f:
            self.solution_func = dill.load(f)
        self.data = 'data/sessions.csv'
        self.student_return = self.student_func(self.data)
        self.original_return = self.solution_func(self.data)
        print('setup')

    #  Check the arguements of the function
    def tests_args(self):
        print(' ')
        print('testing the arguements of the functions')
        print(' ')
        self.args_student = getfullargspec(self.student_func).args
        self.args_original = getfullargspec(self.solution_func).args
        self.assertEqual(len(self.args_student), len(self.args_original),
                         "Expected argument(s) %d, Given %d" % (len(self.args_original), len(self.args_student)))

        # check the defaults of the function

    def tests_defaults(self):
        self.defaults_student = getfullargspec(self.student_func).defaults
        self.defaults_solution = getfullargspec(self.solution_func).defaults
        self.assertEqual(self.defaults_student, self.defaults_solution,
                         "Expected default values do not match given default values")


    def tests_return_dataframe(self):
        assert_frame_equal(self.student_return[0],self.original_return[0],
                           obj="The return values do not match expected values")


    def tests_return_3(self):
        assert_array_equal(self.student_return[1].toarray(), self.original_return[1].toarray(), "The return values do not match expected values")

    def test_return_4(self):
        self.assertListEqual(self.student_return[2], self.original_return[2], "The return values do not match expected values")

if __name__ == '__main__':
    unittest.main() ## Remove this
 