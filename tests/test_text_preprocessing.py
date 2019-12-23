import unittest
import movieclassifier.preprocessing.text_preprocessing as tp

class TestTextPreprocessing(unittest.TestCase):

    def setUp(self):
        pass
        
    def test_to_lower(self):
        text = 'This Is nOt HoW I wrIte'
        post = tp.to_lower(text)
        self.assertEqual(post, 'this is not how i write')
    
    def test_remove_specials(self):
        text = '!)($$&!(^!!!there^$%&^$@there?????}>.,)(&<:[]{'
        post = tp.remove_specials(text)
        self.assertEqual(post, 'therethere')

    def test_remove_stopwords(self):
        words = ['the', 'a', 'no', 'an', 'in', 'them', 'me', 'those', \
            'whom', 'had', 'be', 'or', 'not', 'to', 'be']
        post = tp.remove_stopwords(words)
        self.assertEqual(post, [])

    def test_num2words(self):
        nums = ['1', '2', '3', '10', '2020', '42', '9000']
        nums_in_words = ['one', 'two', 'three', 'ten', 'two thousand and twenty', \
            'forty-two', 'nine thousand']
        post = tp.replace_nums2words(nums)

        self.assertEqual(post, nums_in_words)

    def test_lemmatisation(self):
        words = ['cooking', 'went', 'was', 'has', 'studies', 'going']
        should = ['cook', 'go', 'be', 'have', 'study', 'go']
        post = tp.lemmatisation(words)
        self.assertEqual(post, should)

    def test_text_preprocessing(self):
        text = 'James Bond must unmask the mysterious head of the Janus Syndicate \
        and prevent the leader from utilizing the GoldenEye weapons system to \
        inflict devastating revenge on Britain.'
        should = 'james bond must unmask mysterious head janus syndicate prevent leader utilizing goldeneye weapons system inflict devastating revenge britain'
        
        post = tp.process_text(text)
        self.assertEqual(post, should)
