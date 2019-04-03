import unittest
from postingsReader import two_way_merge, SearchBackend


class TestTwoWayMerge(unittest.TestCase):

    def test_basic_two_way_merge(self):
        # Check non empty arrays, without offsets. In this case since no offset is used
        # both (1,3) or (1,2) could be potentially valid
        # the idea is that we only care about doc ids in this mode
        array1, array2 = [(0, 1), (1, 2)], [(1, 3)]
        self.assertEqual(two_way_merge(array1, array2), [(1, 2)])
        self.assertEqual(two_way_merge(array2, array1), [(1, 3)])

        # Now test with offsets
        self.assertEqual(two_way_merge(array1, array2, use_offset=True, offset=10), [])
        self.assertEqual(two_way_merge(array1, array2, use_offset=True, offset=-1), [(1, 2)])
        self.assertEqual(two_way_merge(array2, array1, use_offset=True, offset=1), [(1, 2)])

        # Perform Bounds Value Analysis (BVA)
        # Check empty arrays
        self.assertEqual(two_way_merge([], []), [])
        # One empty array
        self.assertEqual(two_way_merge([], array2), [])
        self.assertEqual(two_way_merge(array1, []), [])
        # Identity
        self.assertEqual(two_way_merge(array1, array1), array1)


class MockPostingsFilePointers:
    def __init__(self):
        self.dict = {
            "I": [(0, 1), (1, 5), (6, 3)],
            "am": [(1, 4), (6, 4)],
            "hungry": [(1, 3), (6, 5)]
        }

    def get_postings_list(self, word):
        if word not in self.dict:
            return []
        return self.dict[word]


class PhraseQueryTest(unittest.TestCase):
    def test_phrase_query(self):
        postings = MockPostingsFilePointers()
        backend = SearchBackend(postings)
        result = backend.phrase_query("I am hungry")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 6)
        result = backend.phrase_query("hungry am I")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)
        result = backend.phrase_query("hungry am")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)
        result = backend.phrase_query("bob has no clue")
        self.assertEqual(len(result), 0)
