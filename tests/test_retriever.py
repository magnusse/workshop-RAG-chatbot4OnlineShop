import unittest

from ragshop.Retriever.retriever import load_vectorstore


class MyTestCase(unittest.TestCase):
    def test_load(self):
        collection = load_vectorstore()
        self.assertEqual(collection.count(), 20)  # add assertion here


if __name__ == '__main__':
    unittest.main()
