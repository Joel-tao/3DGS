import unittest
from utils.model_classification import parse_classification,cache_key,model_overrides

class ModelClassificationTests(unittest.TestCase):
    def test_strict_json(self):
        self.assertEqual(parse_classification('{"Object caption":"Asset"}'),'Asset')
        self.assertEqual(parse_classification('```json\n{"Object caption":"Ordinary"}\n```'),'Ordinary')
        for bad in ['Asset','{"Object caption":"Furniture"}','["Asset"]','{"Object caption":"Asset"} trailing']:
            with self.assertRaises(ValueError):parse_classification(bad)
    def test_cache_identity(self):
        base=cache_key('cup','model','revision','prompt')
        self.assertNotEqual(base,cache_key('cup','model','new_revision','prompt'))
        self.assertNotEqual(base,cache_key('cup','model','revision','new_prompt'))
    def test_exact_coverage_and_categories(self):
        nodes=[dict(id=1,category='cup')]
        annotations=dict(objects={'1':dict(category='cup',object_type='Ordinary')})
        self.assertEqual(model_overrides(nodes,annotations)['1']['object_type'],'Ordinary')
        with self.assertRaises(ValueError):model_overrides(nodes,dict(objects={}))
        with self.assertRaises(ValueError):model_overrides(nodes,dict(objects={'1':dict(category='book',object_type='Ordinary')}))

if __name__=='__main__':unittest.main()
